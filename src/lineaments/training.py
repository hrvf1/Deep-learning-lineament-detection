"""Entraînement, reprise et inférence sans dépendance à une cellule globale."""
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .model import UNet
from .evaluation import DiceLoss, iou_batch


class SegmentationDataset(Dataset):
    def __init__(self, images, masques):
        self.images = torch.from_numpy(np.ascontiguousarray(images)).float()
        self.masques = torch.from_numpy(np.ascontiguousarray(masques[:, None])).float()
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return self.images[index], self.masques[index]


def fixer_graine(graine, deterministe=True):
    random.seed(graine); np.random.seed(graine); torch.manual_seed(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministe
    torch.use_deterministic_algorithms(deterministe)


def choisir_device(device="auto"):
    choix = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "auto":
        choix = device
    if str(choix).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("GPU CUDA indisponible. Choisir CPU ou activer un GPU dans Colab.")
    return torch.device(choix)


def sauvegarder_checkpoint(chemin, contenu):
    chemin = Path(chemin)
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    torch.save(contenu, temporaire)
    os.replace(temporaire, chemin)


def charger_checkpoint(chemin, confiance=False):
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")
    try:
        ckpt = torch.load(chemin, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not confiance:
            raise ValueError("Checkpoint ancien contenant des objets Python. Pour votre propre modèle de confiance, utiliser confiance=True.") from exc
        ckpt = torch.load(chemin, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Le fichier doit contenir un dictionnaire avec model_state_dict.")
    return ckpt


def creer_modele_checkpoint(ckpt, device="auto"):
    poids = ckpt["model_state_dict"]
    try:
        canaux = poids["encoder.0.conv.0.weight"].shape[1]
        features = [poids[f"encoder.{i}.conv.0.weight"].shape[0] for i in range(32) if f"encoder.{i}.conv.0.weight" in poids]
    except KeyError as exc:
        raise ValueError("Ce checkpoint ne correspond pas à l'architecture U-Net des sources.") from exc
    # La construction d'un modèle d'inférence ne modifie pas le RNG d'un entraînement.
    with torch.random.fork_rng(devices=[]):
        model = UNet(in_channels=canaux, features=features)
    model.load_state_dict(poids, strict=True)
    return model.to(choisir_device(device)).eval(), canaux, list(features)


@torch.inference_mode()
def predire(model, images, batch_size=16, device="auto"):
    device = choisir_device(device)
    model.to(device).eval()
    result = []
    for debut in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[debut:debut+batch_size])).float().to(device)
        result.append(torch.sigmoid(model(x)).cpu().numpy()[:, 0])
    if not result:
        raise ValueError("Aucune image à prédire.")
    return np.concatenate(result)


def etats_aleatoires(generateur):
    npstate = np.random.get_state()
    return {"python": random.getstate(), "numpy": [npstate[0], npstate[1].tolist(), npstate[2], npstate[3], npstate[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generateur.get_state()}


def restaurer_aleatoire(etats, generateur):
    random.setstate(etats["python"])
    n = etats["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(etats["torch"])
    if etats["cuda"]:
        if len(etats["cuda"]) != torch.cuda.device_count():
            raise ValueError("La reprise exacte nécessite le même nombre de GPU.")
        torch.cuda.set_rng_state_all(etats["cuda"])
    generateur.set_state(etats["loader"])


def entrainer_modele(train, validation, config, dossier, meta, device="auto", reprise=None, deterministe=True):
    device = choisir_device(device)
    fixer_graine(config["graine"], deterministe)
    dossier = Path(dossier); dossier.mkdir(parents=True, exist_ok=True)
    generateur = torch.Generator().manual_seed(config["graine"])
    train_loader = DataLoader(SegmentationDataset(*train), batch_size=config["batch_size"], shuffle=True, generator=generateur, num_workers=0)
    # Générateur séparé : la validation n'avance pas le générateur global de l'entraînement.
    val_loader = DataLoader(SegmentationDataset(*validation), batch_size=config["batch_size"], shuffle=False,
                            generator=torch.Generator().manual_seed(config["graine"]), num_workers=0)
    model = UNet(in_channels=train[0].shape[1], features=config["features"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    critere = DiceLoss()
    hist = {"train_loss": [], "val_loss": [], "val_iou": [], "val_iou_global": [], "temps_epoque": []}
    best_iou, best_epoch, start = -float("inf"), 0, 0
    if reprise is not None:
        model.load_state_dict(reprise["model_state_dict"], strict=True)
        optim.load_state_dict(reprise["optimizer_state_dict"])
        hist = reprise["historique"]
        best_iou, best_epoch, start = reprise["best_iou"], reprise["best_epoch"], reprise["epoch"]
        restaurer_aleatoire(reprise["rng"], generateur)
    for epoch in range(start+1, config["epoques"]+1):
        t0 = time.perf_counter(); model.train(); losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = critere(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Perte non finie ; vérifier les données et paramètres.")
            loss.backward(); optim.step(); losses.append(loss.item())
        model.eval(); vals, ious = [], []; tp = fp = fn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device); logits = model(x)
                vals.append(critere(logits, y).item())
                ious.append(iou_batch(logits, y, .5, config["metriques"]))
                p, t = torch.sigmoid(logits) >= .5, y >= .5
                tp += (p & t).sum().item(); fp += (p & ~t).sum().item(); fn += (~p & t).sum().item()
        values = [float(np.mean(losses)), float(np.mean(vals)), float(np.mean(ious)), tp/(tp+fp+fn+1e-8), time.perf_counter()-t0]
        for key, value in zip(hist, values):
            hist[key].append(value)
        improved = values[2] > best_iou
        if improved:
            best_iou, best_epoch = values[2], epoch
        ckpt = {**meta, "format_version": 1, "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(), "val_iou": values[2], "val_loss": values[1],
                "val_iou_global": values[3], "best_iou": best_iou, "best_epoch": best_epoch,
                "features": config["features"], "in_channels": train[0].shape[1], "historique": hist,
                "config": config, "device_type": device.type, "deterministe": deterministe, "rng": etats_aleatoires(generateur)}
        if improved:
            sauvegarder_checkpoint(dossier/"best.pt", ckpt)
        sauvegarder_checkpoint(dossier/"last.pt", ckpt)
        tmp = dossier/"historique.json.tmp"
        tmp.write_text(json.dumps(hist, indent=2), encoding="utf-8"); os.replace(tmp, dossier/"historique.json")
        print(f"Époque {epoch:03d}/{config['epoques']} | perte train {values[0]:.4f} | perte val {values[1]:.4f} | IoU val par lot {values[2]:.4f} | {values[4]:.1f} s")
    if not (dossier/"best.pt").is_file():
        raise ValueError("Aucun meilleur modèle disponible dans ce dossier de reprise.")
    best = charger_checkpoint(dossier/"best.pt")
    model.load_state_dict(best["model_state_dict"]); model.eval()
    return model, hist
