"""Fonctions partagées par les expériences U-Net de cartographie des linéaments.

Ce module ne contient ni chemin absolu, ni configuration propre à une expérience.
Les notebooks conservent la définition de l'architecture U-Net, la boucle
d'entraînement et l'interprétation scientifique de leurs résultats.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:  # Permet d'utiliser les fonctions raster sans PyTorch.
    torch = None
    nn = None
    Dataset = object


def trouver_racine_projet(depart: str | Path | None = None) -> Path:
    """Retrouve la racine contenant ce module et le dossier de données."""
    courant = Path(depart or Path.cwd()).expanduser().resolve()
    for candidat in (courant, *courant.parents):
        if (
            (candidat / "fonctions_utiles.py").is_file()
            and (candidat / "data" / "tichla_data_clean").is_dir()
        ):
            return candidat
    raise FileNotFoundError(
        "Racine du projet introuvable. Ouvrir Jupyter depuis le dossier "
        "Projet_UNet_Lineaments ou définir manuellement PROJECT_ROOT."
    )


def fixer_graine(graine: int = 42) -> None:
    """Fixe les générateurs aléatoires Python, NumPy et PyTorch."""
    random.seed(graine)
    np.random.seed(graine)
    if torch is not None:
        torch.manual_seed(graine)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(graine)


def charger_raster(
    chemin: str | Path,
    dtype=np.float32,
) -> tuple[np.ndarray, dict]:
    """Charge la première bande d'un GeoTIFF et retourne ses métadonnées."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio est requis pour lire les GeoTIFF. "
            "Installer les dépendances avec: pip install -r requirements.txt"
        ) from exc

    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Raster introuvable : {chemin}")

    with rasterio.open(chemin) as source:
        tableau = source.read(1).astype(dtype, copy=False)
        metadata = {
            "path": chemin,
            "shape": tableau.shape,
            "crs": source.crs,
            "transform": source.transform,
            "bounds": source.bounds,
            "nodata": source.nodata,
            "profile": source.profile.copy(),
        }
    return tableau, metadata


def charger_couches(
    chemins: Mapping[str, str | Path],
    dtype=np.float32,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Charge plusieurs rasters et vérifie leur alignement pixel à pixel."""
    couches: dict[str, np.ndarray] = {}
    metadonnees: dict[str, dict] = {}
    for nom, chemin in chemins.items():
        couches[nom], metadonnees[nom] = charger_raster(chemin, dtype=dtype)

    reference = next(iter(metadonnees))
    shape_ref = metadonnees[reference]["shape"]
    crs_ref = metadonnees[reference]["crs"]
    transform_ref = metadonnees[reference]["transform"]

    for nom, tableau in couches.items():
        if tableau.shape != shape_ref:
            raise ValueError(
                f"Dimensions incompatibles : {nom}={tableau.shape}, "
                f"référence={shape_ref}."
            )
        if not np.isfinite(tableau).all():
            raise ValueError(f"Valeurs non finies détectées dans {nom}.")
        if metadonnees[nom]["crs"] != crs_ref:
            raise ValueError(f"SCR incompatible pour {nom}.")
        if metadonnees[nom]["transform"] != transform_ref:
            raise ValueError(f"Grille ou transformée incompatible pour {nom}.")

    return couches, metadonnees


def verifier_masque_binaire(masque: np.ndarray) -> None:
    """Vérifie que le masque contient uniquement les classes 0 et 1."""
    valeurs = np.unique(masque)
    if not set(valeurs.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Le masque n'est pas binaire : valeurs={valeurs}")


def bornes_affichage(tableau: np.ndarray, bas: float = 2, haut: float = 98):
    """Calcule des bornes robustes pour l'affichage d'un raster."""
    valeurs = tableau[np.isfinite(tableau)]
    return tuple(np.percentile(valeurs, (bas, haut)))


def definir_zones_spatiales(
    shape: tuple[int, int],
    fraction_train: float = 0.70,
    gap: int = 0,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Reproduit le découpage spatial gauche/droite de l'expérience pilote."""
    hauteur, largeur = shape
    colonne = int(largeur * fraction_train)
    ligne = int(hauteur * 0.50)
    return {
        "train": {"rows": (0, hauteur), "cols": (0, colonne)},
        "validation": {"rows": (0, ligne), "cols": (colonne + gap, largeur)},
        "test": {"rows": (ligne + gap, hauteur), "cols": (colonne + gap, largeur)},
    }


def extraire_zone(tableau: np.ndarray, zone: Mapping[str, tuple[int, int]]):
    """Extrait une sous-zone décrite par des bornes de lignes et colonnes."""
    r0, r1 = zone["rows"]
    c0, c1 = zone["cols"]
    return tableau[r0:r1, c0:c1]


def extraire_patchs(
    canaux: np.ndarray | Sequence[np.ndarray],
    masque: np.ndarray,
    taille: int = 64,
    stride: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrait des patchs jointifs ou chevauchants au format (N, C, H, W)."""
    if isinstance(canaux, np.ndarray):
        empilement = canaux[np.newaxis, ...] if canaux.ndim == 2 else canaux
    else:
        empilement = np.stack(list(canaux), axis=0)

    if empilement.ndim != 3:
        raise ValueError("Les entrées doivent avoir la forme (C, H, W) ou (H, W).")
    if empilement.shape[1:] != masque.shape:
        raise ValueError(
            f"Entrées {empilement.shape[1:]} et masque {masque.shape} non alignés."
        )
    if taille <= 0 or stride <= 0:
        raise ValueError("taille et stride doivent être strictement positifs.")

    images, cibles, positions = [], [], []
    hauteur, largeur = masque.shape
    for ligne in range(0, hauteur - taille + 1, stride):
        for colonne in range(0, largeur - taille + 1, stride):
            images.append(
                empilement[:, ligne : ligne + taille, colonne : colonne + taille]
            )
            cibles.append(masque[ligne : ligne + taille, colonne : colonne + taille])
            positions.append((ligne, colonne))

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(cibles, dtype=np.float32),
        np.asarray(positions, dtype=np.int32),
    )


def indices_partition_aleatoire(
    nombre: int,
    fraction_train: float = 0.70,
    fraction_validation: float = 0.15,
    graine: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crée un split 70/15/15 reproductible avec NumPy default_rng."""
    if fraction_train + fraction_validation >= 1:
        raise ValueError("La somme train + validation doit être inférieure à 1.")
    indices = np.arange(nombre)
    generateur = np.random.default_rng(graine)
    generateur.shuffle(indices)
    n_train = int(fraction_train * nombre)
    n_validation = int(fraction_validation * nombre)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_validation],
        indices[n_train + n_validation :],
    )


def calculer_stats_normalisation(
    images_train: np.ndarray,
    percentile_bas: float = 2,
    percentile_haut: float = 98,
) -> list[tuple[float, float]]:
    """Calcule les percentiles par canal uniquement sur l'entraînement."""
    images = np.asarray(images_train)
    if images.ndim == 3:
        images = images[:, np.newaxis, :, :]
    if images.ndim != 4:
        raise ValueError("images_train doit avoir la forme (N,C,H,W) ou (N,H,W).")
    return [
        tuple(float(v) for v in np.percentile(images[:, canal], (percentile_bas, percentile_haut)))
        for canal in range(images.shape[1])
    ]


def normaliser_par_canal(
    images: np.ndarray,
    statistiques: Sequence[tuple[float, float]],
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Applique des statistiques fixes à chaque canal sans les recalculer."""
    original = np.asarray(images)
    ajouter_canal = original.ndim == 3
    travail = original[:, np.newaxis, :, :] if ajouter_canal else original
    if travail.ndim != 4 or travail.shape[1] != len(statistiques):
        raise ValueError("Nombre de canaux incompatible avec les statistiques.")

    resultat = np.empty_like(travail, dtype=np.float32)
    for canal, (p_bas, p_haut) in enumerate(statistiques):
        resultat[:, canal] = np.clip(
            (travail[:, canal] - p_bas) / (p_haut - p_bas + epsilon),
            0,
            1,
        )
    return resultat[:, 0] if ajouter_canal else resultat


def normaliser_tableau(
    tableau: np.ndarray,
    p_bas: float,
    p_haut: float,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Normalise un tableau 2D avec des bornes déjà calculées."""
    return np.clip(
        (tableau - p_bas) / (p_haut - p_bas + epsilon), 0, 1
    ).astype(np.float32)


def augmenter_patchs(
    images: np.ndarray,
    masques: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Produit original, 3 rotations et 2 symétries pour chaque patch."""
    images_augmentees, masques_augmentes = [], []
    for image, masque in zip(images, masques):
        for k in range(4):
            images_augmentees.append(np.rot90(image, k, axes=(-2, -1)))
            masques_augmentes.append(np.rot90(masque, k))
        images_augmentees.append(np.flip(image, axis=-1))
        masques_augmentes.append(np.fliplr(masque))
        images_augmentees.append(np.flip(image, axis=-2))
        masques_augmentes.append(np.flipud(masque))

    return (
        np.ascontiguousarray(np.asarray(images_augmentees, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(masques_augmentes, dtype=np.float32)),
    )


def densite_positive_par_patch(masques: np.ndarray) -> np.ndarray:
    """Retourne la proportion de pixels positifs de chaque patch."""
    return masques.reshape(len(masques), -1).mean(axis=1)


def charger_historique(chemin: str | Path) -> dict[str, list[float]]:
    """Charge et contrôle un historique JSON d'entraînement."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Historique introuvable : {chemin}")
    with chemin.open(encoding="utf-8") as flux:
        historique = json.load(flux)

    obligatoires = ("train_loss", "val_loss", "val_iou")
    manquantes = [cle for cle in obligatoires if cle not in historique]
    if manquantes:
        raise KeyError(f"Clés manquantes dans l'historique : {manquantes}")
    longueurs = {cle: len(historique[cle]) for cle in obligatoires}
    if len(set(longueurs.values())) != 1:
        raise ValueError(f"Longueurs incohérentes : {longueurs}")
    return historique


def resumer_historique(historique: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Identifie l'époque de meilleure IoU et les valeurs associées."""
    val_iou = np.asarray(historique["val_iou"], dtype=float)
    indice = int(np.argmax(val_iou))
    return {
        "nombre_epoques": int(len(val_iou)),
        "meilleure_epoque": indice + 1,
        "meilleure_iou_validation": float(val_iou[indice]),
        "val_loss_meilleure_epoque": float(historique["val_loss"][indice]),
        "train_loss_finale": float(historique["train_loss"][-1]),
        "val_loss_finale": float(historique["val_loss"][-1]),
        "val_iou_finale": float(val_iou[-1]),
    }


def tracer_historique(
    historique: Mapping[str, Sequence[float]],
    titre: str,
):
    """Trace les pertes et l'IoU à partir du fichier historique réel."""
    resume = resumer_historique(historique)
    epoques = np.arange(1, resume["nombre_epoques"] + 1)
    meilleure_epoque = resume["meilleure_epoque"]

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epoques, historique["train_loss"], label="Entraînement")
    axes[0].plot(epoques, historique["val_loss"], label="Validation")
    axes[0].axvline(
        meilleure_epoque,
        color="#4056F4",
        linestyle="--",
        label=f"Meilleure IoU : époque {meilleure_epoque}",
    )
    axes[0].set(xlabel="Époque", ylabel="Dice Loss", title="Évolution des pertes")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epoques, historique["val_iou"], color="#138A1A")
    axes[1].scatter(
        [meilleure_epoque],
        [resume["meilleure_iou_validation"]],
        color="#075E0C",
        s=70,
        zorder=5,
        label=f"Maximum : {resume['meilleure_iou_validation']:.4f}",
    )
    axes[1].set(xlabel="Époque", ylabel="IoU", title="IoU de validation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(titre, fontsize=15, fontweight="bold")
    figure.tight_layout()
    return figure, axes


def _exiger_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch est requis pour cette opération. "
            "Installer les dépendances avec: pip install -r requirements.txt"
        )


if torch is not None:

    class SegmentationDataset(Dataset):
        """Dataset PyTorch pour des images (N,C,H,W) et masques (N,H,W)."""

        def __init__(self, images: np.ndarray, masques: np.ndarray):
            images = np.asarray(images)
            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]
            self.images = torch.from_numpy(np.ascontiguousarray(images)).float()
            self.masques = torch.from_numpy(
                np.ascontiguousarray(masques[:, np.newaxis, :, :])
            ).float()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.masques[index]


    class DiceLoss(nn.Module):
        """Dice Loss binaire calculée directement à partir des logits."""

        def __init__(self, epsilon: float = 1e-7):
            super().__init__()
            self.epsilon = epsilon

        def forward(self, logits, cible):
            prediction = torch.sigmoid(logits).reshape(-1)
            cible = cible.reshape(-1)
            intersection = (prediction * cible).sum()
            dice = (2 * intersection + self.epsilon) / (
                prediction.sum() + cible.sum() + self.epsilon
            )
            return 1 - dice


else:

    class SegmentationDataset:
        def __init__(self, *args, **kwargs):
            _exiger_torch()


    class DiceLoss:
        def __init__(self, *args, **kwargs):
            _exiger_torch()


def iou_batch(logits, cible, seuil: float = 0.50, epsilon: float = 1e-7):
    """IoU binaire d'un batch, utilisée pendant l'entraînement."""
    _exiger_torch()
    prediction = (torch.sigmoid(logits) >= seuil).reshape(-1)
    cible = (cible >= 0.5).reshape(-1)
    intersection = torch.logical_and(prediction, cible).sum()
    union = torch.logical_or(prediction, cible).sum()
    return ((intersection + epsilon) / (union + epsilon)).item()


def charger_checkpoint(chemin: str | Path, device="cpu") -> dict:
    """Charge un checkpoint local de confiance avec compatibilité PyTorch."""
    _exiger_torch()
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {chemin}")
    try:
        checkpoint = torch.load(chemin, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(chemin, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Le checkpoint doit être un dictionnaire PyTorch.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Clé model_state_dict absente du checkpoint.")
    return checkpoint


def verifier_checkpoint_historique(
    checkpoint: Mapping,
    historique: Mapping[str, Sequence[float]],
    tolerance: float = 1e-8,
) -> list[str]:
    """Compare l'époque et l'IoU stockées avec le maximum de l'historique."""
    resume = resumer_historique(historique)
    messages = []
    epoque = checkpoint.get("epoch", checkpoint.get("best_epoch"))
    iou = checkpoint.get("val_iou", checkpoint.get("best_iou"))
    if epoque is not None and int(epoque) != resume["meilleure_epoque"]:
        messages.append(
            f"Époque checkpoint={epoque}, historique={resume['meilleure_epoque']}."
        )
    if iou is not None and abs(float(iou) - resume["meilleure_iou_validation"]) > tolerance:
        messages.append(
            "IoU du checkpoint différente du maximum de l'historique : "
            f"{float(iou):.6f} contre {resume['meilleure_iou_validation']:.6f}."
        )
    return messages


@torch.no_grad() if torch is not None else (lambda fonction: fonction)
def predire_loader(modele, loader, device):
    """Calcule les probabilités de tous les patchs d'un DataLoader non mélangé."""
    _exiger_torch()
    modele.eval()
    probabilites, cibles = [], []
    for images, masques in loader:
        logits = modele(images.to(device))
        probabilites.append(torch.sigmoid(logits).cpu().numpy()[:, 0])
        cibles.append(masques.numpy()[:, 0])
    return np.concatenate(probabilites), np.concatenate(cibles)


def metriques_segmentation(
    probabilites: np.ndarray,
    cibles: np.ndarray,
    seuil: float = 0.50,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    """Calcule IoU, Dice, précision et rappel sur tous les pixels."""
    prediction = probabilites >= seuil
    cible = cibles >= 0.5
    vp = np.logical_and(prediction, cible).sum()
    fp = np.logical_and(prediction, ~cible).sum()
    fn = np.logical_and(~prediction, cible).sum()
    return {
        "IoU": float((vp + epsilon) / (vp + fp + fn + epsilon)),
        "Dice": float((2 * vp + epsilon) / (2 * vp + fp + fn + epsilon)),
        "Précision": float((vp + epsilon) / (vp + fp + epsilon)),
        "Rappel": float((vp + epsilon) / (vp + fn + epsilon)),
    }


def table_metriques_patch(
    probabilites: np.ndarray,
    cibles: np.ndarray,
    seuil: float = 0.50,
) -> pd.DataFrame:
    """Calcule les métriques individuellement pour chaque patch."""
    lignes = []
    for index, (proba, cible_valeurs) in enumerate(zip(probabilites, cibles)):
        prediction = proba >= seuil
        cible = cible_valeurs >= 0.5
        vp = int(np.logical_and(prediction, cible).sum())
        fp = int(np.logical_and(prediction, ~cible).sum())
        fn = int(np.logical_and(~prediction, cible).sum())
        union = vp + fp + fn
        denominateur_dice = 2 * vp + fp + fn
        lignes.append(
            {
                "patch": index,
                "pixels_reference": int(cible.sum()),
                "pixels_predits": int(prediction.sum()),
                "IoU": vp / union if union else np.nan,
                "Dice": 2 * vp / denominateur_dice if denominateur_dice else np.nan,
                "Précision": vp / (vp + fp) if (vp + fp) else np.nan,
                "Rappel": vp / (vp + fn) if (vp + fn) else np.nan,
            }
        )
    return pd.DataFrame(lignes).set_index("patch")


def avertir(messages: Iterable[str]) -> None:
    """Affiche une liste d'incohérences sous forme d'avertissements."""
    for message in messages:
        warnings.warn(message, stacklevel=2)
