"""Métriques explicites : conserver les conventions des deux sources."""
import numpy as np
import pandas as pd
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, cible):
        prediction = torch.sigmoid(logits).reshape(-1)
        cible = cible.reshape(-1)
        intersection = (prediction*cible).sum()
        return 1 - (2*intersection+self.epsilon)/(prediction.sum()+cible.sum()+self.epsilon)


def iou_batch(logits, cible, seuil=.5, mode="notebooks"):
    pred, target = torch.sigmoid(logits) >= seuil, cible >= .5
    inter, union = (pred & target).sum(), (pred | target).sum()
    if mode == "fusion":
        return (inter.item()+1e-7)/(union.item()+1e-7)
    return ((inter+1e-7)/(union+1e-7)).item()


def calculer_metriques(probabilites, cibles, seuil=.5, mode="notebooks"):
    if probabilites.shape != cibles.shape or not probabilites.size:
        raise ValueError("Prédictions et cibles doivent avoir la même forme non vide.")
    if not np.isfinite(probabilites).all() or not np.isfinite(cibles).all():
        raise ValueError("Métriques impossibles sur des valeurs non finies.")
    p, t = probabilites >= seuil, cibles >= .5
    tp, fp, fn, tn = (int(a.sum()) for a in (p & t, p & ~t, ~p & t, ~p & ~t))
    eps, num = (1e-7, 1e-7) if mode == "notebooks" else (1e-8, 0)
    return {"seuil": float(seuil), "IoU": (tp+num)/(tp+fp+fn+eps),
            "Dice": (2*tp+num)/(2*tp+fp+fn+eps), "Précision": (tp+num)/(tp+fp+eps),
            "Rappel": (tp+num)/(tp+fn+eps), "Spécificité": tn/(tn+fp+eps),
            "Exactitude": (tp+tn)/(tp+tn+fp+fn+eps), "VP": tp, "FP": fp, "FN": fn, "VN": tn}


def metriques_par_patch(probabilites, cibles, seuil=.5):
    rows = []
    for i, (p, t) in enumerate(zip(probabilites, cibles)):
        m = calculer_metriques(p, t, seuil)
        tp, fp, fn = m["VP"], m["FP"], m["FN"]
        rows.append({"patch": i, "pixels_reference": int((t >= .5).sum()), "pixels_predits": int((p >= seuil).sum()),
                     "IoU": tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                     "Dice": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                     "Précision": tp/(tp+fp) if tp+fp else np.nan,
                     "Rappel": tp/(tp+fn) if tp+fn else np.nan})
    return pd.DataFrame(rows).set_index("patch")


def analyser_seuils(probabilites, cibles, pas=.05, mode="notebooks"):
    seuils = np.arange(.05, .951, pas)
    return pd.DataFrame([calculer_metriques(probabilites, cibles, float(s), mode) for s in seuils]).set_index("seuil")


def carte_erreurs(probabilites, cible, seuil=.5):
    p, t = probabilites >= seuil, cible >= .5
    out = np.zeros(p.shape, dtype=np.uint8)
    out[p & t] = 1
    out[p & ~t] = 2
    out[~p & t] = 3
    return out
