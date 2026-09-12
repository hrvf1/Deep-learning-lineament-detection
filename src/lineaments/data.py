"""Lecture et diagnostic des rasters ; aucune reprojection implicite."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import rasterio


@dataclass
class Couche:
    nom: str
    valeurs: np.ndarray
    chemin: Path
    bande: int
    meta: dict
    cmap: str = "gray"
    unite: str = "valeur du fichier"


def lire_couche(chemin, bande, nom, cmap="gray", unite="valeur du fichier"):
    chemin = Path(chemin).expanduser().resolve()
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier introuvable pour {nom} : {chemin}")
    with rasterio.open(chemin) as src:
        if bande > src.count:
            raise ValueError(f"{nom} : bande {bande} demandée, seulement {src.count} bande(s).")
        arr = src.read(bande, masked=True).astype(np.float32).filled(np.nan)
        arr[~np.isfinite(arr)] = np.nan
        meta = {"shape": (src.height, src.width), "crs": src.crs, "transform": src.transform,
                "bounds": src.bounds, "res": src.res, "nodata": src.nodata,
                "count": src.count, "profile": src.profile.copy(), "description": src.descriptions[bande-1]}
    return Couche(nom, arr, chemin, bande, meta, cmap, unite)


class Donnees:
    def __init__(self, configuration, entrees, masque=None):
        expected = {e["cle"] for e in configuration["entrees"]}
        if set(entrees) != expected:
            raise ValueError(f"Entrées attendues : {sorted(expected)} ; reçues : {sorted(entrees)}.")
        self.couches = []
        self.configuration = configuration
        for e in configuration["entrees"]:
            for bande, nom in zip(e["bandes"], e["noms"]):
                couche = lire_couche(entrees[e["cle"]], bande, nom, e.get("cmap", "gray"), e.get("unite", "valeur du fichier"))
                if e.get("nombre_bandes_attendu") is not None and couche.meta["count"] != e["nombre_bandes_attendu"]:
                    raise ValueError(f"{e['cle']} doit contenir {e['nombre_bandes_attendu']} bandes dans l'ordre déclaré.")
                self.couches.append(couche)
        self.reference = self.couches[0]
        self.masque_source = None if masque is None else lire_couche(masque, 1, "Masque")
        self.masque = None
        if self.masque_source is not None:
            raw = self.masque_source.valeurs
            self.masque = (np.nan_to_num(raw, nan=0) > 0).astype(np.float32) if configuration["masque"] == "positif" else raw.copy()

    def alignee(self, couche):
        a, b = couche.meta, self.reference.meta
        transforms = np.allclose(tuple(a["transform"]), tuple(b["transform"]), atol=1e-9, rtol=0) if self.configuration["metriques"] == "fusion" else a["transform"] == b["transform"]
        return a["shape"] == b["shape"] and a["crs"] == b["crs"] and transforms

    def diagnostic(self):
        rows = []
        for c in self.couches + ([self.masque_source] if self.masque_source else []):
            finite = c.valeurs[np.isfinite(c.valeurs)]
            rows.append({"couche": c.nom, "fichier": c.chemin.name, "bande": c.bande,
                         "dimensions": str(c.meta["shape"]), "CRS": str(c.meta["crs"]),
                         "résolution": str(c.meta["res"]), "emprise": str(tuple(c.meta["bounds"])),
                         "alignée": self.alignee(c), "NoData déclaré": c.meta["nodata"],
                         "pixels invalides": int(c.valeurs.size-finite.size),
                         "minimum": float(finite.min()) if finite.size else np.nan,
                         "maximum": float(finite.max()) if finite.size else np.nan,
                         "moyenne": float(finite.mean()) if finite.size else np.nan,
                         "constante": bool(finite.size and finite.min() == finite.max())})
        return pd.DataFrame(rows).set_index("couche")

    def problemes(self, exiger_masque=False):
        erreurs = []
        if exiger_masque and self.masque is None:
            erreurs.append("Un masque de référence est nécessaire pour préparer un entraînement ou calculer des métriques.")
        for c in self.couches + ([self.masque_source] if self.masque_source else []):
            if c.meta["crs"] is None:
                erreurs.append(f"{c.nom} : CRS absent ; définir le géoréférencement en amont.")
            if not self.alignee(c):
                erreurs.append(f"{c.nom} : grille différente de {self.reference.nom} ; aligner les rasters en amont.")
            t = c.meta["transform"]
            if abs(t.b) > 1e-12 or abs(t.d) > 1e-12 or t.a <= 0 or t.e >= 0:
                erreurs.append(f"{c.nom} : cette version attend une grille orientée nord, sans rotation.")
            masque_fusion = c is self.masque_source and self.configuration["masque"] == "positif"
            if not masque_fusion and self.configuration["nodata"] == "erreur" and not np.isfinite(c.valeurs).all():
                erreurs.append(f"{c.nom} : valeurs invalides/NoData ; corriger le fichier ou choisir explicitement nodata='exclure_patchs'.")
        if self.masque is not None:
            vals = self.masque[np.isfinite(self.masque)]
            if not np.isin(vals, [0, 1]).all():
                erreurs.append("Le masque doit contenir uniquement 0 et 1. Vérifier son codage avant de poursuivre.")
        return erreurs

    def verifier(self, exiger_masque=False):
        errors = self.problemes(exiger_masque)
        if errors:
            raise ValueError("Diagnostic bloquant :\n- " + "\n- ".join(errors))

    def empiler(self):
        self.verifier()
        return np.stack([c.valeurs for c in self.couches])

    def signature(self, inclure_masque=True):
        cache = {}
        result = []
        for c in self.couches + ([self.masque_source] if inclure_masque and self.masque_source else []):
            if c.chemin not in cache:
                h = hashlib.sha256()
                with c.chemin.open("rb") as f:
                    for bloc in iter(lambda: f.read(1024*1024), b""):
                        h.update(bloc)
                cache[c.chemin] = h.hexdigest()
            result.append({"canal": c.nom, "bande": c.bande, "sha256": cache[c.chemin],
                           "shape": list(c.meta["shape"]), "crs": str(c.meta["crs"]),
                           "transform": list(c.meta["transform"])})
        return result
