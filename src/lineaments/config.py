"""Configurations explicites : protocole historique et paramètres personnalisables."""
from copy import deepcopy
import json
from pathlib import Path
import sysconfig

PRESETS = ("mnt", "mnt_hillshade315_pente", "sentinel6", "mnt_pente_sentinel6")


def charger_configuration(experience):
    if isinstance(experience, dict):
        config = deepcopy(experience)
    else:
        if str(experience) in PRESETS:
            racine = Path(__file__).resolve().parents[2] / "configs"
            if not racine.is_dir():
                racine = Path(sysconfig.get_path("data")) / "share/onhym-lineaments/configs"
            texte = (racine / f"{experience}.json").read_text(encoding="utf-8")
        else:
            texte = Path(experience).read_text(encoding="utf-8")
        config = json.loads(texte)
    valider_configuration(config)
    return config


def valider_configuration(c):
    required = {"nom", "entrees", "taille_patch", "pas", "graine", "proportions", "percentiles", "transformations", "features", "epoques", "batch_size", "lr", "weight_decay", "metriques", "seuil", "optimiser_seuil", "pas_seuil", "nodata", "masque"}
    if required - c.keys():
        raise ValueError(f"Configuration incomplète : {sorted(required - c.keys())}")
    noms = []
    for e in c["entrees"]:
        if not e["bandes"] or len(e["bandes"]) != len(e["noms"]):
            raise ValueError("Chaque bande d'entrée doit avoir un nom.")
        if any(not isinstance(b, int) or b < 1 for b in e["bandes"]):
            raise ValueError("Les numéros de bandes commencent à 1.")
        noms.extend(e["noms"])
    if not noms or len(set(noms)) != len(noms):
        raise ValueError("Les canaux doivent avoir des noms uniques.")
    for key in ("taille_patch", "pas", "batch_size", "epoques"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f"{key} doit être un entier strictement positif.")
    if not c["features"] or any(not isinstance(f, int) or f <= 0 for f in c["features"]):
        raise ValueError("features doit contenir des entiers positifs.")
    facteur = 2 ** len(c["features"])
    if c["taille_patch"] < facteur * 2 or c["taille_patch"] % facteur:
        raise ValueError(f"Choisir une taille de patch multiple de {facteur}, au moins {facteur * 2}, pour cet U-Net.")
    p = c["proportions"]
    if len(p) != 3 or any(x <= 0 for x in p) or abs(sum(p) - 1) > 1e-9:
        raise ValueError("Les trois proportions doivent être positives et de somme 1.")
    low, high = c["percentiles"]
    if not 0 <= low < high <= 100:
        raise ValueError("Percentiles attendus : 0 ≤ bas < haut ≤ 100.")
    allowed = {"original", "rotation90", "rotation180", "rotation270", "miroir_h", "miroir_v"}
    if not c["transformations"] or c["transformations"][0] != "original" or len(set(c["transformations"])) != len(c["transformations"]) or set(c["transformations"]) - allowed:
        raise ValueError("Transformations uniques, commençant par original, requises.")
    if c["lr"] <= 0 or c["weight_decay"] < 0:
        raise ValueError("lr doit être positif et weight_decay non négatif.")
    if not 0 < c["seuil"] < 1 or not 0 < c["pas_seuil"] <= .9:
        raise ValueError("Seuil ou pas de recherche invalide.")
    if c["nodata"] not in ("erreur", "exclure_patchs") or c["masque"] not in ("binaire", "positif") or c["metriques"] not in ("notebooks", "fusion"):
        raise ValueError("Politique NoData, masque ou métriques inconnue.")


def noms_canaux(config):
    return [nom for e in config["entrees"] for nom in e["noms"]]


def experiences_disponibles():
    import pandas as pd
    rows = []
    for key in PRESETS:
        c = charger_configuration(key)
        rows.append({"experience": key, "canaux": len(noms_canaux(c)), "ordre": ", ".join(noms_canaux(c))})
    return pd.DataFrame(rows).set_index("experience")
