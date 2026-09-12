"""Interface guidée, identique pour les quatre expériences."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import warnings
import uuid
import numpy as np
import pandas as pd
from .config import charger_configuration, valider_configuration, noms_canaux
from .data import Donnees
from .preprocessing import extraire_patchs, repartir_indices, calculer_stats_normalisation, normaliser_par_canal
from .augmentation import augmenter_patchs
from .evaluation import calculer_metriques, metriques_par_patch, analyser_seuils
from .utils import ecrire_json, environnement


class Experience:
    """Un jeu de données, un protocole, une exécution ; aucune donnée globale."""
    def __init__(self, experience, entrees, masque=None, sortie="resultats", device="auto"):
        self._config = charger_configuration(experience)
        self.entrees = {k: str(Path(v).expanduser()) for k, v in entrees.items()}
        self.chemin_masque = None if masque is None else str(Path(masque).expanduser())
        self.sortie = Path(sortie).expanduser()
        self.device = device
        self.donnees = None
        self._checkpoint = None
        self._stats_modele = None
        self._source_modele = None
        self._figures = {}
        self._analyse_lineaments = None
        self._vider_preparation()

    def __repr__(self):
        return f"Experience({self._config['nom']!r}, canaux={len(noms_canaux(self._config))}, étape={self.etape!r})"

    @property
    def configuration(self):
        return deepcopy(self._config)

    @property
    def statistiques_normalisation(self):
        statistiques = self._stats_modele if self._checkpoint else self.stats_norm
        if statistiques is None:
            raise RuntimeError("Normaliser les données ou charger un modèle avant de consulter les statistiques.")
        return pd.DataFrame(statistiques, index=noms_canaux(self._config), columns=["percentile_bas", "percentile_haut"])

    def _vider_resultats(self):
        self.modele = None
        self.historique = None
        self.dossier = None
        self.resultats = None
        self.seuil = self._config["seuil"]
        self._probabilites = {}

    def _vider_normalisation(self):
        self.stats_norm = None
        self._normalises = {}
        self._augmente = None
        self._vider_resultats()
        self._figures = {k:v for k,v in self._figures.items() if k.startswith(("donnees_", "analyse_lineaments", "decoupage", "repartition"))}

    def _vider_preparation(self):
        self.images = self.masques = self.positions = self.positions_exclues = None
        self.partitions = None
        self._max_patchs_train = None
        self._vider_normalisation()
        self.etape = "à charger"

    def _modifier(self, **kwargs):
        unknown = set(kwargs) - set(self._config)
        if unknown:
            raise ValueError(f"Paramètres inconnus : {sorted(unknown)}")
        cfg = deepcopy(self._config); cfg.update(kwargs); valider_configuration(cfg)
        self._config = cfg
        if self.donnees is not None:
            self.donnees.configuration = self._config

    def charger(self):
        if self.donnees is None:
            self.donnees = Donnees(self._config, self.entrees, self.chemin_masque)
            self.etape = "données chargées"
        return self

    def diagnostiquer(self):
        self.charger()
        table = self.donnees.diagnostic()
        for msg in self.donnees.problemes():
            print("À corriger :", msg)
        if self._config["masque"] == "positif" and self.donnees.masque is not None:
            print("Protocole fusion : valeurs de masque > 0 converties en 1 ; valeurs invalides converties en fond (0).")
        if not self.donnees.problemes():
            print("Grilles et valeurs compatibles avec la configuration sélectionnée.")
        return table

    def analyser_lineaments(self, chemin_lineaments, pas_angle=5, couche=None):
        """Décrit les polylignes d'un shapefile ou d'une couche GeoPackage."""
        self.charger()
        self.donnees.verifier()
        from .lineament_analysis import analyser_fichier_lineaments
        resume, objets, figure = analyser_fichier_lineaments(
            chemin_lineaments,
            couche=couche,
            crs_reference=self.donnees.reference.meta["crs"],
            emprise_reference=tuple(self.donnees.reference.meta["bounds"]),
            pas_angle=pas_angle,
        )
        self._analyse_lineaments = (resume, objets)
        self._figures["analyse_lineaments"] = figure
        return resume, objets, figure

    def apercu_decoupage(self, taille_patch=None, pas=None):
        self.charger()
        taille = self._config["taille_patch"] if taille_patch is None else taille_patch
        stride = self._config["pas"] if pas is None else pas
        cfg = self.configuration; cfg.update(taille_patch=taille, pas=stride); valider_configuration(cfg)
        h, w = self.donnees.reference.valeurs.shape
        rows, cols = max(0, (h-taille)//stride+1), max(0, (w-taille)//stride+1)
        table = pd.DataFrame([{"taille_patch": taille, "pas": stride, "lignes": rows, "colonnes": cols,
                               "patchs_potentiels": rows*cols, "recouvrement": stride < taille,
                               "bordure_basse_px": h-((rows-1)*stride+taille) if rows else h,
                               "bordure_droite_px": w-((cols-1)*stride+taille) if cols else w,
                               "images_patchs_Mio": rows*cols*taille**2*len(noms_canaux(cfg))*4/2**20}])
        from .visualization import carte_decoupage
        positions = np.array([(r, c) for r in range(0, h-taille+1, stride) for c in range(0, w-taille+1, stride)]).reshape(-1, 2)
        fig = carte_decoupage(self.donnees, positions, taille)
        return table, fig

    def decouper(self, taille_patch=None, pas=None, nodata=None, limite_memoire_mio=2048):
        changes = {k: v for k, v in {"taille_patch": taille_patch, "pas": pas, "nodata": nodata}.items() if v is not None}
        if self._checkpoint and any(k in changes and changes[k] != self._config[k] for k in ("taille_patch", "pas")):
            raise ValueError("Le parcours modèle enregistré conserve son découpage. Créer une nouvelle expérience pour le modifier.")
        self._modifier(**changes)
        self.charger(); self.donnees.verifier()
        self._vider_preparation()
        self._figures = {k:v for k,v in self._figures.items() if k.startswith(("donnees_", "analyse_lineaments"))}
        self.images, self.masques, self.positions, self.positions_exclues = extraire_patchs(
            self.donnees.empiler(), self.donnees.masque, self._config["taille_patch"], self._config["pas"],
            self._config["nodata"] == "exclure_patchs", limite_memoire_mio)
        self.etape = "patchs extraits"
        return pd.DataFrame([{"patchs_retenus": len(self.images), "patchs_exclus": len(self.positions_exclues),
                               "canaux": self.images.shape[1], "taille": self.images.shape[2],
                               "patchs_avec_lineaments": None if self.masques is None else int((self.masques.sum(axis=(1,2))>0).sum())}])

    def repartir(self, proportions=None, graine=None, max_patchs_train=None):
        if self._checkpoint:
            raise ValueError("Un modèle enregistré utilise sa partition de référence ou toute la nouvelle zone ; ne pas refaire un split au hasard.")
        if self.images is None:
            raise RuntimeError("Exécuter decouper() avant repartir().")
        if self.masques is None:
            raise ValueError("Un masque est nécessaire pour entraîner.")
        if self._config["pas"] < self._config["taille_patch"]:
            raise ValueError("Le split aléatoire ne permet pas des patchs chevauchants entre ensembles. Choisir pas ≥ taille_patch. Le split spatial n'est pas inclus dans cette version.")
        self._modifier(**{k: v for k, v in {"proportions": proportions, "graine": graine}.items() if v is not None})
        partitions = repartir_indices(len(self.images), self._config["proportions"], self._config["graine"], max_patchs_train)
        self._vider_normalisation()
        self._figures.pop("repartition", None)
        self.partitions = partitions; self._max_patchs_train = max_patchs_train
        self.etape = "patchs répartis"
        return self.resume_partitions()

    def resume_partitions(self):
        if self.partitions is None:
            raise RuntimeError("Exécuter repartir() avant ce diagnostic.")
        rows = []
        for groupe, idx in self.partitions.items():
            y = self.masques[idx]
            positives = y.reshape(len(y), -1).sum(axis=1)>0
            rows.append({"groupe": groupe, "patchs": len(idx), "avec_lineaments": int(positives.sum()),
                         "sans_lineaments": int((~positives).sum()), "pixels_positifs_pct": float(y.mean()*100)})
        return pd.DataFrame(rows).set_index("groupe")

    def normaliser(self, percentiles=None):
        if self._checkpoint:
            raise ValueError("Les statistiques du modèle sont restaurées automatiquement par predire().")
        if self.partitions is None:
            raise RuntimeError("Exécuter repartir() avant normaliser().")
        if percentiles is not None:
            self._modifier(percentiles=list(percentiles))
        self._vider_normalisation()
        self.stats_norm = calculer_stats_normalisation(self.images[self.partitions["train"]], self._config["percentiles"])
        if self._config["metriques"] == "fusion" and any(hi <= lo for lo, hi in self.stats_norm):
            raise ValueError("Protocole fusion : au moins un canal est constant entre les deux percentiles.")
        self._normalises = {g: normaliser_par_canal(self.images[idx], self.stats_norm) for g, idx in self.partitions.items()}
        self.etape = "données normalisées"
        return pd.DataFrame(self.stats_norm, index=noms_canaux(self._config), columns=["percentile_bas", "percentile_haut"])

    def decrire_modele(self):
        """Décrit l'architecture sans avancer le générateur aléatoire d'entraînement."""
        import torch
        from .model import UNet
        from .training import choisir_device
        with torch.random.fork_rng(devices=[]):
            modele = UNet(in_channels=len(noms_canaux(self._config)), features=self._config["features"])
        return pd.DataFrame([{
            "canaux_entree": len(noms_canaux(self._config)),
            "taille_patch": self._config["taille_patch"],
            "features": str(self._config["features"]),
            "parametres_entrainables": sum(p.numel() for p in modele.parameters() if p.requires_grad),
            "calcul": str(choisir_device(self.device)),
            "batch_size": self._config["batch_size"],
            "perte": "Dice Loss", "optimiseur": "AdamW",
        }])

    def augmenter(self, transformations=None, limite_memoire_mio=2048):
        if not self._normalises:
            raise RuntimeError("Exécuter normaliser() avant augmenter().")
        if transformations is not None:
            self._modifier(transformations=list(transformations))
        xs, ys = self.groupe("train")
        facteur = len(self._config["transformations"])
        mib = facteur*(xs.nbytes+ys.nbytes)/2**20
        if limite_memoire_mio is not None and mib > limite_memoire_mio:
            raise MemoryError(f"Augmentation estimée à {mib:.0f} Mio. Réduire max_patchs_train lors de repartir(), les transformations, ou ajuster la limite RAM.")
        self._vider_resultats()
        self._figures = {k:v for k,v in self._figures.items() if not k.startswith(("predictions", "historique", "augmentation"))}
        self._augmente = augmenter_patchs(xs, ys, self._config["transformations"])
        self.etape = "données augmentées"
        return pd.DataFrame([{"train_original": len(xs), "facteur": facteur, "train_apres_augmentation": len(self._augmente[0]),
                              "validation": len(self.partitions["validation"]), "test": len(self.partitions["test"])}])

    def preparer(self, taille_patch=None, pas=None, proportions=None, graine=None, max_patchs_train=None, percentiles=None, transformations=None):
        self.decouper(taille_patch, pas)
        self.repartir(proportions, graine, max_patchs_train)
        self.normaliser(percentiles)
        self.augmenter(transformations)
        return self

    def groupe(self, nom):
        if nom == "train_augmente":
            if self._augmente is None:
                raise RuntimeError("Exécuter augmenter() pour ce groupe.")
            return self._augmente
        if nom == "nouvelle_zone":
            if self._stats_modele is None or self.images is None:
                raise RuntimeError("Charger un modèle et découper la nouvelle zone.")
            return normaliser_par_canal(self.images, self._stats_modele), self.masques
        if self.partitions is None or nom not in self.partitions:
            raise ValueError("Groupe attendu : train, validation, test ou train_augmente après préparation.")
        idx = self.partitions[nom]
        return self._normalises.get(nom, self.images[idx]), self.masques[idx]

    def _nouveau_dossier(self, suffixe=None):
        name = self._config["nom"] + ("_"+suffixe if suffixe else "")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dossier = self.sortie / f"{name}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.dossier.mkdir(parents=True, exist_ok=False)
        return self.dossier

    def entrainer(self, epoques=None, batch_size=None, lr=None, weight_decay=None, deterministe=True):
        if self._checkpoint:
            raise ValueError("Pour un nouvel entraînement, créer une nouvelle Experience avec ses données et son masque.")
        if self._augmente is None:
            raise RuntimeError("Terminer normaliser() et augmenter() avant entrainer().")
        self._modifier(**{k: v for k, v in {"epoques": epoques, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay}.items() if v is not None})
        from .training import entrainer_modele
        self._vider_resultats(); self._nouveau_dossier()
        self._meta = {"stats_norm": [list(v) for v in self.stats_norm], "noms_canaux": noms_canaux(self._config),
                      "patch_size": self._config["taille_patch"], "stride": self._config["pas"],
                      "positions_all": self.positions.tolist(), "partitions": {k: v.tolist() for k, v in self.partitions.items()},
                      "signature_donnees": self.donnees.signature(), "max_patchs_train": self._max_patchs_train,
                      "environnement": environnement()}
        ecrire_json(self.dossier/"configuration.json", self._config)
        ecrire_json(self.dossier/"preparation.json", self._meta)
        self.donnees.diagnostic().to_csv(self.dossier/"diagnostic.csv")
        self.modele, self.historique = entrainer_modele(self._augmente, self.groupe("validation"), self._config, self.dossier, self._meta, self.device, deterministe=deterministe)
        self.etape = "entraînement terminé"
        return self.dossier

    def reprendre(self, modele_last, epoques_total, confiance=False):
        from .training import charger_checkpoint, choisir_device, entrainer_modele
        ck = charger_checkpoint(modele_last, confiance)
        if not all(k in ck for k in ("rng", "config", "signature_donnees", "partitions", "historique")):
            raise ValueError("Reprise complète disponible pour les nouveaux last.pt ; ancien modèle utilisable en inférence.")
        if self._augmente is None:
            raise RuntimeError("Préparer les mêmes données avant reprendre().")
        if ck["signature_donnees"] != self.donnees.signature() or ck["partitions"] != {k: v.tolist() for k, v in self.partitions.items()}:
            raise ValueError("Les données ou partitions diffèrent de l'entraînement à reprendre.")
        cfg = self.configuration; original = deepcopy(ck["config"])
        cfg.pop("epoques"); original.pop("epoques")
        if cfg != original or not np.array_equal(self.stats_norm, ck["stats_norm"]):
            raise ValueError("Les paramètres de préparation/entraînement diffèrent du checkpoint.")
        if choisir_device(self.device).type != ck["device_type"]:
            raise ValueError("Reprendre sur le même type de calcul (CPU ou CUDA).")
        actuel = environnement()
        if any(ck["environnement"].get(k) != actuel.get(k) for k in ("python", "torch", "numpy")):
            raise ValueError("La reprise contrôlée nécessite les mêmes versions Python, PyTorch et NumPy.")
        if epoques_total <= ck["epoch"]:
            raise ValueError("epoques_total doit dépasser l'époque sauvegardée.")
        dossier = Path(modele_last).parent
        if not (dossier/"best.pt").is_file():
            raise FileNotFoundError("Conserver best.pt à côté de last.pt pour reprendre.")
        self._modifier(epoques=epoques_total)
        self._vider_resultats(); self.dossier = dossier
        meta = {k: ck[k] for k in ("stats_norm", "noms_canaux", "patch_size", "stride", "positions_all", "partitions", "signature_donnees", "max_patchs_train", "environnement")}
        self.modele, self.historique = entrainer_modele(self._augmente, self.groupe("validation"), self._config, dossier, meta, self.device, ck, ck["deterministe"])
        ecrire_json(dossier/"configuration.json", self._config)
        self.etape = "entraînement repris"
        return dossier

    @classmethod
    def depuis_modele(cls, modele, entrees, masque=None, experience=None, sortie="resultats", device="auto", confiance=False, statistiques=None):
        from .training import charger_checkpoint, creer_modele_checkpoint
        ck = charger_checkpoint(modele, confiance)
        if "config" not in ck and experience is None:
            raise ValueError("Ancien checkpoint : préciser experience pour connaître l'ordre des canaux et le protocole.")
        cfg = charger_configuration(ck.get("config", experience))
        if experience is not None and "config" in ck and noms_canaux(charger_configuration(experience)) != noms_canaux(cfg):
            raise ValueError("L'expérience demandée ne correspond pas aux canaux du modèle.")
        cfg["taille_patch"] = int(ck.get("patch_size", cfg["taille_patch"]))
        cfg["pas"] = int(ck.get("stride", cfg["pas"]))
        cfg["features"] = list(ck.get("features", cfg["features"]))
        obj = cls(cfg, entrees, masque, sortie, device)
        obj._checkpoint = ck
        obj._source_modele = str(Path(modele).resolve())
        stats = ck.get("stats_norm", statistiques)
        if stats is None:
            raise ValueError("Statistiques d'entraînement absentes. Fournir statistiques ; ne pas les recalculer sur la nouvelle zone.")
        obj._stats_modele = [tuple(float(x) for x in pair) for pair in stats]
        if len(obj._stats_modele) != len(noms_canaux(cfg)) or not np.isfinite(obj._stats_modele).all() or any(hi < lo for lo, hi in obj._stats_modele):
            raise ValueError("Statistiques de normalisation incompatibles avec l'entrée du modèle.")
        if "noms_canaux" in ck:
            # Ancien script fusion emploie des noms descriptifs, équivalents aux noms de bandes.
            aliases = {"Pente": "Pente", "B2_Bleu": "B2", "B3_Vert": "B3", "B4_Rouge": "B4", "B8_NIR": "B8", "B11_SWIR1": "B11", "B12_SWIR2": "B12"}
            stored = [aliases.get(n, n) for n in ck["noms_canaux"]]
            if stored != noms_canaux(cfg):
                raise ValueError("L'ordre des canaux ne correspond pas au modèle enregistré.")
        obj.charger(); obj.decouper()
        obj.modele, count, features = creer_modele_checkpoint(ck, device)
        if count != len(noms_canaux(cfg)) or features != cfg["features"]:
            raise ValueError("Architecture et configuration incompatibles.")
        obj._nouveau_dossier("application")
        obj.historique = ck.get("historique")
        historique = Path(modele).parent / "historique.json"
        if obj.historique is None and historique.is_file():
            import json
            obj.historique = json.loads(historique.read_text(encoding="utf-8"))
        obj.seuil = float(ck.get("seuil_selectionne_validation", cfg["seuil"]))
        # Le seuil optimisé est stocké séparément après l'évaluation d'un entraînement.
        evaluation = Path(modele).parent/"evaluation.json"
        if evaluation.is_file() and ck.get("format_version") == 1:
            import json
            info = json.loads(evaluation.read_text(encoding="utf-8"))
            if info.get("epoque_modele") == ck.get("epoch"):
                obj.seuil = float(info["seuil_retenu"])
        if "format_version" not in ck:
            warnings.warn("Modèle historique : ordre/protocole fourni par l'expérience ; la reproduction du split doit être vérifiée avec les données d'origine.")
        return obj

    def predire(self, groupe=None):
        from .training import predire
        if self.modele is None:
            raise RuntimeError("Entraîner ou charger un modèle avant predire().")
        groupe = groupe or ("nouvelle_zone" if self._checkpoint else "test")
        if groupe not in self._probabilites:
            self._probabilites[groupe] = predire(self.modele, self.groupe(groupe)[0], self._config["batch_size"], self.device)
        return self._probabilites[groupe]

    def evaluer(self, seuil=None, reference=False):
        if self.modele is None:
            raise RuntimeError("Entraîner ou charger un modèle avant evaluer().")
        if self.masques is None:
            raise ValueError("Un masque est nécessaire pour calculer les métriques. Utiliser predire() pour des images seules.")
        table_seuils = None
        if self._checkpoint:
            if reference:
                ck = self._checkpoint
                if "signature_donnees" not in ck or "partitions" not in ck:
                    raise ValueError("Ce checkpoint ancien ne mémorise pas la partition d'origine. Utiliser reference=False pour évaluer la zone fournie ; ce résultat ne certifie pas le test historique.")
                if ck["signature_donnees"] != self.donnees.signature() or ck["positions_all"] != self.positions.tolist():
                    raise ValueError("Les données ou positions ne correspondent pas au test de référence.")
                self.partitions = {k: np.asarray(v, dtype=np.int64) for k, v in ck["partitions"].items()}
                self._normalises = {k: normaliser_par_canal(self.images[v], self._stats_modele) for k, v in self.partitions.items()}
                groupes = ["validation", "test"]
            else:
                groupes = ["nouvelle_zone"]
        else:
            groupes = ["validation", "test"]
            table_seuils = analyser_seuils(self.predire("validation"), self.groupe("validation")[1], self._config["pas_seuil"], self._config["metriques"])
            if seuil is None and self._config["optimiser_seuil"]:
                self.seuil = float(table_seuils["IoU"].idxmax())
        if seuil is not None:
            if not 0 < seuil < 1:
                raise ValueError("Le seuil doit être compris entre 0 et 1.")
            self.seuil = float(seuil)
        rows = {}
        for groupe in groupes:
            probas, y = self.predire(groupe), self.groupe(groupe)[1]
            rows[groupe] = calculer_metriques(probas, y, self.seuil, self._config["metriques"])
            metriques_par_patch(probas, y, self.seuil).to_csv(self.dossier/f"metriques_patch_{groupe}.csv")
        self.resultats = pd.DataFrame.from_dict(rows, orient="index")
        self.resultats.to_csv(self.dossier/"metriques.csv")
        if table_seuils is not None:
            table_seuils.to_csv(self.dossier/"seuils_validation.csv")
        from .training import charger_checkpoint
        ck = self._checkpoint or charger_checkpoint(self.dossier/"best.pt")
        info = {"seuil_retenu": self.seuil, "epoque_modele": ck.get("epoch"),
                "mode": "reference" if reference or not self._checkpoint else "nouvelle_zone",
                "convention_metriques": self._config["metriques"], "resultats": rows,
                "test_au_seuil_0_5": calculer_metriques(self.predire("test"), self.groupe("test")[1], .5, self._config["metriques"]) if "test" in groupes else None}
        ecrire_json(self.dossier/"evaluation.json", info)
        self.etape = "évaluation terminée"
        return self.resultats

    def exporter(self, predictions=False):
        if self.dossier is None:
            self._nouveau_dossier("diagnostic")
        ecrire_json(self.dossier/"configuration.json", self._config)
        ecrire_json(self.dossier/"environnement.json", environnement())
        if self.stats_norm is not None or self._stats_modele is not None:
            self.statistiques_normalisation.to_csv(self.dossier / "normalisation.csv")
        if self._source_modele is not None:
            import hashlib
            digest = hashlib.sha256()
            with Path(self._source_modele).open("rb") as fichier:
                for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
                    digest.update(bloc)
            ecrire_json(self.dossier / "modele_source.json", {
                "fichier": self._source_modele, "sha256": digest.hexdigest(),
                "epoque": self._checkpoint.get("epoch"), "seuil": self.seuil,
            })
        if self.donnees:
            self.donnees.diagnostic().to_csv(self.dossier/"diagnostic.csv")
            meta=self.donnees.reference.meta
            ecrire_json(self.dossier/"grille.json", {"crs":str(meta["crs"]),"transform":list(meta["transform"]),"shape":meta["shape"],"canaux":noms_canaux(self._config)})
        if self._analyse_lineaments is not None:
            resume, objets = self._analyse_lineaments
            resume.to_csv(self.dossier / "analyse_lineaments_resume.csv")
            objets.to_csv(self.dossier / "analyse_lineaments_traces.csv")
        if self.positions is not None:
            np.savez_compressed(self.dossier/"positions.npz", positions=self.positions, exclues=self.positions_exclues)
        if predictions:
            for groupe, probas in self._probabilites.items():
                if groupe == "train_augmente":
                    transformations = self._config["transformations"]
                    positions = np.repeat(self.positions[self.partitions["train"]], len(transformations), axis=0)
                    noms = np.tile(transformations, len(self.partitions["train"]))
                else:
                    positions = self.positions if groupe == "nouvelle_zone" else self.positions[self.partitions[groupe]]
                    noms = np.full(len(positions), "original")
                np.savez_compressed(self.dossier/f"probabilites_{groupe}.npz", probabilites=probas, positions=positions, seuil=self.seuil, transformations=noms)
        if self._figures:
            dest=self.dossier/"figures";dest.mkdir(exist_ok=True)
            for nom,fig in self._figures.items():
                fig.savefig(dest/f"{nom}.png",dpi=160,bbox_inches="tight")
        return self.dossier

    def visualiser_donnees(self, superposer_masque=True):
        self.charger()
        from .visualization import visualiser_donnees
        figs=visualiser_donnees(self.donnees, superposer_masque)
        self._figures.update({f"donnees_{i:02d}":f for i,f in enumerate(figs)})
        return figs

    def visualiser_decoupage(self, repartition=False):
        if self.positions is None:
            raise RuntimeError("Exécuter decouper() avant cette visualisation.")
        from .visualization import carte_decoupage
        fig=carte_decoupage(self.donnees, self.positions, self._config["taille_patch"], self.masques,
                          self.partitions if repartition else None, self.positions_exclues)
        self._figures["repartition" if repartition else "decoupage"]=fig
        return fig

    def visualiser_patchs(self, groupe="train", nombre=8):
        from .visualization import visualiser_patchs
        fig=visualiser_patchs(self, groupe, nombre);self._figures[f"patchs_{groupe}"]=fig;return fig

    def visualiser_normalisation(self, canal=None, index=0):
        from .visualization import visualiser_normalisation
        fig=visualiser_normalisation(self, canal, index);self._figures["normalisation"]=fig;return fig

    def visualiser_augmentation(self, index=0):
        from .visualization import visualiser_augmentation
        fig=visualiser_augmentation(self, index);self._figures["augmentation"]=fig;return fig

    def visualiser_historique(self):
        if self.historique is None:
            raise RuntimeError("Aucun historique chargé pour cette exécution.")
        from .visualization import visualiser_historique
        fig=visualiser_historique(self);self._figures["historique"]=fig;return fig

    def visualiser_seuils(self):
        """Sensibilité calculée sur la validation, jamais optimisée sur le test."""
        if self.dossier is None or not (self.dossier / "seuils_validation.csv").is_file():
            raise RuntimeError("La courbe nécessite evaluer() après un entraînement de cette exécution.")
        from .visualization import visualiser_seuils
        table = pd.read_csv(self.dossier / "seuils_validation.csv", index_col="seuil")
        fig = visualiser_seuils(table, self.seuil)
        self._figures["seuils_validation"] = fig
        return fig

    def explorer_predictions(self, groupe=None):
        from .visualization import explorer_predictions
        return explorer_predictions(self, groupe)

    def visualiser_predictions(self, groupe=None, indices=None, seuil=None, enregistrer=False):
        from .visualization import visualiser_predictions
        groupe = groupe or ("nouvelle_zone" if self._checkpoint else "test")
        fig=visualiser_predictions(self, groupe, indices, seuil, enregistrer);self._figures[f"predictions_{groupe}"]=fig;return fig
