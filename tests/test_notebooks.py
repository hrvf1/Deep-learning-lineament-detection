"""Exécute les vraies cellules sur de petits rasters ; aucun fichier utilisateur requis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch
from lineaments import charger_configuration
from donnees_synthetiques import creer_donnees_demo

RACINE = Path(__file__).resolve().parents[1]
CAS = [
    ("01_mnt_seul.ipynb", "mnt"),
    ("02_mnt_hillshade_pente.ipynb", "mnt_hillshade315_pente"),
    ("03_sentinel2_6bandes.ipynb", "sentinel6"),
    ("04_fusion_mnt_pente_sentinel2.ipynb", "mnt_pente_sentinel6"),
]
torch.set_num_threads(1)


def executer_cellules(notebook, reglages, monkeypatch):
    """Même espace de noms et même ordre que Jupyter ; les cellules sont en Python standard."""
    monkeypatch.chdir(RACINE)
    ns = {}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "installation" in cell.metadata.get("tags", []):
            # L'installation editable est contrôlée séparément, hors réseau pendant les tests.
            ligne = 'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=RACINE)'
            assert ligne in source
            source = source.replace(ligne, 'print("Dépendances déjà installées pour ce test.")')
        exec(compile(source, f"notebook-cellule-{i}", "exec"), ns)
        if "parametres" in cell.metadata.get("tags", []):
            ns.update(reglages)
        plt.close("all")
    return ns


@pytest.mark.parametrize("nom,preset", CAS)
def test_cellules_entrainement_et_modele_enregistre(tmp_path, monkeypatch, nom, preset):
    nb = nbformat.read(RACINE / "notebooks" / nom, as_version=4)
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells)
    source = "\n".join(cell.source for cell in nb.cells)
    assert "ACTIVER_ANALYSE_LINEAMENTS = False" in source
    assert "COUCHE_LINEAMENTS = None" in source
    assert "couche=COUCHE_LINEAMENTS" in source
    assert "MAX_PATCHS_TRAIN" not in source
    fichiers = creer_donnees_demo(tmp_path / "donnees", taille=128)
    cfg = charger_configuration(preset)
    cfg["features"] = [4, 8, 16, 32]  # Architecture par défaut testée séparément à poids identiques.
    entrees = {e["cle"]: fichiers[e["cle"]] for e in cfg["entrees"]}
    reglages = dict(CONFIGURATION=cfg, MODE="entrainer", ENTREES=entrees,
        MASQUE=fichiers["masque"], DOSSIER_SORTIE=tmp_path / "sorties",
        TAILLE_PATCH=32, PAS=32, EPOQUES=1, BATCH_SIZE=4, NOMBRE_PATCHS_AFFICHES=1,
        ACTIVER_EXPLORATEUR=True, ACTIVER_ANALYSE_LINEAMENTS=True,
        CHEMIN_LINEAMENTS=(fichiers["lineaments_gpkg"] if preset == "mnt" else fichiers["lineaments"]),
        COUCHE_LINEAMENTS=None, DEVICE="cpu")
    ns = executer_cellules(nb, reglages, monkeypatch)
    exp = ns["exp"]
    nombre_train_augmente = len(exp.partitions["train"]) * len(exp.configuration["transformations"])
    assert exp.groupe("train_augmente")[0].shape == (
        nombre_train_augmente, len(exp.statistiques_normalisation), 32, 32
    )
    assert exp.resultats.index.tolist() == ["validation", "test"]
    dossier = exp.dossier
    assert (dossier / "figures/seuils_validation.png").is_file()
    assert (dossier / "best.pt").is_file()
    test_probas = exp.predire("test").copy()

    # Le widget doit naviguer entre un grand et un petit ensemble sans erreur d'index.
    menu = ns["explorateur"].children[0]
    index = ns["explorateur"].children[1].children[1]
    menu.value = "train_augmente"
    index.value = index.max
    menu.value = "validation"
    assert index.value == 0

    # L'exploration du train augmenté doit aussi rester exportable et traçable.
    exp.exporter(predictions=True)
    export = np.load(dossier / "probabilites_train_augmente.npz")
    assert len(export["positions"]) == len(export["transformations"]) == nombre_train_augmente

    # Seconde exécution du même notebook : récupérer son véritable test sauvegardé.
    reglages.update(MODE="modele_enregistre", CHEMIN_MODELE=dossier / "best.pt", TEST_REFERENCE=True)
    charge = executer_cellules(nb, reglages, monkeypatch)["exp"]
    np.testing.assert_array_equal(test_probas, charge.predire("test"))
    np.testing.assert_array_equal(exp.resultats[["IoU", "Dice"]], charge.resultats[["IoU", "Dice"]])


def test_notebook_prediction_sans_masque(tmp_path, monkeypatch):
    from lineaments import Experience
    fichiers = creer_donnees_demo(tmp_path / "donnees", taille=128)
    cfg = charger_configuration("mnt")
    cfg.update(features=[4, 8, 16, 32], taille_patch=32, pas=32, epoques=1)
    exp = Experience(cfg, {"mnt": fichiers["mnt"]}, fichiers["masque"], tmp_path / "train")
    exp.preparer()
    dossier = exp.entrainer()
    nb = nbformat.read(RACINE / "notebooks/01_mnt_seul.ipynb", as_version=4)
    reglages = dict(CONFIGURATION=cfg, MODE="modele_enregistre", CHEMIN_MODELE=dossier / "best.pt",
        ENTREES={"mnt": fichiers["mnt"]}, MASQUE=None, DOSSIER_SORTIE=tmp_path / "application",
        DEVICE="cpu", ACTIVER_EXPLORATEUR=True)
    ns = executer_cellules(nb, reglages, monkeypatch)
    assert ns["exp"].resultats is None
    assert ns["probabilites"].shape == (16, 32, 32)
    assert (ns["destination"] / "probabilites_nouvelle_zone.npz").is_file()
