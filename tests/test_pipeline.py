import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
from lineaments import Experience,charger_configuration
from donnees_synthetiques import creer_donnees_demo
from lineaments.training import charger_checkpoint

torch.set_num_threads(1)


def exp_demo(tmp_path,preset="mnt",taille=128):
    files=creer_donnees_demo(tmp_path/"donnees",taille=taille)
    cfg=charger_configuration(preset)
    cfg.update(taille_patch=32,pas=32,features=[4,8,16,32],epoques=1)
    inputs={e["cle"]:files[e["cle"]] for e in cfg["entrees"]}
    return Experience(cfg,inputs,files["masque"],tmp_path/"sorties",device="cpu"),files


@pytest.mark.parametrize("preset",["mnt","mnt_hillshade315_pente","sentinel6","mnt_pente_sentinel6"])
def test_parcours_complet_et_rechargement(tmp_path,preset):
    exp,files=exp_demo(tmp_path,preset)
    assert exp.diagnostiquer()["alignée"].all()
    exp.preparer();dossier=exp.entrainer();scores=exp.evaluer()
    assert list(scores.index)==["validation","test"]
    probas=exp.predire("test").copy()
    best=charger_checkpoint(dossier/"best.pt")
    assert best["epoch"]==1 and len(best["stats_norm"])==len(best["noms_canaux"])
    applied=Experience.depuis_modele(dossier/"best.pt",exp.entrees,files["masque"],sortie=tmp_path/"relecture",device="cpu")
    if preset=="mnt_pente_sentinel6":assert applied.seuil==exp.seuil
    reloaded=applied.evaluer(reference=True)
    np.testing.assert_array_equal(probas,applied.predire("test"))
    np.testing.assert_array_equal(scores[["IoU","Dice"]],reloaded[["IoU","Dice"]])
    pred=Experience.depuis_modele(dossier/"best.pt",exp.entrees,sortie=tmp_path/"inference",device="cpu")
    assert pred.predire().shape==(16,32,32)
    with pytest.raises(ValueError,match="masque"):pred.evaluer()
    pred.exporter(predictions=True)
    assert (pred.dossier/"probabilites_nouvelle_zone.npz").is_file()
    with pytest.raises(ValueError,match="statistiques"):pred.normaliser()


def test_diagnostic_alignment_et_pas(tmp_path):
    exp,files=exp_demo(tmp_path)
    with rasterio.open(files["masque"],"r+") as dst:dst.transform=from_origin(500010,2500000,10,10)
    assert not bool(exp.diagnostiquer().loc["Masque","alignée"])
    with pytest.raises(ValueError,match="grille"):exp.decouper()
    other,_=exp_demo(tmp_path/"autre")
    other.decouper(taille_patch=64,pas=32)
    with pytest.raises(ValueError,match="chevauchants"):other.repartir()


def test_analyse_lineaments_vectoriels(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    from lineaments.lineament_analysis import analyser_shapefile_lineaments

    chemin = tmp_path / "ligne_test.shp"
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(500020, 2499960), (500060, 2499960)])],
        crs="EPSG:32628",
    ).to_file(chemin)
    resume_simple, traces_simples, _ = analyser_shapefile_lineaments(
        chemin,
        crs_reference="EPSG:32628",
        emprise_reference=(500000, 2499900, 500100, 2500000),
    )
    assert resume_simple.loc["Linéaments analysés", "valeur"] == 1
    assert traces_simples.iloc[0]["longueur"] == pytest.approx(40.0)
    assert traces_simples.iloc[0]["orientation_deg"] == pytest.approx(90.0)

    exp,files = exp_demo(tmp_path / "experience")
    resume, traces, figure = exp.analyser_lineaments(files["lineaments"], pas_angle=5)
    assert resume.loc["Linéaments analysés", "valeur"] == 3
    assert (traces["longueur"] > 0).all()
    assert traces["orientation_deg"].between(0, 180, inclusive="left").all()
    assert set(traces["unite"]) == {"m"}
    assert figure.axes[1].name == "polar"
    exp.exporter()
    assert (exp.dossier / "analyse_lineaments_resume.csv").is_file()
    assert (exp.dossier / "analyse_lineaments_traces.csv").is_file()
    assert (exp.dossier / "figures/analyse_lineaments.png").is_file()


def test_analyse_lineaments_geopackage_et_choix_couche(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    from lineaments.lineament_analysis import analyser_fichier_lineaments

    ligne = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(500020, 2499960), (500060, 2499960)])],
        crs="EPSG:32628",
    )
    chemin_simple = tmp_path / "lineaments_simple.gpkg"
    ligne.to_file(chemin_simple, layer="lineaments", driver="GPKG")
    resume, traces, _ = analyser_fichier_lineaments(chemin_simple)
    assert resume.loc["Linéaments analysés", "valeur"] == 1
    assert traces.iloc[0]["longueur"] == pytest.approx(40.0)

    chemin_multiple = tmp_path / "lineaments_multiple.gpkg"
    ligne.to_file(chemin_multiple, layer="lineaments", driver="GPKG")
    gpd.GeoDataFrame(
        {"id": [2]},
        geometry=[LineString([(500000, 2499900), (500000, 2499930)])],
        crs="EPSG:32628",
    ).to_file(chemin_multiple, layer="contexte", driver="GPKG")

    with pytest.raises(ValueError, match="plusieurs couches.*COUCHE_LINEAMENTS"):
        analyser_fichier_lineaments(chemin_multiple)
    resume, traces, _ = analyser_fichier_lineaments(
        chemin_multiple, couche="contexte"
    )
    assert resume.loc["Linéaments analysés", "valeur"] == 1
    assert traces.iloc[0]["orientation_deg"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Couche GeoPackage inconnue.*lineaments.*contexte"):
        analyser_fichier_lineaments(chemin_multiple, couche="absente")


def test_analyse_lineaments_refuse_couche_pour_shapefile(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    from lineaments.lineament_analysis import analyser_fichier_lineaments

    chemin = tmp_path / "lineaments.shp"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:32628"
    ).to_file(chemin)
    with pytest.raises(ValueError, match="uniquement.*gpkg"):
        analyser_fichier_lineaments(chemin, couche="lineaments")


def test_changement_invalide_resultats(tmp_path):
    exp,_=exp_demo(tmp_path);exp.preparer();exp.entrainer();exp.evaluer()
    old=exp.dossier
    exp.repartir(max_patchs_train=4)
    assert exp.modele is None and exp.stats_norm is None and exp.resultats is None
    assert (old/"best.pt").exists()
    with pytest.raises(RuntimeError):exp.entrainer()


def test_reprise_identique(tmp_path):
    full,_=exp_demo(tmp_path/"continu");full.preparer();full.entrainer(epoques=2)
    split,_=exp_demo(tmp_path/"reprise");split.preparer();folder=split.entrainer(epoques=1)
    split.reprendre(folder/"last.pt",epoques_total=2)
    a=charger_checkpoint(full.dossier/"last.pt");b=charger_checkpoint(split.dossier/"last.pt")
    for key in a["model_state_dict"]:torch.testing.assert_close(a["model_state_dict"][key],b["model_state_dict"][key],rtol=0,atol=0)
    assert a["historique"]["val_iou"]==b["historique"]["val_iou"]


def test_statistiques_modele_pas_recalculees(tmp_path,monkeypatch):
    exp,files=exp_demo(tmp_path);exp.preparer();folder=exp.entrainer()
    import lineaments.experience as module
    monkeypatch.setattr(module,"calculer_stats_normalisation",lambda *args:pytest.fail("Statistiques recalculées en inférence"))
    app=Experience.depuis_modele(folder/"best.pt",exp.entrees,sortie=tmp_path/"app",device="cpu")
    app.predire()


def test_checkpoint_historique_et_ordre_canaux(tmp_path):
    exp,files=exp_demo(tmp_path,"mnt_hillshade315_pente");exp.preparer();folder=exp.entrainer()
    ck=charger_checkpoint(folder/"best.pt")
    legacy={k:ck[k] for k in ("model_state_dict","features","stats_norm","epoch","val_iou","patch_size")}
    path=tmp_path/"ancien.pt";torch.save(legacy,path)
    app=Experience.depuis_modele(path,exp.entrees,experience=exp.configuration,sortie=tmp_path/"old",device="cpu")
    assert app.predire().shape==(16,32,32)
    wrong=exp.configuration;wrong["entrees"][1],wrong["entrees"][2]=wrong["entrees"][2],wrong["entrees"][1]
    with pytest.raises(ValueError,match="canaux"):Experience.depuis_modele(folder/"best.pt",exp.entrees,experience=wrong,sortie=tmp_path/"bad")


def test_reference_donnees_differentes_refusee(tmp_path):
    exp,files=exp_demo(tmp_path);exp.preparer();folder=exp.entrainer()
    with rasterio.open(files["mnt"],"r+") as dst:
        arr=dst.read();arr[0,0,0]+=1;dst.write(arr)
    app=Experience.depuis_modele(folder/"best.pt",exp.entrees,files["masque"],sortie=tmp_path/"app",device="cpu")
    with pytest.raises(ValueError,match="correspondent"):app.evaluer(reference=True)
