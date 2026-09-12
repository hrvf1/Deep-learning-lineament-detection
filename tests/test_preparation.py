import importlib.util
from pathlib import Path
import numpy as np
import pytest
from lineaments.preprocessing import extraire_patchs,repartir_indices,calculer_stats_normalisation,normaliser_par_canal
from lineaments.augmentation import augmenter_patchs
from lineaments.config import charger_configuration
from lineaments.data import controler_chemins

spec=importlib.util.spec_from_file_location("source_utiles",Path(__file__).parent/"reference"/"fonctions_utiles.py")
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)


def test_controle_chemins_libres(tmp_path):
    mnt = tmp_path / "mon_dem_avec_un_nom_libre.tif"
    masque = tmp_path / "annotations_zone_sud.tiff"
    mnt.touch(); masque.touch()
    table, erreurs = controler_chemins(
        {"mnt": mnt}, masque=masque, sortie=tmp_path / "sorties_libres"
    )
    assert not erreurs
    assert table.loc["Entrée — mnt", "statut"] == "prêt"
    assert table.loc["Masque", "statut"] == "prêt"
    assert table.loc["Dossier de sortie", "statut"] == "sera créé"


def test_controle_chemins_signale_absence_et_format(tmp_path):
    table, erreurs = controler_chemins(
        {"mnt": tmp_path / "absent.tif"},
        masque=tmp_path / "masque.png",
        sortie=None,
    )
    assert table.loc["Entrée — mnt", "statut"] == "introuvable"
    assert table.loc["Masque", "statut"] == "format inattendu"
    assert len(erreurs) == 3


@pytest.mark.parametrize("canaux",[1,3,6,8])
def test_parite_sources(canaux):
    rng=np.random.default_rng(123)
    images=rng.normal(size=(canaux,199,207)).astype(np.float32)
    masque=(images[0]>.9).astype(np.float32)
    x,y,pos,_=extraire_patchs(images,masque,32,32)
    a,b,p=source.extraire_patchs(images,masque,32,32)
    np.testing.assert_array_equal(x,a);np.testing.assert_array_equal(y,b);np.testing.assert_array_equal(pos,p)
    split=repartir_indices(len(x));s=source.indices_partition_aleatoire(len(x))
    for key,ids in zip(("train","validation","test"),s):np.testing.assert_array_equal(split[key],ids)
    stats=calculer_stats_normalisation(x[split["train"]]);original=source.calculer_stats_normalisation(x[s[0]])
    np.testing.assert_array_equal(stats,original)
    xn=normaliser_par_canal(x,stats);np.testing.assert_array_equal(xn,source.normaliser_par_canal(x,stats))
    xa,ya=augmenter_patchs(xn[s[0]],y[s[0]],charger_configuration("mnt")["transformations"])
    aa,bb=source.augmenter_patchs(xn[s[0]],y[s[0]])
    np.testing.assert_array_equal(xa,aa);np.testing.assert_array_equal(ya,bb)


def test_limite_train_ne_change_pas_validation_test():
    entier=repartir_indices(100);limite=repartir_indices(100,max_patchs_train=20)
    assert len(limite["train"])==20
    for key in ("validation","test"):np.testing.assert_array_equal(entier[key],limite[key])
    assert not set(limite["train"]) & set(limite["test"])


def test_cas_limites_et_exclusion():
    with pytest.raises(ValueError,match="Aucun patch"):extraire_patchs(np.zeros((1,20,20)),taille=64)
    with pytest.raises(ValueError):repartir_indices(3)
    with pytest.raises(ValueError):repartir_indices(100,max_patchs_train=-1)
    with pytest.raises(ValueError):repartir_indices(100,max_patchs_train=71)
    x=np.ones((1,128,128),dtype=np.float32);x[0,2,2]=np.nan
    with pytest.raises(ValueError,match="invalide"):extraire_patchs(x)
    patches,y,pos,exclues=extraire_patchs(x,exclure_invalides=True)
    assert len(patches)==3 and y is None
    np.testing.assert_array_equal(exclues,[[0,0]])
