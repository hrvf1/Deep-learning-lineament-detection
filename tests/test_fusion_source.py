"""Contrôles spécifiques contre les fonctions extraites du script fusion original."""
import importlib.util
from pathlib import Path
import numpy as np
from lineaments.preprocessing import extraire_patchs, normaliser_par_canal
from lineaments.augmentation import augmenter_patchs
from lineaments.config import charger_configuration
from lineaments.evaluation import calculer_metriques

spec = importlib.util.spec_from_file_location("fusion_source", Path(__file__).parent / "reference/fusion_functions.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def test_fusion_preparation_source():
    rng = np.random.default_rng(2)
    images = rng.normal(size=(8, 131, 140)).astype(np.float32)
    images[3, 0, 0] = np.nan
    masque = (rng.random((131, 140)) > .9).astype(np.float32)
    nouveau = extraire_patchs(images, masque, 32, 32, exclure_invalides=True)
    ancien = source.extraire_patchs_pack(images, masque, 32, 32)
    for a, b in zip(nouveau[:3], ancien):
        np.testing.assert_array_equal(a, b)
    stats = [(float(np.percentile(nouveau[0][:, i], 2)), float(np.percentile(nouveau[0][:, i], 98))) for i in range(8)]
    normalise = normaliser_par_canal(nouveau[0], stats)
    np.testing.assert_array_equal(normalise, source.normaliser_multi(nouveau[0], stats))
    transforme = augmenter_patchs(normalise, nouveau[1], charger_configuration("mnt_pente_sentinel6")["transformations"])
    original = source.augmenter_patchs_multi(normalise, nouveau[1])
    for a, b in zip(transforme, original):
        np.testing.assert_array_equal(a, b)


def test_fusion_metriques_source():
    p = np.array([.05, .6, .7, .8, .1], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    ancien = source.calculer_metriques(p, y, .5)
    nouveau = calculer_metriques(p, y, .5, mode="fusion")
    correspondance = {"IoU": "IoU", "Dice_F1": "Dice", "Precision": "Précision", "Recall": "Rappel"}
    for original, actuel in correspondance.items():
        np.testing.assert_allclose(ancien[original], nouveau[actuel], rtol=0, atol=0)
