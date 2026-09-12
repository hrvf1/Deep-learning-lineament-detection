"""Découpage et prétraitements NumPy, indépendants des notebooks."""
import numpy as np


def extraire_patchs(images, masque=None, taille=64, pas=64, exclure_invalides=False, limite_memoire_mio=2048):
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3 or taille <= 0 or pas <= 0:
        raise ValueError("Entrée (C,H,W), taille et pas strictement positifs requis.")
    if masque is not None and masque.shape != images.shape[1:]:
        raise ValueError("Le masque et les images doivent partager la même grille.")
    _, h, w = images.shape
    positions = np.array([(r, c) for r in range(0, h-taille+1, pas) for c in range(0, w-taille+1, pas)], dtype=np.int32).reshape(-1, 2)
    if not len(positions):
        raise ValueError(f"Aucun patch {taille} × {taille} dans une image {h} × {w}.")
    mib = len(positions) * taille * taille * (images.shape[0] + (masque is not None)) * 4 / 2**20
    if limite_memoire_mio is not None and mib > limite_memoire_mio:
        raise MemoryError(f"Extraction estimée à {mib:.0f} Mio. Réduire l'emprise/le recouvrement ou ajuster limite_memoire_mio après contrôle de la RAM.")
    xs, ys, gardees, exclues = [], [], [], []
    for r, c in positions:
        x = images[:, r:r+taille, c:c+taille]
        y = None if masque is None else masque[r:r+taille, c:c+taille]
        valide = np.isfinite(x).all() and (y is None or np.isfinite(y).all())
        if not valide:
            if not exclure_invalides:
                raise ValueError(f"Valeur invalide dans le patch ligne={r}, colonne={c}. Consulter diagnostiquer().")
            exclues.append((r, c))
            continue
        xs.append(x); gardees.append((r, c))
        if y is not None:
            ys.append(y)
    if not xs:
        raise ValueError("Tous les patchs contiennent des valeurs invalides.")
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32) if masque is not None else None,
            np.asarray(gardees, dtype=np.int32), np.asarray(exclues, dtype=np.int32).reshape(-1, 2))


def repartir_indices(nombre, proportions=(.70, .15, .15), graine=42, max_patchs_train=None):
    if len(proportions) != 3 or any(p <= 0 for p in proportions) or not np.isclose(sum(proportions), 1):
        raise ValueError("Trois proportions positives de somme 1 sont nécessaires.")
    ordre = np.arange(nombre)
    np.random.default_rng(graine).shuffle(ordre)
    ntrain, nval = int(proportions[0]*nombre), int(proportions[1]*nombre)
    split = {"train": ordre[:ntrain], "validation": ordre[ntrain:ntrain+nval], "test": ordre[ntrain+nval:]}
    if any(len(v) == 0 for v in split.values()):
        raise ValueError("Nombre de patchs insuffisant pour trois ensembles non vides. Réduire leur taille ou modifier les proportions.")
    if max_patchs_train is not None:
        if not isinstance(max_patchs_train, int) or isinstance(max_patchs_train, bool) or max_patchs_train < 1:
            raise ValueError("max_patchs_train doit être un entier positif ou None.")
        if max_patchs_train > ntrain:
            raise ValueError(f"Seulement {ntrain} patchs train disponibles ; utiliser None pour tous.")
        split["train"] = split["train"][:max_patchs_train]
    return split


def calculer_stats_normalisation(images_train, percentiles=(2, 98)):
    if len(images_train) == 0 or not np.isfinite(images_train).all():
        raise ValueError("La normalisation nécessite des données train finies et non vides.")
    return [tuple(float(v) for v in np.percentile(images_train[:, c], percentiles)) for c in range(images_train.shape[1])]


def normaliser_par_canal(images, statistiques):
    if images.ndim != 4 or images.shape[1] != len(statistiques):
        raise ValueError("Les statistiques ne correspondent pas aux canaux d'entrée.")
    resultat = np.empty_like(images, dtype=np.float32)
    for c, (low, high) in enumerate(statistiques):
        resultat[:, c] = np.clip((images[:, c]-low)/(high-low+1e-10), 0, 1)
    return resultat

