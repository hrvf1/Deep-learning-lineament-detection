"""Augmentation géométrique : mêmes transformations sur tous les canaux et le masque."""
import numpy as np


def augmenter_patchs(images, masques, transformations):
    """Retourne six versions par patch avec la configuration historique, original inclus."""
    if len(images) != len(masques):
        raise ValueError("Chaque image doit avoir son masque.")
    xs, ys = [], []
    for x, y in zip(images, masques):
        for transformation in transformations:
            if transformation == "original":
                xx, yy = x, y
            elif transformation in ("rotation90", "rotation180", "rotation270"):
                k = {"rotation90": 1, "rotation180": 2, "rotation270": 3}[transformation]
                xx, yy = np.rot90(x, k, axes=(-2, -1)), np.rot90(y, k)
            elif transformation in ("miroir_h", "miroir_v"):
                axe = -1 if transformation == "miroir_h" else -2
                xx, yy = np.flip(x, axis=axe), np.flip(y, axis=axe)
            else:
                raise ValueError(f"Transformation inconnue : {transformation}")
            xs.append(xx)
            ys.append(yy)
    return np.ascontiguousarray(xs, dtype=np.float32), np.ascontiguousarray(ys, dtype=np.float32)
