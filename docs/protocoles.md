# Correspondance avec les sources

| Expérience | Source fournie | Ordre des canaux |
|---|---|---|
| MNT seul | `02_mnt_protocole_ameliore(1).ipynb` | MNT |
| Relief à trois canaux | `03_mnt_pente_hillshade315(1).ipynb` | MNT, hillshade 315°, pente |
| Sentinel-2 | `04_stack_sentinel2_6bandes(1).ipynb` et `(2)` | B2, B3, B4, B8, B11, B12 |
| Fusion | `unet_mnt_pente_sentinel6_150ep(1).py` | MNT, pente, B2, B3, B4, B8, B11, B12 |

## Paramètres repris

- Patches 64 × 64, pas 64, bordures incomplètes écartées.
- Mélange NumPy `default_rng(42)` ; train 70 %, validation 15 %, test restant ; effectifs arrondis à l'entier inférieur pour train et validation.
- Normalisation par canal, percentiles 2 et 98 calculés uniquement sur le train, limitation dans [0, 1], epsilon 10⁻¹⁰.
- Augmentation ×6 après split : original, rotations 90°/180°/270°, miroirs horizontal/vertical, même ordre et mêmes transformations image/masque.
- U-Net 32–64–128–256 avec BatchNorm, Dice Loss, AdamW, taux d'apprentissage et weight decay à 10⁻⁴, lots de 16, 150 époques sans early stopping.
- Meilleur checkpoint choisi sur l'IoU de validation moyenne par lot à 0,50. L'IoU globale est également enregistrée.

## Particularités conservées

| Point | MNT / relief / Sentinel | Fusion |
|---|---|---|
| Masque | 0/1 requis | Valeurs positives → 1, valeurs invalides → fond, comme dans le script |
| Entrées invalides | Signalées comme erreur | Patches invalides exclus |
| Seuil final | 0,50 | Maximum d'IoU validation, recherche de 0,05 à 0,95 par pas de 0,01 |


