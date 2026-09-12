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



## Changements d'organisation et contrôles ajoutés

Les chemins personnels et chargements fixes d'anciens modèles sont remplacés par les paramètres du notebook. Après un entraînement, les analyses portent sur le meilleur modèle de **cette exécution**. Le mode modèle enregistré est explicite. Les fichiers d'origine ne sont pas modifiés.

La lecture respecte les pixels NoData déclarés dans les métadonnées raster, les remplace par NaN et les signale. C'est un contrôle plus strict que la lecture brute de certains notebooks ; une source déclarant des valeurs valides comme NoData doit être corrigée avant comparaison. La politique fusion concernant les valeurs invalides du masque est conservée mais n'autorise pas à considérer une zone non annotée comme du fond fiable.

Les états aléatoires, versions, signatures, partitions et statistiques sont enregistrés. La reprise complète et les dossiers uniques sont ajoutés. Les tracés utilisent une convention cohérente, notamment vert/orange/bleu pour VP/FP/FN. Le seuil des widgets est uniquement visuel et ne modifie pas le seuil d'évaluation.

## Limites de reproduction

Les petits tests techniques contrôlent la fidélité des opérations et des architectures, pas la validité scientifique des modèles. Les rasters et checkpoints réels n'ont pas été fournis sous une forme accessible pour cette validation finale. Les 150 époques et les scores historiques ne sont donc pas certifiés ici.

Les différences d'ordre de consommation des générateurs PyTorch et les contrôles déterministes peuvent modifier la trajectoire d'entraînement par rapport à une exécution historique, même avec une graine égale. Les tests vérifient les prédictions à **poids identiques**, ainsi que la répétabilité et la reprise dans l'environnement testé.

Le split aléatoire ne garantit pas l'indépendance spatiale des ensembles. Des rasters ayant un CRS identique ne suffisent pas à garantir une comparabilité scientifique : résolution, unités, calcul des dérivés et conventions d'annotation comptent également.

Cette version part de rasters et masques préparés. Elle ne crée pas d'annotations depuis un fichier vectoriel, ne calcule pas implicitement pente/hillshade, ne télécharge pas Sentinel et ne reconstruit pas de mosaïque GeoTIFF à partir des prédictions. Un shapefile complet ou une couche de polylignes d'un GeoPackage déjà préparé peut toutefois être fourni facultativement pour analyser les longueurs et orientations des annotations vectorielles ; le fichier n'est ni rasterisé ni modifié par le pipeline et n'intervient pas dans l'entraînement. Lorsqu'un GeoPackage possède plusieurs couches, le nom de la couche doit être donné explicitement.
