# Validation technique

Validation locale du 12 septembre 2026. Aucun résultat scientifique ONHYM n'est calculé à partir des données artificielles des tests.

## Vérifications réalisées

- Installation du paquet en mode editable depuis `pyproject.toml`.
- Validation du format des quatre notebooks et compilation de toutes leurs cellules Python.
- Comparaison du découpage, des partitions, des percentiles, de la normalisation et des augmentations avec `fonctions_utiles.py`, pour 1, 3, 6 et 8 canaux.
- Comparaison spécifique des traitements et métriques avec les fonctions extraites du script fusion.
- Chargement strict des poids et comparaison des sorties de l'U-Net avec l'architecture notebook de référence pour les quatre nombres de canaux.
- Entraînement court, sauvegarde, rechargement et reproduction des prédictions à poids identiques.
- Reprise d'un entraînement interrompu : poids identiques à une exécution continue dans l'environnement testé.
- Contrôles d'alignement, patches invalides, limites du train et statistiques conservées en inférence.
- Analyse de fichiers vectoriels synthétiques : shapefile complet, GeoPackage à couche unique, sélection explicite dans un GeoPackage multicouche, longueurs et orientations axiales.
- Exécution séquentielle des cellules des quatre notebooks en mode entraînement, puis en mode modèle enregistré avec leur partition de test sauvegardée.
- Exécution du notebook MNT en prédiction sans masque.
- Navigation entre groupes dans l'explorateur, y compris retour d'un grand ensemble à un plus petit.
- Export des probabilités et des positions des patches augmentés avec leurs transformations.
- Configuration des liens GitHub/Colab sur une copie temporaire ; aucun dépôt distant modifié.
- Inspection visuelle d'une figure de prédictions et carte d'erreurs générée par les tests.

Les cellules des notebooks sont exécutées dans un espace de noms Python partagé, dans leur ordre réel. Les seuls remplacements portent sur les paramètres d'entrée : rasters artificiels, CPU, patches 32, petit U-Net et une époque. L'installation est vérifiée séparément et n'est pas relancée dans chaque test. Les paramètres scientifiques de livraison restent ceux des sources.

## Environnement du contrôle

| Composant | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| Rasterio | 1.5.1 |
| GeoPandas | 1.1.4 |
| Shapely | 2.1.2 |
| Pyogrio | 0.13.0 |
| ipywidgets | 8.1.9 |
| nbformat | 5.11.1 |
| pytest | 9.1.1 |

Le programme enregistre les versions de chaque nouvelle exécution dans `environnement.json`. Les plages de dépendances du projet sont des plages d'installation ; toutes leurs combinaisons n'ont pas été testées.

## À vérifier avec les fichiers réels

Les rasters originaux et les checkpoints historiques réels n'étaient pas accessibles pour cette validation. L'entraînement de 150 époques et les scores du rapport n'ont pas été reproduits. Les tests sur des checkpoints historiques sont des contrôles de format reconstitué, pas une certification des fichiers de modèles du stage.

L'exécution dans l'interface hébergée Google Colab, le rendu des widgets dans un navigateur Colab/VS Code et l'entraînement GPU restent à vérifier dans ces environnements. Les cellules, calculs et sorties ont été contrôlés localement.

Les avertissements de dépréciation d'Affine/Rasterio dans l'environnement testé ne sont pas des erreurs de calcul. Le chargement d'un checkpoint historique signale explicitement les limites de reconstitution de son ancien split.

Pour relancer les contrôles : installer les dépendances de développement puis exécuter `python -m pytest` à la racine du projet. La suite contient 31 tests.
