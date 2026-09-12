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
- Contrôle des chemins d'entrée, de masque, de vecteur, de modèle et de sortie sans imposer de nom de fichier ni d'arborescence.
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

