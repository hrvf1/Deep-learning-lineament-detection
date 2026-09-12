# Réglages et commandes du pipeline

## Utilisation courante

Modifier la cellule « Vos fichiers et paramètres » du notebook sélectionné. Chaque entrée possède son propre chemin : aucun dossier ni nom de fichier n'est imposé. Toutes les cellules suivantes utilisent ces valeurs. Pour changer une préparation, relancer à partir de cette cellule : les données préparées et résultats en mémoire seront recalculés.

| Appel dans le notebook | Résultat |
|---|---|
| `exp.diagnostiquer()` | Tableau des grilles et des valeurs, avec problèmes à corriger |
| `exp.analyser_lineaments(CHEMIN_LINEAMENTS, couche=COUCHE_LINEAMENTS)` | Longueurs et orientations des polylignes d'un shapefile ou d'une couche GeoPackage |
| `exp.visualiser_donnees()` | Couches, masque, superposition, RGB si disponible |
| `exp.apercu_decoupage()` | Nombre potentiel de patches, bordures et carte |
| `exp.decouper()` | Extraction synchronisée des images et masques |
| `exp.repartir()` | Partition train/validation/test et tableau des effectifs |
| `exp.visualiser_decoupage(repartition=True)` | Carte des ensembles |
| `exp.visualiser_patchs(groupe=..., nombre=...)` | Exemples d'un ensemble |
| `exp.normaliser()` | Bornes par canal calculées sur le train |
| `exp.visualiser_normalisation(canal=..., index=...)` | Avant/après et histogrammes |
| `exp.augmenter()` | Transformations du train uniquement |
| `exp.visualiser_augmentation(index=...)` | Versions transformées du patch et du masque |
| `exp.decrire_modele()` | Architecture, nombre de paramètres et matériel |
| `exp.entrainer()` | Dossier de résultats et meilleur modèle chargé |
| `exp.visualiser_historique()` | Courbes des pertes et IoU |
| `exp.evaluer()` | Métriques et exports CSV/JSON |
| `exp.visualiser_seuils()` | Courbe de sensibilité sur la validation |
| `exp.visualiser_predictions(...)` | Entrée, masque, probabilités, prédictions, erreurs |
| `exp.explorer_predictions()` | Interface interactive de sélection des exemples |
| `exp.exporter(predictions=True)` | Paramètres, figures, positions et probabilités calculées |

## Choisir librement les fichiers

Dans Colab, choisir `UTILISER_DRIVE=True` pour monter Google Drive, puis renseigner les chemins complets commençant généralement par `/content/drive/MyDrive/`. Avec des fichiers chargés directement dans la session, utiliser leurs chemins sous `/content/`. Sur ordinateur, conserver `UTILISER_DRIVE=False` et indiquer des chemins locaux absolus ou relatifs.

Les variables proposées dépendent de l'expérience : `CHEMIN_MNT`, `CHEMIN_PENTE`, `CHEMIN_HILLSHADE`, `CHEMIN_SENTINEL`, `CHEMIN_MASQUE`, puis `DOSSIER_SORTIE`. Elles acceptent les noms et dossiers choisis par l'utilisateur. `CHEMIN_MODELE` n'est requis qu'en mode `modele_enregistre` et `CHEMIN_LINEAMENTS` uniquement lorsque l'analyse vectorielle est activée.

La cellule d'initialisation affiche un tableau récapitulatif. `prêt` indique un fichier trouvé avec une extension attendue ; `sera créé` concerne un dossier de sortie encore absent. Les statuts `non renseigné`, `introuvable` ou `format inattendu` bloquent la suite avec un message ciblé. Le diagnostic géospatial détaillé intervient ensuite.

## Données attendues

MNT, pente et hillshade : GeoTIFF à une bande. Sentinel : GeoTIFF à six bandes, B2/B3/B4/B8/B11/B12. Respecter les unités, résolutions et traitements de l'expérience choisie pour une comparaison scientifique. L'échelle spatiale vue par le réseau vaut taille du patch × résolution du pixel : un patch 64 à 10 m et un patch 64 à 30 m ne couvrent pas la même zone.

L'entraînement nécessite des annotations couvrant la zone et définissant le fond connu. Pour les trois premiers protocoles, un masque 0/1 est attendu. La fusion reprend la conversion des valeurs positives en 1. Si le fichier masque déclare `NoData=0` alors que 0 désigne réellement le fond annoté, corriger ses métadonnées en amont : les pixels inconnus et le fond connu doivent être distingués.

Le diagnostic ne calcule ni annotations ni dérivés topographiques ; il ne rééchantillonne pas automatiquement les données. Une résolution ou une grille incompatible doit être corrigée en amont. Le CRS, les dimensions et la transformation des pixels sont contrôlés avant extraction.

L'analyse descriptive des linéaments est facultative et s'applique au fichier vectoriel original, pas au masque raster ni aux prédictions. Le fichier peut se trouver dans n'importe quel dossier. Renseigner soit le chemin d'un shapefile complet (`.shp`, `.shx`, `.dbf`, `.prj`, et éventuellement `.cpg`, avec le même nom de base), soit celui d'un GeoPackage `.gpkg`. Régler ensuite `ACTIVER_ANALYSE_LINEAMENTS=True` et faire pointer `CHEMIN_LINEAMENTS` vers ce chemin exact.

`COUCHE_LINEAMENTS=None` convient au shapefile et à un GeoPackage contenant une seule couche, qui est alors choisie automatiquement. Si le GeoPackage contient plusieurs couches, donner leur nom exact, par exemple `COUCHE_LINEAMENTS="lineaments_interpretes"`. Sans ce choix, le programme refuse l'ambiguïté et affiche la liste des couches disponibles. `COUCHE_LINEAMENTS` doit rester à `None` avec un shapefile.

La couche choisie doit contenir uniquement des polylignes (`LineString` ou `MultiLineString`), avoir le même CRS que les rasters et recouper leur emprise.

Chaque polyligne est mesurée dans l'unité du CRS. Avec un CRS projeté métrique, les longueurs sont donc exprimées en mètres. L'orientation est un azimut axial compris entre 0° et 180°, mesuré depuis le nord dans le sens horaire ; la rose représente le nombre de linéaments par classe angulaire. Une géométrie `MultiLineString` est décomposée en plusieurs polylignes. L'analyse n'est jamais utilisée comme entrée du U-Net et peut rester désactivée sans modifier le découpage, l'entraînement, l'évaluation ou la prédiction.

## Paramètres avancés

Le fichier JSON de l'expérience permet aussi de régler `features`, `nodata`, `seuil`, `optimiser_seuil` et `pas_seuil`. Modifier le JSON crée une variante de l'expérience ; conserver les configurations originales pour comparer au protocole du stage. Les valeurs explicites de la cellule de paramètres priment sur celles du JSON pour l'entraînement.

`nodata="exclure_patchs"` exclut les patches contenant des valeurs invalides. Le choix par défaut des trois premiers notebooks est `erreur` pour signaler la préparation à corriger. Les bordures incomplètes ne sont pas complétées artificiellement. Tous les patches affectés au train sont utilisés. Les limites mémoire sont des estimations par opération, pas la consommation totale du programme.

Le split aléatoire original est conservé. Un pas inférieur à la taille est accepté pour l'aperçu et l'extraction, mais refusé lors d'une nouvelle répartition d'entraînement. Un découpage spatial indépendant avec zone tampon demanderait un protocole distinct.

## Reprendre un entraînement interrompu

Conserver `last.pt` et `best.pt` dans le même dossier. Refaire les étapes 1 à 8 avec les mêmes fichiers, réglages et environnement que l'entraînement interrompu, puis remplacer l'appel d'entraînement de l'étape 9 par :

```python
dossier_resultats = exp.reprendre(
    "/chemin/vers/le/dossier/last.pt",
    epoques_total=150,
)
```

La reprise utilise l'état de l'optimiseur, les générateurs aléatoires et les partitions enregistrées. Elle vérifie les signatures des fichiers, la préparation, les versions Python/NumPy/PyTorch et le type CPU/CUDA. Le nombre total d'époques doit dépasser l'époque sauvegardée. Cette reprise complète concerne les `last.pt` produits par ce projet ; les anciens checkpoints sans ces informations servent à la prédiction/évaluation.

## Retrouver les résultats

Chaque nouvel entraînement crée un dossier distinct avec :

- `best.pt`, `last.pt` : meilleur et dernier modèle, configuration et normalisation incluses ;
- `configuration.json`, `preparation.json` : réglages, positions, partitions, signatures des données et statistiques ;
- `historique.json` : pertes, IoU et temps par époque ;
- `diagnostic.csv`, `grille.json`, `positions.npz`, `environnement.json` : données et environnement ;
- `analyse_lineaments_resume.csv`, `analyse_lineaments_traces.csv` : statistiques du fichier vectoriel de référence lorsqu'elles ont été demandées ;
- `metriques.csv`, `metriques_patch_*.csv`, `seuils_validation.csv`, `evaluation.json` : scores et seuil ;
- `figures/` : cartes, courbes et exemples sélectionnés ;
- `probabilites_*.npz` : probabilités par patch, coordonnées ligne/colonne et transformations.

En mode modèle enregistré, le dossier d'application contient les prédictions et diagnostics, sans copie automatique du modèle source. Certaines sorties d'entraînement n'existent donc pas. `normalisation.csv` conserve les bornes utilisées ; `modele_source.json` identifie le checkpoint par son chemin et son empreinte SHA-256.

Pour reproduire une version du code, conserver son commit/tag GitHub, le checkpoint, les paramètres et les mêmes fichiers d'entrée. L'identité des résultats entre matériels et versions différentes n'est pas garantie.
