# Cartographie des linéaments par apprentissage profond

**Étude comparative de données topographiques et Sentinel-2 avec U-Net — secteur de Tichla, Maroc**

J’ai réalisé ce projet dans le cadre de mon stage de fin d’année en Génie Minéral à l’École Mohammadia d’Ingénieurs, au sein de l’Office National des Hydrocarbures et des Mines (ONHYM). J’y ai étudié l’apport de l’apprentissage profond à la cartographie des linéaments, en comparant plusieurs combinaisons de données topographiques et satellitaires.

Mon objectif était de comprendre quelles informations permettent au modèle de mieux identifier les structures linéaires dans un contexte saharien à faible relief. J’ai construit une chaîne de segmentation avec U-Net, depuis la préparation des données et des annotations jusqu’à l’évaluation quantitative et à l’examen des prédictions.

Ce dépôt rassemble les quatre configurations d’entrée retenues pour la comparaison. J’ai regroupé leurs traitements dans des fonctions communes et organisé les notebooks pour rendre la démarche consultable, reproductible et réutilisable sur d’autres jeux de données.

## Contexte et question de recherche

Les linéaments sont des formes linéaires observables à la surface du terrain, susceptibles de traduire des discontinuités géologiques : fractures, failles ou zones de cisaillement. Leur cartographie contribue à l’analyse structurale et à l’orientation de l’exploration minière. Leur interprétation sur des images satellitaires ou des modèles de terrain demande toutefois du temps et reste dépendante de l’expérience de l’opérateur.

L’étude porte sur le **secteur de Tichla, dans la région de Dakhla-Oued Eddahab**, au sud du Maroc. Le relief peu marqué de cette zone pose une difficulté particulière : certaines structures peuvent être peu visibles dans la topographie, mais se distinguer par leur réponse spectrale.

J’ai donc cherché à répondre à trois questions :

- Le MNT suffit-il à fournir une information exploitable pour segmenter les linéaments ?
- L’ajout de la pente et de l’ombrage, ou l’utilisation de Sentinel-2, améliore-t-il leur identification ?
- La combinaison des informations topographiques et spectrales apporte-t-elle un gain par rapport à leur utilisation séparée ?

## Données et expériences

L’étude s’appuie sur le **MNT Copernicus GLO-30**, des images **Sentinel-2 L2A** et des annotations de référence issues d’une digitalisation manuelle. Le problème est formulé comme une segmentation binaire : le réseau estime, pour chaque pixel, la probabilité d’appartenir à un linéament.

| Expérience | Entrées du modèle, dans l’ordre des canaux | Question étudiée |
|---|---|---|
| [01 — MNT seul](notebooks/01_mnt_seul.ipynb) | MNT — **1 canal** | Évaluer l’information portée par le relief seul. |
| [02 — MNT, ombrage et pente](notebooks/02_mnt_hillshade_pente.ipynb) | MNT, hillshade 315°, pente — **3 canaux** | Examiner l’apport des dérivés topographiques. |
| [03 — Sentinel-2](notebooks/03_sentinel2_6bandes.ipynb) | B2, B3, B4, B8, B11, B12 — **6 canaux** | Évaluer l’apport des informations spectrales. |
| [04 — Fusion topographique et spectrale](notebooks/04_fusion_mnt_pente_sentinel2.ipynb) | MNT, pente, B2, B3, B4, B8, B11, B12 — **8 canaux** | Tester la complémentarité des deux sources d’information. |

Les bandes Sentinel retenues couvrent le visible, le proche infrarouge et l’infrarouge à ondes courtes. L’ombrage est utilisé dans la deuxième expérience ; il ne fait pas partie de la fusion à huit canaux.

## Démarche expérimentale

J’ai commencé par une expérience pilote afin d’identifier les difficultés de préparation et de généralisation du modèle. J’ai ensuite fait évoluer le protocole avant de comparer les quatre configurations d’entrée présentées ici. Le dépôt reprend ces quatre expériences ; le pilote constitue une étape préalable de l’étude.

La comparaison repose sur une architecture U-Net commune et sur les mêmes réglages principaux de préparation et d’entraînement. Les différences propres aux expériences, notamment le traitement des valeurs invalides et le choix du seuil final, sont conservées et détaillées dans les [protocoles expérimentaux](docs/protocoles.md).

| Étape | Traitement |
|---|---|
| Contrôle des données | Vérification des dimensions, du CRS, de la résolution, de l’emprise, de l’alignement et des valeurs invalides. |
| Analyse des annotations | Analyse facultative d'un shapefile ou GeoPackage : longueurs des polylignes et rose des orientations par effectifs. |
| Découpage | Extraction conjointe des images et des masques en patches de 64 × 64 pixels, sans chevauchement par défaut. |
| Répartition | Ensembles d’entraînement, de validation et de test selon les proportions 70/15/15, avec une graine fixée à 42. |
| Normalisation | Percentiles 2 et 98 calculés sur le train uniquement, puis appliqués aux trois ensembles. |
| Augmentation | Six versions par patch d’entraînement : original, trois rotations et deux miroirs, appliqués simultanément aux canaux et au masque. |
| Apprentissage | U-Net, Dice Loss, optimiseur AdamW, lots de 16 et entraînement configuré sur 150 époques. |
| Évaluation | Sélection du meilleur modèle sur l’IoU de validation moyenne par lot, puis calcul des métriques et analyse visuelle des prédictions. |

L’évaluation associe l’**IoU**, le **Dice/F1**, la **précision** et le **rappel** à une lecture des cartes de probabilités et des erreurs. Cette analyse qualitative permet d’examiner la continuité des tracés, les omissions et les détections supplémentaires que les scores globaux ne décrivent pas à eux seuls.

## Utiliser le projet

Chaque expérience dispose de son propre notebook, avec une cellule de paramètres et des explications Markdown au-dessus des traitements. Il est possible d’entraîner un modèle avec de nouvelles données ou de charger un checkpoint compatible pour obtenir des prédictions.

### Préparer les entrées

Les notebooks prennent en entrée des **GeoTIFF déjà préparés et alignés**. Le stack Sentinel doit contenir les six bandes dans l’ordre indiqué plus haut. Les couches de pente et d’ombrage, lorsqu’elles sont nécessaires, doivent également être fournies.

Un **masque de linéaments** est nécessaire pour entraîner et pour calculer les métriques. Avec un modèle enregistré, la prédiction seule peut être réalisée sans masque. Les données et checkpoints historiques ne sont pas distribués dans ce dépôt.

Le **fichier vectoriel des linéaments est facultatif**. Il sert uniquement à reproduire l'analyse descriptive des annotations : nombre de linéaments, longueurs, statistiques et rose des orientations. Pour utiliser cette option, placer **au choix** dans `data/ma_zone` :

- un shapefile complet : `lineaments.shp`, `lineaments.shx`, `lineaments.dbf` et `lineaments.prj`, tous avec le même nom de base ; `lineaments.cpg` peut être ajouté s'il existe ;
- ou un GeoPackage : `lineaments.gpkg`, qui tient dans un seul fichier.

Choisir ensuite `ACTIVER_ANALYSE_LINEAMENTS=True` et faire pointer `CHEMIN_LINEAMENTS` vers le fichier `.shp` ou `.gpkg`. Pour un shapefile ou un GeoPackage à une seule couche, conserver `COUCHE_LINEAMENTS=None`. Si le GeoPackage contient plusieurs couches, indiquer dans `COUCHE_LINEAMENTS` le nom exact de la couche de linéaments ; sinon le programme s'arrête en affichant les couches disponibles.

Dans les deux formats, la couche choisie doit contenir uniquement des polylignes (`LineString` ou `MultiLineString`), posséder le même CRS que les rasters et recouper leur emprise.

Cette analyse n'intervient ni dans le découpage, ni dans l'entraînement, ni dans l'évaluation du U-Net. Sans shapefile ni GeoPackage, laisser l'option à `False` et exécuter normalement le reste du pipeline.

### Exécuter dans Google Colab

<!-- COLAB:START -->
- [01_mnt_seul](https://colab.research.google.com/github/hrvf1/Deep-learning-lineament-detection/blob/main/notebooks/01_mnt_seul.ipynb)
- [02_mnt_hillshade_pente](https://colab.research.google.com/github/hrvf1/Deep-learning-lineament-detection/blob/main/notebooks/02_mnt_hillshade_pente.ipynb)
- [03_sentinel2_6bandes](https://colab.research.google.com/github/hrvf1/Deep-learning-lineament-detection/blob/main/notebooks/03_sentinel2_6bandes.ipynb)
- [04_fusion_mnt_pente_sentinel2](https://colab.research.google.com/github/hrvf1/Deep-learning-lineament-detection/blob/main/notebooks/04_fusion_mnt_pente_sentinel2.ipynb)
<!-- COLAB:END -->

1. Exécuter la cellule d’installation ; elle récupère le code commun du dépôt.
2. Dans la cellule de paramètres, indiquer les fichiers d’entrée, le masque et le dossier de sortie.
3. Pour utiliser Google Drive, activer `UTILISER_DRIVE` et renseigner des chemins vers son propre Drive.
4. Ajuster les réglages souhaités, puis exécuter les cellules dans l’ordre en examinant les diagnostics et les figures.
5. Retrouver les modèles, métriques et visualisations dans le dossier de résultats de l’exécution.

Pour conserver les résultats après la fin d’une session Colab, choisir un dossier de sortie sur Drive. Un GPU est préférable pour les entraînements longs.

### Exemple : entraîner sur un nouveau MNT

Dans le notebook **01 — MNT seul**, renseigner le chemin du MNT et celui de son masque, puis conserver `MODE="entrainer"`. Un premier essai avec `EPOQUES=2` permet de vérifier le parcours avant de lancer un entraînement plus long. Les paramètres par défaut du protocole sont conservés dans `configs/mnt.json`.

Deux paramètres répondent à des besoins différents :

- `TAILLE_PATCH=64` fixe les dimensions de chaque patch à 64 × 64 pixels ;
- `NOMBRE_PATCHS_AFFICHES=4` affiche quatre exemples sans modifier les données utilisées pour apprendre.

Tous les patches affectés au train sont utilisés automatiquement pour l'apprentissage.

Pour appliquer un modèle existant, choisir `MODE="modele_enregistre"` et renseigner `CHEMIN_MODELE` dans le même notebook. Les statistiques de normalisation associées au modèle sont réutilisées. Les modalités de reprise d’un entraînement interrompu sont décrites dans le [guide d’utilisation](docs/utilisation.md).

### Exécuter sur ordinateur

Depuis la racine du projet, avec Python 3.10 ou plus récent :

```bash
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

Ouvrir ensuite un notebook dans VS Code ou Jupyter, sélectionner l’environnement Python correspondant et renseigner les chemins locaux. Conserver `UTILISER_DRIVE=False` dans ce cas.

## Organisation du code et sorties

J’ai séparé les traitements réutilisables des cellules de conduite des expériences. Les notebooks décrivent le déroulement du travail ; les modules regroupent la lecture des données, les prétraitements, le modèle, l’apprentissage, l’évaluation et les visualisations. Les configurations permettent de retrouver les réglages de chaque expérience.

| Emplacement | Contenu |
|---|---|
| `notebooks/` | Les quatre expériences et leurs explications. |
| `configs/` | Les paramètres par défaut de chaque configuration d’entrée. |
| `src/lineaments/` | Les fonctions communes et l’interface `Experience` appelée par les notebooks. |
| `tests/` | Les contrôles des traitements, des modèles et de l’enchaînement des cellules. |
| `docs/` | Les protocoles, le guide d’utilisation et le bilan de validation. |
| `tools/` | La configuration des liens GitHub et Colab pour la publication. |

Chaque nouvel entraînement crée un dossier distinct contenant notamment le meilleur modèle (`best.pt`), le dernier état d’entraînement (`last.pt`), les paramètres, les statistiques de normalisation, les partitions, l’historique, les métriques et les figures. Lorsque l'analyse vectorielle facultative est activée, les statistiques des polylignes, leur table détaillée et la rose des orientations sont également exportées. L’explorateur interactif permet de parcourir les patches et d’enregistrer des exemples avec leur seuil d’affichage.

Les prédictions sont exportées **par patch**, avec leurs positions. La reconstruction d’une mosaïque GeoTIFF complète ne fait pas partie de cette version.

## Reproductibilité et validation du code

Les graines aléatoires, configurations, partitions, statistiques de normalisation et versions logicielles sont conservées pour documenter chaque exécution. Les résultats numériques peuvent néanmoins varier avec les données, le matériel et les versions des bibliothèques.

La validation locale comprend **31 tests sur des données synthétiques**, couvrant notamment la fidélité des prétraitements aux fonctions d’origine, l’analyse de polylignes depuis un shapefile et des GeoPackages à une ou plusieurs couches, le chargement des poids, la reprise d’entraînement et l’exécution des cellules des quatre notebooks. Un premier entraînement technique du notebook MNT seul a également été exécuté localement avec succès sur les données du projet. Ces contrôles vérifient le fonctionnement du code ; ils ne reproduisent pas les scores historiques présentés plus haut. L’exécution dans l’interface hébergée Colab et les entraînements réels des trois autres expériences restent à vérifier.

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Les détails sont disponibles dans le [bilan de validation](docs/validation.md), les [protocoles et limites](docs/protocoles.md) et le [guide d’utilisation](docs/utilisation.md). La préparation des liens avant publication est décrite dans le [guide de publication](docs/publication.md).

---

**Achraf Ait Alla** · Génie Minéral, École Mohammadia d’Ingénieurs  
Stage de fin d’année à l’ONHYM · Année universitaire 2025–2026
