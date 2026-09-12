# Préparer la publication GitHub

Ce guide concerne la mise en ligne initiale du dépôt. L’utilisation des expériences est décrite dans le README et dans les notebooks.

## Configurer l’adresse du dépôt

Après avoir créé ou choisi le dépôt GitHub, lancer cette commande depuis la racine du projet :

```bash
python tools/configurer_github.py VOTRE_COMPTE/VOTRE_DEPOT
```

Remplacer `VOTRE_COMPTE/VOTRE_DEPOT` par le compte et le nom réels du dépôt. Le script renseigne `DEPOT_GITHUB` dans les quatre notebooks et ajoute leurs liens Colab au README. Il ne publie aucun fichier.

La branche utilisée par défaut est `main`. Pour figer une version :

```bash
python tools/configurer_github.py VOTRE_COMPTE/VOTRE_DEPOT --reference v1.0
```

Le tag `v1.0` doit exister sur GitHub avant d’utiliser ces liens.

## Publier les fichiers

Déposer le contenu du dossier du projet à la racine du dépôt. Le README, `notebooks/`, `src/` et `configs/` doivent être directement accessibles, sans dossier intermédiaire supplémentaire.

Les données personnelles, checkpoints et résultats restent sur le stockage choisi pour les expériences. Le fichier `.gitignore` exclut les formats usuels lors des envois avec Git. Pour un téléversement manuel sur le site GitHub, sélectionner uniquement les fichiers du projet et vérifier que les sorties des notebooks ne contiennent pas de données ou chemins personnels.

## Vérifier le parcours publié

Ouvrir un lien Colab depuis le README, exécuter l’installation et renseigner un petit jeu de données alignées. Effectuer un entraînement court avant un essai complet. Cette vérification valide les liens, la récupération du dépôt et l’exécution dans l’environnement Colab utilisé.
