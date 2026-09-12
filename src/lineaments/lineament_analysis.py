"""Analyse descriptive des polylignes de référence."""
from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS


def _verifier_source_vectorielle(chemin):
    chemin = Path(chemin).expanduser().resolve()
    if chemin.suffix.lower() not in {".shp", ".gpkg"}:
        raise ValueError("CHEMIN_LINEAMENTS doit désigner un fichier .shp ou .gpkg.")
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier vectoriel introuvable : {chemin}")
    if chemin.suffix.lower() == ".shp":
        fichiers = {p.name.lower() for p in chemin.parent.iterdir() if p.is_file()}
        manquants = [
            chemin.with_suffix(extension).name
            for extension in (".shp", ".shx", ".dbf", ".prj")
            if chemin.with_suffix(extension).name.lower() not in fichiers
        ]
        if manquants:
            raise FileNotFoundError(
                "Shapefile incomplet. Fichier(s) manquant(s) dans le même dossier : "
                + ", ".join(manquants)
            )
    return chemin


def _lister_couches_geopackage(chemin):
    """Retourne les couches d'un GeoPackage avec l'API disponible de GeoPandas."""
    if hasattr(gpd, "list_layers"):
        couches = gpd.list_layers(chemin)
        return couches["name"].astype(str).tolist()
    try:  # GeoPandas 0.14 utilise encore Fiona par défaut.
        import fiona
    except ImportError as erreur:  # pragma: no cover - dépend de la version installée
        raise RuntimeError(
            "Impossible de lister les couches du GeoPackage. Installer Pyogrio ou Fiona."
        ) from erreur
    return [str(nom) for nom in fiona.listlayers(chemin)]


def _choisir_couche(chemin, couche):
    if chemin.suffix.lower() == ".shp":
        if couche is not None:
            raise ValueError("COUCHE_LINEAMENTS s'utilise uniquement avec un fichier .gpkg.")
        return None

    couches = _lister_couches_geopackage(chemin)
    if not couches:
        raise ValueError("Le GeoPackage ne contient aucune couche vectorielle.")
    if couche is None:
        if len(couches) == 1:
            return couches[0]
        raise ValueError(
            "Le GeoPackage contient plusieurs couches ("
            + ", ".join(couches)
            + "). Renseigner COUCHE_LINEAMENTS avec le nom exact de la couche à analyser."
        )
    if couche not in couches:
        raise ValueError(
            f"Couche GeoPackage inconnue : {couche!r}. Couches disponibles : "
            + ", ".join(couches)
        )
    return couche


def _unite_crs(crs):
    if crs is None:
        raise ValueError("Le fichier vectoriel ne possède pas de CRS.")
    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        warnings.warn(
            "Le fichier vectoriel utilise un CRS géographique : les longueurs seraient exprimées en degrés. "
            "Le reprojeter dans un CRS projeté adapté avant l'analyse.",
            UserWarning,
            stacklevel=3,
        )
        return "degré"
    unite = (crs.axis_info[0].unit_name if crs.axis_info else "unité du CRS").lower()
    if unite in {"metre", "meter", "metres", "meters", "mètre", "mètres"}:
        return "m"
    return unite


def _orientation_extremites(geometrie):
    """Azimut axial [0, 180[, mesuré dans le sens horaire depuis le nord."""
    coordonnees = np.asarray(geometrie.coords, dtype=float)
    dx = coordonnees[-1, 0] - coordonnees[0, 0]
    dy = coordonnees[-1, 1] - coordonnees[0, 1]
    if np.hypot(dx, dy) > 0:
        return float(np.degrees(np.arctan2(dx, dy)) % 180.0)

    # Cas rare d'une ligne fermée : l'axe principal de tous les sommets remplace les extrémités.
    points = coordonnees[:, :2] - coordonnees[:, :2].mean(axis=0)
    valeurs, vecteurs = np.linalg.eigh(points.T @ points)
    vx, vy = vecteurs[:, int(np.argmax(valeurs))]
    return float(np.degrees(np.arctan2(vx, vy)) % 180.0)


def _figure_analyse(longueurs, orientations, unite, pas_angle):
    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    axe_longueurs = fig.add_subplot(1, 2, 1)
    axe_rose = fig.add_subplot(1, 2, 2, projection="polar")

    axe_longueurs.boxplot(
        longueurs,
        patch_artist=True,
        boxprops={"facecolor": "#d9eef8", "edgecolor": "#1878b4"},
        medianprops={"color": "#d95f02", "linewidth": 2},
    )
    axe_longueurs.set_title("Distribution des longueurs des linéaments")
    axe_longueurs.set_ylabel(f"Longueur ({unite})")
    axe_longueurs.set_xticks([1], [f"{len(longueurs)} linéament(s)"])
    axe_longueurs.grid(axis="y", alpha=0.25)

    bornes = np.arange(0, 180 + pas_angle, pas_angle, dtype=float)
    effectifs, _ = np.histogram(orientations, bins=bornes)
    centres = bornes[:-1] + pas_angle / 2
    axe_rose.bar(
        np.deg2rad(centres),
        effectifs,
        width=np.deg2rad(pas_angle),
        color="#1485c1",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    axe_rose.set_theta_zero_location("N")
    axe_rose.set_theta_direction(-1)
    axe_rose.set_thetagrids(np.arange(0, 360, 45))
    axe_rose.set_title("Distribution des orientations", pad=18)
    axe_rose.set_ylabel("Nombre de linéaments", labelpad=28)
    axe_rose.grid(alpha=0.35)
    fig.suptitle("Analyse des linéaments vectoriels de référence", fontsize=14, fontweight="bold")
    return fig


def analyser_fichier_lineaments(
    chemin,
    couche=None,
    crs_reference=None,
    emprise_reference=None,
    pas_angle=5,
):
    """Calcule longueurs et orientations depuis un shapefile ou un GeoPackage."""
    if pas_angle <= 0 or 180 % pas_angle != 0:
        raise ValueError("pas_angle doit être un diviseur positif de 180, par exemple 5, 10 ou 15.")

    chemin = _verifier_source_vectorielle(chemin)
    couche = _choisir_couche(chemin, couche)
    vecteurs = gpd.read_file(chemin, **({"layer": couche} if couche is not None else {}))
    if vecteurs.empty:
        raise ValueError("Le fichier vectoriel ne contient aucune entité.")
    if vecteurs.crs is None:
        raise ValueError("Le fichier vectoriel ne possède pas de CRS.")
    if crs_reference is not None and CRS.from_user_input(vecteurs.crs) != CRS.from_user_input(crs_reference):
        raise ValueError(
            f"CRS différents : vecteur={vecteurs.crs}, rasters={crs_reference}. "
            "Reprojeter le fichier vectoriel sur le CRS des rasters avant de poursuivre."
        )

    vecteurs = vecteurs.loc[vecteurs.geometry.notna() & ~vecteurs.geometry.is_empty].copy()
    if vecteurs.empty:
        raise ValueError("Le fichier vectoriel ne contient aucune géométrie exploitable.")
    if not vecteurs.geometry.is_valid.all():
        raise ValueError("Le fichier vectoriel contient des géométries invalides à corriger dans le SIG.")
    types = set(vecteurs.geom_type)
    if not types.issubset({"LineString", "MultiLineString"}):
        raise ValueError(
            "Le fichier vectoriel doit contenir uniquement des polylignes. Types rencontrés : "
            + ", ".join(sorted(types))
        )

    vecteurs["entite_source"] = np.arange(1, len(vecteurs) + 1)
    lignes = vecteurs.explode(index_parts=False, ignore_index=True)
    longueurs = lignes.geometry.length.to_numpy(dtype=float)
    valides = np.isfinite(longueurs) & (longueurs > 0)
    lignes = lignes.loc[valides].reset_index(drop=True)
    longueurs = longueurs[valides]
    if not len(lignes):
        raise ValueError("Aucune polyligne de longueur positive n'a été trouvée.")

    if emprise_reference is not None:
        gauche, bas, droite, haut = emprise_reference
        xmin, ymin, xmax, ymax = lignes.total_bounds
        if xmax < gauche or xmin > droite or ymax < bas or ymin > haut:
            raise ValueError("L'emprise du fichier vectoriel ne recoupe pas celle des rasters.")
        if xmin < gauche or xmax > droite or ymin < bas or ymax > haut:
            warnings.warn(
                "Une partie des linéaments se trouve hors de l'emprise des rasters.",
                UserWarning,
                stacklevel=2,
            )

    unite = _unite_crs(lignes.crs)
    orientations = np.asarray([_orientation_extremites(geom) for geom in lignes.geometry])
    details = pd.DataFrame({
        "lineament": np.arange(1, len(lignes) + 1),
        "entite_source": lignes["entite_source"].to_numpy(),
        "longueur": longueurs,
        "unite": unite,
        "orientation_deg": orientations,
    }).set_index("lineament")

    effectifs, bornes = np.histogram(
        orientations, bins=np.arange(0, 180 + pas_angle, pas_angle)
    )
    indice_dominant = int(np.argmax(effectifs))
    orientation_dominante = f"{bornes[indice_dominant]:.0f}–{bornes[indice_dominant + 1]:.0f}°"
    resume = pd.DataFrame([
        {"indicateur": "Entités du fichier vectoriel", "valeur": len(vecteurs), "unite": "entité"},
        {"indicateur": "Linéaments analysés", "valeur": len(details), "unite": "polyligne"},
        {"indicateur": "Longueur cumulée", "valeur": float(longueurs.sum()), "unite": unite},
        {"indicateur": "Longueur moyenne", "valeur": float(longueurs.mean()), "unite": unite},
        {"indicateur": "Longueur médiane", "valeur": float(np.median(longueurs)), "unite": unite},
        {"indicateur": "Longueur minimale", "valeur": float(longueurs.min()), "unite": unite},
        {"indicateur": "Longueur maximale", "valeur": float(longueurs.max()), "unite": unite},
        {"indicateur": "Orientation dominante", "valeur": orientation_dominante, "unite": "azimut axial"},
    ]).set_index("indicateur")
    details = details.sort_values("longueur", ascending=False)
    figure = _figure_analyse(longueurs, orientations, unite, pas_angle)
    return resume, details, figure


def analyser_shapefile_lineaments(
    chemin,
    crs_reference=None,
    emprise_reference=None,
    pas_angle=5,
):
    """Compatibilité avec l'ancien nom limité aux shapefiles."""
    return analyser_fichier_lineaments(
        chemin,
        crs_reference=crs_reference,
        emprise_reference=emprise_reference,
        pas_angle=pas_angle,
    )
