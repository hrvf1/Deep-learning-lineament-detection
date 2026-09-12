"""Rasters artificiels réservés aux tests techniques ; aucune donnée ONHYM."""
import json
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import LineString
from lineaments.utils import ecrire_json


def creer_donnees_demo(dossier, taille=192, graine=42):
    dossier=Path(dossier);dossier.mkdir(parents=True,exist_ok=True)
    noms={"mnt":"mnt_clean.tif","pente":"pente_clean.tif","hillshade":"hillshade_315_clean.tif","sentinel":"stack_clean.tif","masque":"mask_clean.tif","lineaments":"lineaments.shp","lineaments_gpkg":"lineaments.gpkg"}
    chemins={k:str(dossier/v) for k,v in noms.items()}
    marker=dossier/"donnees_artificielles.json"
    if marker.exists():
        old=json.loads(marker.read_text(encoding="utf-8"))
        if old.get("taille")==taille and old.get("graine")==graine and all(Path(p).is_file() for p in chemins.values()):
            return chemins
        raise ValueError("Ce dossier contient une autre démonstration ou des fichiers incomplets. Choisir un nouveau dossier.")
    if any(Path(p).exists() for p in chemins.values()):
        raise FileExistsError("Les fichiers existent déjà. Choisir un dossier vide pour les données artificielles.")
    rng=np.random.default_rng(graine);r,c=np.indices((taille,taille))
    masque=(((c-2*r)%47<2)|((3*c+r)%71<2)).astype(np.float32)
    mnt=(200+.2*r+.4*c+15*np.sin(c/17)+4*masque+rng.normal(0,.4,(taille,taille))).astype(np.float32)
    pente=(10+5*np.sin(r/21)+4*np.cos(c/16)+2*masque).astype(np.float32)
    hill=(120+40*np.sin((r+c)/27)+25*masque).astype(np.float32)
    bands=np.stack([(700+120*i+1.4*c+.7*r+100*np.sin((r+i*c)/(20+i))+200*masque+rng.normal(0,10,(taille,taille))).astype(np.float32) for i in range(6)])
    arrays={"mnt":mnt[None],"pente":pente[None],"hillshade":hill[None],"sentinel":bands,"masque":masque[None]}
    for key,array in arrays.items():
        with rasterio.open(chemins[key],"w",driver="GTiff",height=taille,width=taille,count=array.shape[0],dtype="float32",crs="EPSG:32628",transform=from_origin(500000,2500000,10,10),compress="deflate") as dst:
            dst.write(array)
            if key=="sentinel":
                for i,name in enumerate(["B2","B3","B4","B8","B11","B12"],1):dst.set_band_description(i,name)
    largeur = taille * 10
    lignes = [
        LineString([(500000 + .1 * largeur, 2500000 - .2 * largeur),
                    (500000 + .7 * largeur, 2500000 - .2 * largeur)]),
        LineString([(500000 + .2 * largeur, 2500000 - .8 * largeur),
                    (500000 + .7 * largeur, 2500000 - .3 * largeur)]),
        LineString([(500000 + .8 * largeur, 2500000 - .8 * largeur),
                    (500000 + .8 * largeur, 2500000 - .3 * largeur)]),
    ]
    vecteurs = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=lignes, crs="EPSG:32628")
    vecteurs.to_file(chemins["lineaments"])
    vecteurs.to_file(chemins["lineaments_gpkg"], layer="lineaments", driver="GPKG")
    ecrire_json(marker,{"nature":"Données artificielles ; aucune valeur scientifique ou résultat ONHYM", "taille":taille,"graine":graine})
    return chemins
