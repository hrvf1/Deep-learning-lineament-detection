"""Cartes et explorateurs communs ; les réglages visuels ne changent pas l'apprentissage."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from .config import noms_canaux
from .evaluation import carte_erreurs, metriques_par_patch

COULEURS_ERREURS = ["#F5F5F5", "#2E8B57", "#E67E22", "#2878B5"]


def etirer_affichage(bande):
    vals = bande[np.isfinite(bande)]
    if not vals.size:
        return np.zeros_like(bande)
    lo, hi = np.percentile(vals, [2, 98])
    return np.nan_to_num(np.clip((bande-lo)/(hi-lo+1e-10), 0, 1), nan=0)


def apercu(images, noms):
    if all(n in noms for n in ("B4", "B3", "B2")):
        return np.stack([etirer_affichage(images[noms.index(n)]) for n in ("B4", "B3", "B2")], axis=-1), None
    return images[0], "terrain" if noms[0] == "MNT" else "gray"


def finaliser_carte(ax, couche, titre):
    b = couche.meta["bounds"]
    ax.set(xlim=(b.left,b.right), ylim=(b.bottom,b.top), title=titre)
    geographic = couche.meta["crs"] is not None and couche.meta["crs"].is_geographic
    unit = "°" if geographic else (couche.meta["crs"].linear_units if couche.meta["crs"] else "unité du CRS")
    ax.set_xlabel(f"{'Longitude' if geographic else 'Est'} ({unit})")
    ax.set_ylabel(f"{'Latitude' if geographic else 'Nord'} ({unit})")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_aspect("equal"); ax.grid(False)


def visualiser_donnees(donnees, superposer=True):
    figures = []
    for couche in donnees.couches + ([donnees.masque_source] if donnees.masque_source else []):
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        b = couche.meta["bounds"]
        image = ax.imshow(couche.valeurs, cmap=couche.cmap, extent=[b.left,b.right,b.bottom,b.top], interpolation="nearest")
        finaliser_carte(ax, couche, couche.nom)
        fig.colorbar(image, ax=ax, shrink=.8, label=couche.unite)
        figures.append(fig)
    if all(n in [c.nom for c in donnees.couches] for n in ("B4","B3","B2")) and all(donnees.alignee(c) for c in donnees.couches):
        rgb, _ = apercu(np.stack([c.valeurs for c in donnees.couches]), [c.nom for c in donnees.couches])
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        b = donnees.reference.meta["bounds"]
        ax.imshow(rgb, extent=[b.left,b.right,b.bottom,b.top]); finaliser_carte(ax, donnees.reference, "Sentinel-2 — RGB B4/B3/B2")
        figures.append(fig)
    if superposer and donnees.masque is not None:
        if not donnees.alignee(donnees.masque_source):
            print("Superposition non affichée : grilles différentes. Consulter le diagnostic.")
        else:
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            # Seules les grilles doivent correspondre pour inspecter une zone avec NoData.
            arrays = np.stack([c.valeurs for c in donnees.couches]) if all(donnees.alignee(c) for c in donnees.couches) else donnees.reference.valeurs[None]
            names = [c.nom for c in donnees.couches] if len(arrays) == len(donnees.couches) else [donnees.reference.nom]
            image, cmap = apercu(arrays, names); b=donnees.reference.meta["bounds"]
            extent=[b.left,b.right,b.bottom,b.top]
            ax.imshow(image, cmap=cmap, extent=extent)
            overlay=np.ma.masked_where(donnees.masque != 1, donnees.masque)
            ax.imshow(overlay, cmap=ListedColormap(["#D900A6"]), vmin=0, vmax=1, alpha=.85, extent=extent, interpolation="nearest")
            ax.legend(handles=[Patch(color="#D900A6", label="Linéaments de référence")])
            finaliser_carte(ax, donnees.reference, "Superposition des entrées et du masque")
            figures.append(fig)
    return figures


def carte_decoupage(donnees, positions, taille, masques=None, partitions=None, exclues=None):
    ref = donnees.reference; t = ref.meta["transform"]; b=ref.meta["bounds"]
    if abs(t.b)>1e-12 or abs(t.d)>1e-12:
        raise ValueError("Reprojeter la grille orientée avant d'afficher les limites des patchs.")
    fig, ax=plt.subplots(figsize=(10,8), constrained_layout=True)
    ax.imshow(ref.valeurs, cmap=ref.cmap, extent=[b.left,b.right,b.bottom,b.top])
    couleurs = {"train":"#2878B5", "validation":"#D48C20", "test":"#7556A6"}
    classes = {}
    if partitions is not None:
        classes = {int(i): g for g, ids in partitions.items() for i in ids}
    step=max(1,int(np.ceil(len(positions)/5000)))
    if step>1:print(f"Aperçu : un contour sur {step} affiché. Tous les patchs restent disponibles pour la préparation.")
    for i in range(0,len(positions),step):
        r,c=positions[i]
        x,y = t*(int(c),int(r+taille))
        if partitions is not None:
            color = couleurs.get(classes.get(i), "#AAAAAA")
        elif masques is not None:
            color = "#248441" if masques[i].any() else "#C23D3D"
        else:
            color = "#2878B5"
        ax.add_patch(Rectangle((x,y), taille*t.a, -taille*t.e, fill=False, edgecolor=color, linewidth=.6))
    for r,c in ([] if exclues is None else exclues):
        x,y=t*(int(c),int(r+taille))
        ax.add_patch(Rectangle((x,y),taille*t.a,-taille*t.e,facecolor="none",edgecolor="#777777",hatch="//",linewidth=.4))
    if partitions is not None:
        leg=[Patch(facecolor="none",edgecolor=col,label=nom) for nom,col in couleurs.items()]
        if len(classes)<len(positions):leg.append(Patch(facecolor="none",edgecolor="#AAAAAA",label="Train non retenu"))
    elif masques is not None:
        leg=[Patch(facecolor="none",edgecolor="#248441",label="Avec linéaments"),Patch(facecolor="none",edgecolor="#C23D3D",label="Sans linéaments")]
    else:
        leg=[Patch(facecolor="none",edgecolor="#2878B5",label="Patch potentiel")]
    if exclues is not None and len(exclues):leg.append(Patch(facecolor="none",edgecolor="#777777",hatch="//",label="Exclu : valeurs invalides"))
    ax.legend(handles=leg, loc="best")
    finaliser_carte(ax,ref,f"Découpage — {taille} × {taille} pixels | {len(positions)} patchs")
    return fig


def visualiser_patchs(exp, groupe="train", nombre=8):
    xs,ys=exp.groupe(groupe)
    if not isinstance(nombre,int) or nombre<1 or nombre>32:
        raise ValueError("Choisir entre 1 et 32 patchs par figure.")
    ids=np.random.default_rng(exp.configuration["graine"]).choice(len(xs),size=min(nombre,len(xs)),replace=False)
    fig,axes=plt.subplots(len(ids),2,figsize=(8,2.5*len(ids)),squeeze=False,constrained_layout=True)
    for row,i in enumerate(ids):
        img,cmap=apercu(xs[i],noms_canaux(exp.configuration));axes[row,0].imshow(img,cmap=cmap)
        axes[row,0].set_title(f"{groupe} — patch {i}")
        if ys is not None:axes[row,1].imshow(ys[i],cmap="gray",vmin=0,vmax=1)
        axes[row,1].set_title("Masque de référence" if ys is not None else "Masque non fourni")
    for ax in axes.flat:ax.axis("off")
    return fig


def visualiser_normalisation(exp, canal=None, index=0):
    if not exp._normalises:
        raise RuntimeError("Exécuter normaliser() avant cet affichage.")
    names=noms_canaux(exp.configuration)
    if canal is None:c=0
    elif isinstance(canal,int):c=canal
    else:
        matching=[i for i,n in enumerate(names) if n.casefold()==canal.casefold()]
        if not matching:raise ValueError(f"Canaux disponibles : {names}")
        c=matching[0]
    if not 0<=c<len(names) or not 0<=index<len(exp.partitions["train"]):
        raise ValueError("Canal ou index de patch hors limites.")
    raw=exp.images[exp.partitions["train"],c];norm=exp._normalises["train"][:,c]
    fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    cmap=exp.donnees.couches[c].cmap
    for j,arr in enumerate((raw,norm)):
        im=ax[0,j].imshow(arr[index],cmap=cmap);fig.colorbar(im,ax=ax[0,j],shrink=.75)
        # Histogramme borné pour les grands jeux : échantillonnage d'affichage uniquement.
        vals=arr.reshape(-1);step=max(1,len(vals)//200000)
        ax[1,j].hist(vals[::step],bins=50,color="#2878B5")
        ax[1,j].set(xlabel="Valeur",ylabel="Pixels échantillonnés")
        ax[0,j].set_title("Avant normalisation" if j==0 else "Après normalisation")
    fig.suptitle(f"{names[c]} — statistiques calculées uniquement sur le train")
    return fig


def visualiser_augmentation(exp,index=0):
    if exp._augmente is None:raise RuntimeError("Exécuter augmenter() avant cet affichage.")
    if not 0<=index<len(exp.partitions["train"]):raise ValueError("Index de patch hors limites.")
    transformations=exp.configuration["transformations"];n=len(transformations)
    fig,axes=plt.subplots(2,n,figsize=(3*n,6),squeeze=False,constrained_layout=True)
    for j,t in enumerate(transformations):
        i=index*n+j;img,cmap=apercu(exp._augmente[0][i],noms_canaux(exp.configuration))
        axes[0,j].imshow(img,cmap=cmap);axes[1,j].imshow(exp._augmente[1][i],cmap="gray",vmin=0,vmax=1)
        axes[0,j].set_title(t)
    for ax in axes.flat:ax.axis("off")
    fig.suptitle(f"Transformations synchronisées — patch train {index}")
    return fig


def visualiser_historique(exp):
    h=exp.historique;epochs=np.arange(1,len(h["train_loss"])+1);best=int(np.argmax(h["val_iou"]))+1
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].plot(epochs,h["train_loss"],label="Train");axes[0].plot(epochs,h["val_loss"],label="Validation")
    axes[0].set(xlabel="Époque",ylabel="Dice Loss",title="Pertes")
    axes[1].plot(epochs,h["val_iou"],label="IoU moyenne par lot")
    if "val_iou_global" in h:axes[1].plot(epochs,h["val_iou_global"],label="IoU globale",alpha=.8)
    axes[1].set(xlabel="Époque",ylabel="IoU",title="Validation")
    for ax in axes:
        ax.axvline(best,color="#555555",linestyle="--",label=f"Meilleur modèle : époque {best}");ax.legend();ax.grid(alpha=.2)
    if exp.dossier:
        dest=exp.dossier/"figures";dest.mkdir(exist_ok=True);fig.savefig(dest/"historique.png",dpi=160,bbox_inches="tight")
    return fig


def visualiser_predictions(exp,groupe,indices=None,seuil=None,enregistrer=False):
    seuil=exp.seuil if seuil is None else float(seuil)
    if not 0<seuil<1:raise ValueError("Seuil compris entre 0 et 1 attendu.")
    xs,ys=exp.groupe(groupe);probas=exp.predire(groupe)
    indices=list(range(min(3,len(xs)))) if indices is None else list(indices)
    if not indices or len(indices)>16 or any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(xs) for i in indices):
        raise ValueError("Choisir 1 à 16 indices de patchs disponibles.")
    columns=5 if ys is not None else 3
    fig,axes=plt.subplots(len(indices),columns,figsize=(3*columns,3*len(indices)),squeeze=False,constrained_layout=True)
    for row,i in enumerate(indices):
        img,cmap=apercu(xs[i],noms_canaux(exp.configuration));axes[row,0].imshow(img,cmap=cmap);axes[row,0].set_title(f"Entrée — patch {i}")
        j=1
        if ys is not None:
            axes[row,j].imshow(ys[i],cmap="gray",vmin=0,vmax=1);axes[row,j].set_title("Référence");j+=1
        im=axes[row,j].imshow(probas[i],cmap="hot",vmin=0,vmax=1);axes[row,j].set_title("Probabilité");fig.colorbar(im,ax=axes[row,j],shrink=.7)
        axes[row,j+1].imshow(probas[i]>=seuil,cmap="gray",vmin=0,vmax=1)
        axes[row,j+1].set_title(f"Prédiction — seuil {seuil:.2f}\n{int((probas[i] >= seuil).sum())} pixels positifs")
        if ys is not None:
            axes[row,j+2].imshow(carte_erreurs(probas[i],ys[i],seuil),cmap=ListedColormap(COULEURS_ERREURS),vmin=0,vmax=3,interpolation="nearest")
            score=metriques_par_patch(probas[i:i+1],ys[i:i+1],seuil).iloc[0]["IoU"]
            axes[row,j+2].set_title("Erreurs — IoU "+(f"{score:.3f}" if np.isfinite(score) else "n.d."))
    for ax in axes.flat:ax.axis("off")
    fig.suptitle(f"{groupe} — seuil d'affichage {seuil:.2f}")
    if ys is not None:
        fig.legend(handles=[Patch(color=COULEURS_ERREURS[i],label=l) for i,l in enumerate(["Fond correct","Vrais positifs","Faux positifs","Faux négatifs"])],loc="outside lower center",ncol=4)
    if enregistrer:
        dest=exp.exporter()/"figures";dest.mkdir(exist_ok=True)
        ident="-".join(map(str,indices));path=dest/f"predictions_{groupe}_{ident}_seuil-{seuil:.2f}.png"
        fig.savefig(path,dpi=160,bbox_inches="tight")
        from .utils import ecrire_json
        ecrire_json(path.with_suffix(".json"),{"groupe":groupe,"indices":indices,"seuil_affichage":seuil,"seuil_evaluation":exp.seuil})
    return fig


def explorer_predictions(exp,groupe=None):
    import ipywidgets as w
    from IPython.display import display, clear_output
    if exp.modele is None:raise RuntimeError("Entraîner ou charger un modèle avant l'exploration.")
    groupes=["nouvelle_zone"] if exp._checkpoint else ["train","train_augmente","validation","test"]
    groupe=groupe or ("nouvelle_zone" if exp._checkpoint else "test")
    menu=w.Dropdown(options=groupes,value=groupe,description="Groupe")
    index=w.BoundedIntText(value=0,min=0,max=len(exp.groupe(groupe)[0])-1,description="Patch")
    seuil=w.FloatSlider(value=exp.seuil,min=.05,max=.95,step=.01,description="Seuil",continuous_update=False)
    prev=w.Button(description="Précédent");nxt=w.Button(description="Suivant")
    add=w.Button(description="Sélectionner");empty=w.Button(description="Vider la sélection");save=w.Button(description="Exporter la sélection")
    label=w.Label(value="Aucun patch sélectionné");out=w.Output();status=w.Output();selection=[]
    def draw(*_):
        with out:
            clear_output(wait=True)
            fig=visualiser_predictions(exp,menu.value,[index.value],seuil.value)
            display(fig);plt.close(fig)
    def change_group(change):
        index.value=0;index.max=len(exp.groupe(change["new"])[0])-1;draw()
    def select(_):
        item=(menu.value,index.value)
        if item not in selection:selection.append(item)
        label.value=", ".join(f"{g} #{i}" for g,i in selection)
    def clear(_):
        selection.clear();label.value="Aucun patch sélectionné"
    def export(_):
        with status:
            clear_output(wait=True)
            if not selection:
                print("Sélectionner au moins un patch.");return
            for g in dict.fromkeys(g for g,i in selection):
                ids=[i for group,i in selection if group==g]
                for start in range(0,len(ids),16):
                    fig=visualiser_predictions(exp,g,ids[start:start+16],seuil.value,True);plt.close(fig)
            print("Figures enregistrées dans",exp.dossier/"figures")
    menu.observe(change_group,names="value");index.observe(draw,names="value");seuil.observe(draw,names="value")
    prev.on_click(lambda _:setattr(index,"value",max(0,index.value-1)))
    nxt.on_click(lambda _:setattr(index,"value",min(index.max,index.value+1)))
    add.on_click(select);empty.on_click(clear);save.on_click(export)
    box=w.VBox([menu,w.HBox([prev,index,nxt]),seuil,w.HBox([add,empty,save]),label,out,status])
    display(box);draw()
    return box


def visualiser_seuils(table, seuil):
    """Trace les scores de validation selon le seuil de segmentation."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for nom in ("IoU", "Dice", "Précision", "Rappel"):
        ax.plot(table.index, table[nom], label=nom)
    ax.axvline(seuil, color="#555555", linestyle="--", label=f"Seuil retenu : {seuil:.2f}")
    ax.set(xlabel="Seuil", ylabel="Score", ylim=(0, 1.03), title="Sensibilité au seuil — validation")
    ax.grid(alpha=.2)
    ax.legend()
    return fig
