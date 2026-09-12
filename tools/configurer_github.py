"""Renseigne une fois le dépôt et les liens Colab avant publication."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depot",help="compte/nom-du-depot")
    parser.add_argument("--reference",default="main",help="Branche ou tag à ouvrir dans Colab")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",args.depot):parser.error("Format requis : compte/depot")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+",args.reference):parser.error("Utiliser un nom simple de branche ou de tag")
    root=Path(__file__).resolve().parents[1]
    rows=[]
    for p in sorted((root/"notebooks").glob("*.ipynb")):
        nb=json.loads(p.read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"]=="code":
                source="".join(c["source"])
                source=re.sub(r'^DEPOT_GITHUB = .*$',f'DEPOT_GITHUB = "{args.depot}"',source,flags=re.M)
                source=re.sub(r'^REFERENCE_GITHUB = .*$',f'REFERENCE_GITHUB = "{args.reference}"',source,flags=re.M)
                c["source"]=source.splitlines(keepends=True)
        p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
        url=f"https://colab.research.google.com/github/{args.depot}/blob/{args.reference}/notebooks/{p.name}"
        rows.append(f"- [{p.stem}]({url})")
    path=root/"README.md";s=path.read_text(encoding="utf-8")
    block="<!-- COLAB:START -->\n"+"\n".join(rows)+"\n<!-- COLAB:END -->"
    s=re.sub(r'<!-- COLAB:START -->.*?<!-- COLAB:END -->',lambda _:block,s,flags=re.S)
    path.write_text(s,encoding="utf-8")
    print("Notebooks et liens préparés pour",args.depot,"référence",args.reference)
    print("Aucun fichier n'a été publié : ajouter ensuite ces fichiers à votre dépôt.")


if __name__=="__main__":main()
