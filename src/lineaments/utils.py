from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import numpy as np


def ecrire_json(chemin, contenu):
    def convertir(x):
        if isinstance(x, dict):
            return {str(k): convertir(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convertir(v) for v in x]
        if isinstance(x, np.ndarray):
            return convertir(x.tolist())
        if isinstance(x, np.generic):
            return convertir(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    chemin = Path(chemin); chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(convertir(contenu), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def environnement():
    versions = {"python": platform.python_version(), "plateforme": platform.platform()}
    for nom in ("onhym-lineaments", "numpy", "pandas", "matplotlib", "rasterio", "torch", "ipywidgets"):
        try:
            versions[nom] = str(metadata.version(nom))
        except metadata.PackageNotFoundError:
            versions[nom] = "inconnu"
    try:
        versions["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        versions["git_commit"] = None
    return versions
