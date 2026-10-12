"""Expérience de dérive : jeu de test brut, capteur décalé, capteur mort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.derive import rapport_derive
from aeroguard.donnees_ml import preparer_xy


def afficher(titre, rapport, n=5):
    print(
        f"\n=== {titre} : {rapport['status'].upper()} "
        f"({rapport['n_major']} fortes, {rapport['n_moderate']} modérées) ==="
    )
    for ligne in rapport["features"][:n]:
        print(f"  {ligne['feature']:<22} PSI {ligne['psi']:7.3f}  {ligne['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeles", default="local/models")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--decalage", type=float, default=0.03, help="Décalage relatif du capteur")
    args = parser.parse_args()

    modeles = Path(args.modeles)
    reference = json.loads((modeles / "reference_derive.json").read_text(encoding="utf-8"))
    stats = np.load(modeles / "autoencodeur_centre_v5_stats.npz", allow_pickle=True)
    colonnes = reference["colonnes"]
    medianes = stats["medianes_imputation"]

    X_test, _ = preparer_xy(pd.read_parquet(args.vols), "test")
    X = X_test[colonnes].to_numpy(dtype="float64")
    X = np.where(np.isnan(X), medianes, X)
    j = next(i for i, c in enumerate(colonnes) if c.startswith("T48"))

    afficher("1. Jeu de test tel quel", rapport_derive(reference, X))

    decale = X.copy()
    decale[:, j] *= 1 + args.decalage
    titre = f"2. Capteur {colonnes[j]} recalibré +{args.decalage:.0%}"
    afficher(titre, rapport_derive(reference, decale))

    mort = X.copy()
    mort[:, j] = medianes[j]
    titre = f"3. Capteur {colonnes[j]} mort (imputé à la médiane)"
    afficher(titre, rapport_derive(reference, mort))


if __name__ == "__main__":
    main()
