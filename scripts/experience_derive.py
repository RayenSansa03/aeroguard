"""Expérience de dérive : jeu de test brut, capteur biaisé, capteur mort."""

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


def ligne_feature(rapport, nom):
    """PSI et statut d'une feature précise, même si elle n'est pas dans le top."""
    return next(ligne for ligne in rapport["features"] if ligne["feature"] == nom)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeles", default="local/models")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument(
        "--sigmas", type=float, default=0.5, help="Biais ajouté, en écarts-types de la feature"
    )
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
    capteur = colonnes[j]

    moyenne, ecart_type = X[:, j].mean(), X[:, j].std()
    print(f"{capteur} sur le jeu de test : moyenne {moyenne:.3f}, écart-type {ecart_type:.3f}")

    afficher("1. Jeu de test tel quel", rapport_derive(reference, X))

    decale = X.copy()
    decale[:, j] += args.sigmas * X[:, j].std()
    rapport_decale = rapport_derive(reference, decale)
    afficher(f"2. Capteur {capteur} biaisé de +{args.sigmas} écart-type", rapport_decale)
    ligne = ligne_feature(rapport_decale, capteur)
    print(f"  -> {capteur} : PSI {ligne['psi']:.3f} ({ligne['status']})")

    mort = X.copy()
    mort[:, j] = medianes[j]
    titre = f"3. Capteur {capteur} mort (imputé à la médiane)"
    afficher(titre, rapport_derive(reference, mort))


if __name__ == "__main__":
    main()
