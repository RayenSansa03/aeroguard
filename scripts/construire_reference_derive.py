"""Calcule la référence de dérive (déciles d'entraînement) à côté des modèles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.derive import construire_reference
from aeroguard.donnees_ml import preparer_xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeles", default="local/models")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    args = parser.parse_args()

    modeles = Path(args.modeles)
    stats = np.load(modeles / "autoencodeur_centre_v5_stats.npz", allow_pickle=True)
    colonnes = [str(c) for c in stats["colonnes"]]

    X_train, _ = preparer_xy(pd.read_parquet(args.vols), "train")
    X = X_train[colonnes].to_numpy(dtype="float64")
    X = np.where(np.isnan(X), stats["medianes_imputation"], X)

    reference = construire_reference(X, colonnes)
    sortie = modeles / "reference_derive.json"
    sortie.write_text(json.dumps(reference), encoding="utf-8")
    print(f"{len(colonnes)} features, {reference['n_reference']} vols d'entraînement")
    print(f"Référence écrite dans {sortie} ({sortie.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
