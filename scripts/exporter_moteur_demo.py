"""Exporte un moteur de test en CSV pour le notebook de démo « Try AeroGuard »."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.rejeu import vols_par_moteur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moteur", default="DS02_14", help="moteur de test à exporter")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--sortie", default=None, help="chemin du CSV (par défaut dans local/)")
    args = parser.parse_args()

    vols = pd.read_parquet(args.vols)
    par_moteur = vols_par_moteur(vols)
    if args.moteur not in par_moteur:
        raise SystemExit(f"Moteur inconnu : {args.moteur}. Disponibles : {sorted(par_moteur)}")

    cycles, X = par_moteur[args.moteur]
    demo = X.copy()
    demo.insert(0, "cycle", np.asarray(cycles))
    demo.insert(1, "RUL", vols.loc[X.index, "RUL"].to_numpy())

    sortie = Path(args.sortie or f"local/demo_engine_{args.moteur}.csv")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(sortie, index=False)

    print(f"{sortie} : {len(demo)} vols, {demo.shape[1] - 2} features")
    print(f"RUL de {demo['RUL'].max()} à {demo['RUL'].min()} vols")


if __name__ == "__main__":
    main()
