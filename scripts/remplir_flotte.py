"""Envoie à l'API une partie de la vie de plusieurs moteurs de test, pour le tableau de bord."""

import argparse

import httpx
import numpy as np
import pandas as pd

from aeroguard.rejeu import envoyer_vols, vols_par_moteur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moteurs", type=int, default=8, help="nombre de moteurs à envoyer")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--graine", type=int, default=0)
    parser.add_argument("--debut", type=float, default=0.3, help="part de vie minimale envoyée")
    parser.add_argument("--fin", type=float, default=1.0, help="part de vie maximale envoyée")
    args = parser.parse_args()

    rng = np.random.default_rng(args.graine)
    par_moteur = vols_par_moteur(pd.read_parquet(args.vols))
    choisis = sorted(rng.choice(sorted(par_moteur), size=args.moteurs, replace=False))

    with httpx.Client(base_url=args.url, timeout=120) as client:
        for moteur in choisis:
            cycles, X = par_moteur[moteur]
            # Chaque moteur est coupé au hasard entre --debut et --fin de sa vie.
            jusqu_a = max(1, int(len(cycles) * rng.uniform(args.debut, args.fin)))
            recos = envoyer_vols(client, moteur, cycles, X, jusqu_a=jusqu_a)
            print(f"{moteur:<10} {len(recos):>3}/{len(cycles)} vols  ->  {recos[-1]['niveau']}")


if __name__ == "__main__":
    main()
