"""Simulation en direct : les moteurs de test volent, l'API alerte, l'atelier intervient."""

import argparse
import time

import httpx
import numpy as np
import pandas as pd

from aeroguard.rejeu import vols_par_moteur
from aeroguard.simulateur import MoteurSimule, Simulateur

SYMBOLES = {"alerte": "ALERTE     ", "maintenance": "ATELIER    ", "panne": "PANNE (!!) "}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moteurs", type=int, default=39, help="nombre de moteurs de test")
    parser.add_argument("--en-vol", type=int, default=8, help="moteurs en vol en même temps")
    parser.add_argument("--delai", type=int, default=5, help="vols entre l'alerte et l'atelier")
    parser.add_argument("--pause", type=float, default=0.5, help="secondes entre deux tours")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()

    vols = pd.read_parquet(args.vols)
    par_moteur = vols_par_moteur(vols)
    noms = list(np.random.default_rng(args.graine).permutation(sorted(par_moteur)))
    moteurs = []
    for nom in noms[: args.moteurs]:
        cycles, X = par_moteur[nom]
        rul = vols.loc[X.index, "RUL"].to_numpy()
        moteurs.append(MoteurSimule(nom, cycles, X, rul))

    with httpx.Client(base_url=args.url, timeout=120) as client:

        def envoyer(nom, cycle, mesures):
            vol = {"moteur": nom, "cycle": cycle, "mesures": mesures}
            reponse = client.post("/flotte/vols", json=vol)
            reponse.raise_for_status()
            return reponse.json()

        def reinitialiser(nom):
            client.delete(f"/flotte/moteurs/{nom}")

        simulateur = Simulateur(moteurs, envoyer, reinitialiser, args.delai, args.en_vol)
        tour = 0
        while not simulateur.termine():
            tour += 1
            for e in simulateur.tour():
                libelle = SYMBOLES[e["evenement"]]
                print(f"tour {tour:>3}  {libelle} {e['moteur']:<10} RUL {e['rul']}")
            time.sleep(args.pause)

    bilan = simulateur.bilan()
    print("\nBilan")
    print(f"  moteurs suivis jusqu'au bout : {bilan['moteurs']}")
    print(f"  pannes évitées (atelier)     : {bilan['pannes_evitees']}")
    print(f"  pannes en service            : {bilan['pannes_en_service']}")
    print(f"  avance médiane de l'alerte   : {bilan['avance_mediane']} vols")
    print(f"  vols restants à l'atelier    : {bilan['vols_restants_moyens']} en moyenne")


if __name__ == "__main__":
    main()
