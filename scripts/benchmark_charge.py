"""Test de charge de l'API : latence (p50, p95, p99) et débit selon le nombre de clients."""

import argparse

import httpx
import pandas as pd

from aeroguard.charge import executer_charge
from aeroguard.rejeu import en_mesures, vols_par_moteur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--moteurs", type=int, default=16, help="un client par moteur")
    parser.add_argument("--vols-par-moteur", type=int, default=30)
    parser.add_argument("--concurrences", default="1,4,16", help="clients en parallèle")
    parser.add_argument("--etiquette", default="api", help="nom du mode testé, pour le tableau")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    args = parser.parse_args()

    par_moteur = vols_par_moteur(pd.read_parquet(args.vols))
    noms = sorted(par_moteur)[: args.moteurs]
    lignes = []
    with httpx.Client(base_url=args.url, timeout=120) as client:

        def envoyer(vol):
            client.post("/flotte/vols", json=vol).raise_for_status()

        # Échauffement : charge les modèles avant de mesurer (premier appel lent).
        client.get("/flotte/moteurs").raise_for_status()

        for concurrence in [int(c) for c in args.concurrences.split(",")]:
            sequences = []
            for nom in noms:
                # Un nom propre à chaque essai : l'historique repart de zéro.
                moteur = f"BENCH{concurrence}_{nom}"
                client.delete(f"/flotte/moteurs/{moteur}")
                cycles, X = par_moteur[nom]
                fin = min(args.vols_par_moteur, len(cycles))
                sequences.append(
                    [
                        {"moteur": moteur, "cycle": int(c), "mesures": en_mesures(ligne)}
                        for c, (_, ligne) in zip(cycles[:fin], X.iloc[:fin].iterrows(), strict=True)
                    ]
                )
            resultat = executer_charge(sequences, envoyer, concurrence)
            lignes.append({"mode": args.etiquette, "clients": concurrence, **resultat})
            for moteur in (s[0]["moteur"] for s in sequences):
                client.delete(f"/flotte/moteurs/{moteur}")

    tableau = pd.DataFrame(lignes)[
        ["mode", "clients", "n", "erreurs", "debit", "p50", "p95", "p99", "duree_s"]
    ]
    print(tableau.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
