"""Vérifie que l'API donne les mêmes décisions que le notebook 7.1 pour un moteur de test."""

import argparse

import httpx
import numpy as np
import pandas as pd

from aeroguard.donnees_ml import preparer_xy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moteur", default="DS02_14")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--rejeu", default="local/data/71_rejeu_decisions.csv")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    vols = pd.read_parquet(args.vols)
    X_test, _ = preparer_xy(vols, "test")
    infos = vols.loc[X_test.index, ["moteur", "cycle"]]
    masque = (infos["moteur"] == args.moteur).to_numpy()
    cycles = infos["cycle"].to_numpy()[masque]
    ordre = np.argsort(cycles)
    X_moteur = X_test[masque].iloc[ordre]
    cycles = cycles[ordre]

    attendu = pd.read_csv(args.rejeu)
    attendu = attendu[attendu["moteur"] == args.moteur].sort_values("cycle")

    resultats = []
    with httpx.Client(base_url=args.url, timeout=120) as client:
        client.delete(f"/flotte/moteurs/{args.moteur}")
        for cycle, (_, ligne) in zip(cycles, X_moteur.iterrows(), strict=True):
            mesures = {str(c): (None if pd.isna(v) else float(v)) for c, v in ligne.items()}
            vol = {"moteur": args.moteur, "cycle": int(cycle), "mesures": mesures}
            reponse = client.post("/flotte/vols", json=vol)
            reponse.raise_for_status()
            resultats.append(reponse.json())

    api = pd.DataFrame(resultats)
    assert len(api) == len(attendu), f"{len(api)} vols côté API, {len(attendu)} côté notebook"
    accord = api["niveau"].to_numpy() == attendu["niveau"].to_numpy()
    ecart_proba = np.abs(api["proba_usure"].to_numpy() - attendu["proba_usure"].to_numpy()).max()
    print(f"{len(api)} vols rejoués pour {args.moteur}")
    print(f"Niveaux identiques au notebook : {accord.sum()}/{len(accord)}")
    print(f"Écart max sur la proba d'usure : {ecart_proba:.2e}")
    if not accord.all():
        differences = api.loc[~accord, ["cycle", "niveau"]]
        print(differences.assign(notebook=attendu["niveau"].to_numpy()[~accord]))


if __name__ == "__main__":
    main()
