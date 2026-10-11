"""Rejouer des vols de test dans l'API, moteur par moteur, comme s'ils arrivaient en direct."""

import pandas as pd

from aeroguard.donnees_ml import preparer_xy


def vols_par_moteur(vols, partie="test"):
    """Moteur -> (cycles triés, features dans le même ordre)."""
    X, _ = preparer_xy(vols, partie)
    infos = vols.loc[X.index, ["moteur", "cycle"]]
    resultat = {}
    for moteur, groupe in infos.groupby("moteur"):
        ordre = groupe.sort_values("cycle")
        resultat[str(moteur)] = (ordre["cycle"].to_numpy(), X.loc[ordre.index])
    return resultat


def en_mesures(ligne):
    """Une ligne de features en dictionnaire JSON (NaN devient None, que l'API impute)."""
    return {str(c): (None if pd.isna(v) else float(v)) for c, v in ligne.items()}


def envoyer_vols(client, moteur, cycles, X, jusqu_a=None):
    """Réinitialise le moteur, envoie ses vols dans l'ordre et renvoie les recommandations."""
    client.delete(f"/flotte/moteurs/{moteur}")
    fin = len(cycles) if jusqu_a is None else min(jusqu_a, len(cycles))
    recommandations = []
    for cycle, (_, ligne) in zip(cycles[:fin], X.iloc[:fin].iterrows(), strict=True):
        vol = {"moteur": moteur, "cycle": int(cycle), "mesures": en_mesures(ligne)}
        reponse = client.post("/flotte/vols", json=vol)
        reponse.raise_for_status()
        recommandations.append(reponse.json())
    return recommandations
