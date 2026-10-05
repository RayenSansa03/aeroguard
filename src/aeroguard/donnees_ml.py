"""Préparer les données pour le machine learning : X, y et poids des classes."""

import numpy as np

STATS = ("mean", "std", "max")
PHASES = ("montee", "croisiere", "descente")


def colonnes_features(vols):
    """Les 126 colonnes de features (capteur_stat_phase)."""
    return [
        c
        for c in vols.columns
        if c.count("_") == 2 and c.split("_")[1] in STATS and c.split("_")[2] in PHASES
    ]


def preparer_xy(vols, groupe):
    """X (features) et y (1 = usé, 0 = sain) pour un groupe : train, val ou test."""
    sous = vols[vols["groupe"] == groupe]
    X = sous[colonnes_features(vols)]
    y = (1 - sous["hs"]).astype(int)
    return X, y


def poids_classes(y):
    """Poids 'balanced' : n / (2 × n_classe). Une classe rare reçoit un poids plus fort."""
    y = np.asarray(y)
    n = len(y)
    poids = {}
    for classe in (0, 1):
        n_classe = int((y == classe).sum())
        if n_classe == 0:
            raise ValueError(f"La classe {classe} est absente : impossible de calculer son poids.")
        poids[classe] = n / (2 * n_classe)
    return poids
