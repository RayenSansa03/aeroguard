"""Features : résumer chaque vol en une seule ligne."""
import numpy as np
import pandas as pd

STATS = ["mean", "std", "max"]
PHASES = ["montee", "croisiere", "descente"]


def ajouter_phase(df, seuil=0.9):
    """Ajoute une colonne 'phase' : montee / croisiere / descente."""
    df = df.copy()
    vol = [df["unit"], df["cycle"]]
    t = df.groupby(["unit", "cycle"]).cumcount()                  # seconde n° 0, 1, 2… dans le vol
    haut = df["alt"] >= seuil * df.groupby(["unit", "cycle"])["alt"].transform("max")
    debut = t.where(haut).groupby(vol).transform("min")           # 1re seconde de croisière
    fin = t.where(haut).groupby(vol).transform("max")             # dernière seconde de croisière
    phase = np.where(t < debut, "montee", np.where(t > fin, "descente", "croisiere"))
    df["phase"] = pd.Categorical(phase, categories=PHASES)
    return df


def features_par_vol(df, capteurs, stats=STATS):
    """Une ligne par vol, colonnes '<capteur>_<stat>_<phase>'."""
    agg = (df.groupby(["unit", "cycle", "phase"], observed=True)[capteurs]
             .agg(stats)
             .unstack("phase"))
    agg.columns = [f"{c}_{s}_{p}" for c, s, p in agg.columns]
    return agg.reset_index()
