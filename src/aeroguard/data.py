"""Outils pour les données N-CMAPSS : étiquettes par vol."""

import h5py
import numpy as np
import pandas as pd

COMPOSANTS = ["fan", "LPC", "HPC", "HPT", "LPT"]
COLONNES_T = [f"{c}_{t}_mod" for c in COMPOSANTS for t in ["eff", "flow"]]


def composant_touche(ligne, seuil=1e-3):
    """Retourne le composant le plus dégradé, ou 'aucun' si tout est sous le seuil."""
    meilleur, pire_valeur = "aucun", seuil
    for c in COMPOSANTS:
        degradation = max(abs(ligne[f"{c}_eff_mod"]), abs(ligne[f"{c}_flow_mod"]))
        if degradation > pire_valeur:
            meilleur, pire_valeur = c, degradation
    return meilleur


def table_vols(df):
    """Résume un DataFrame 'une ligne par seconde' en 'une ligne par vol'."""
    return (
        df.groupby(["unit", "cycle"])
        .agg(hs=("hs", "first"), RUL=("RUL", "first"), duree_s=("hs", "size"))
        .reset_index()
    )


def lire_noms(f, nom):
    """Lit une liste de noms dans le fichier HDF5 (bytes → texte)."""
    valeurs = f[nom][()]
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.ravel(valeurs)]


def charger_ncmapss(chemin, partie="dev"):
    """Charge un fichier N-CMAPSS (partie 'dev' ou 'test') dans un DataFrame."""
    with h5py.File(chemin, "r") as f:
        blocs = [f[f"{t}_{partie}"][()] for t in ["A", "W", "X_s", "T", "Y"]]
        colonnes = (
            lire_noms(f, "A_var")
            + lire_noms(f, "W_var")
            + lire_noms(f, "X_s_var")
            + lire_noms(f, "T_var")
            + ["RUL"]
        )
    df = pd.DataFrame(np.hstack(blocs).astype("float32"), columns=colonnes)
    for col in ["unit", "cycle", "Fc", "hs", "RUL"]:
        df[col] = df[col].astype("int32")
    return df


def mode_de_panne(ligne, part_min=0.2, seuil=1e-3):
    """Tous les composants abîmés (≥ part_min de la pire dégradation), ex. 'HPT+LPT'."""
    degradation = {
        c: max(abs(ligne[f"{c}_eff_mod"]), abs(ligne[f"{c}_flow_mod"])) for c in COMPOSANTS
    }
    pire = max(degradation.values())
    if pire <= seuil:
        return "aucun"
    return "+".join(c for c in COMPOSANTS if degradation[c] >= part_min * pire)
