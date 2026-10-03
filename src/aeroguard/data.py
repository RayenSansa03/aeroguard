"""Outils pour les données N-CMAPSS : étiquettes par vol."""
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
    return (df.groupby(["unit", "cycle"])
              .agg(hs=("hs", "first"), RUL=("RUL", "first"), duree_s=("hs", "size"))
              .reset_index())