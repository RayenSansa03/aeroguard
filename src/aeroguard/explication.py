"""Expliquer les modèles : noms de features lisibles, importances, contributions SHAP."""

import numpy as np
import pandas as pd

STATS_FR = {"mean": "moyenne", "std": "variation", "max": "maximum"}
PHASES_FR = {"montee": "en montée", "croisiere": "en croisière", "descente": "en descente"}


def decrire_feature(nom):
    """'T48_mean_croisiere' → 'T48 moyenne en croisière'."""
    morceaux = nom.split("_")
    if len(morceaux) != 3 or morceaux[1] not in STATS_FR or morceaux[2] not in PHASES_FR:
        raise ValueError(f"Nom de feature inattendu : {nom}")
    capteur, stat, phase = morceaux
    return f"{capteur} {STATS_FR[stat]} {PHASES_FR[phase]}"


def importances_triees(importances, colonnes, n=10):
    """Les n features les plus importantes, de la plus forte à la plus faible."""
    if len(importances) != len(colonnes):
        raise ValueError("importances et colonnes doivent avoir la même longueur.")
    table = pd.DataFrame({"feature": list(colonnes), "importance": list(importances)})
    table = table.sort_values("importance", ascending=False).head(n).reset_index(drop=True)
    table["description"] = table["feature"].map(decrire_feature)
    return table


def importance_par_capteur(importances, colonnes):
    """Additionne les importances des 9 features de chaque capteur (3 stats × 3 phases)."""
    if len(importances) != len(colonnes):
        raise ValueError("importances et colonnes doivent avoir la même longueur.")
    serie = pd.Series(np.asarray(importances, dtype=float), index=list(colonnes))
    par_capteur = serie.groupby(lambda nom: nom.split("_")[0]).sum()
    return par_capteur.sort_values(ascending=False)


def expliquer_vol(contributions, colonnes, n=5):
    """Les n features qui ont le plus pesé sur la décision pour UN vol (valeurs SHAP)."""
    if len(contributions) != len(colonnes):
        raise ValueError("contributions et colonnes doivent avoir la même longueur.")
    table = pd.DataFrame(
        {"feature": list(colonnes), "contribution": np.asarray(contributions, dtype=float)}
    )
    ordre = table["contribution"].abs().sort_values(ascending=False).index[:n]
    table = table.loc[ordre].reset_index(drop=True)
    table["description"] = table["feature"].map(decrire_feature)
    table["sens"] = np.where(table["contribution"] > 0, "pour", "contre")
    return table
