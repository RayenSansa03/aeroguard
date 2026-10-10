"""Mesurer la rareté des cas : effectifs par classe, déséquilibre et zone critique."""

import numpy as np
import pandas as pd


def bilan_classes(etiquettes):
    """Effectif et proportion de chaque classe, de la plus rare à la plus fréquente."""
    serie = pd.Series(np.asarray(etiquettes))
    if serie.empty:
        raise ValueError("Il faut au moins une étiquette.")
    comptes = serie.value_counts()
    table = pd.DataFrame(
        {
            "classe": comptes.index,
            "effectif": comptes.to_numpy(),
            "proportion": comptes.to_numpy() / comptes.sum(),
        }
    )
    table = table.sort_values(["effectif", "classe"], kind="stable")
    return table.set_index("classe")


def ratio_desequilibre(etiquettes):
    """Effectif de la classe la plus fréquente divisé par celui de la plus rare."""
    effectifs = bilan_classes(etiquettes)["effectif"]
    return float(effectifs.max() / effectifs.min())


def zone_critique(rul, seuil=10):
    """True pour les vols à `seuil` vols ou moins de la panne (RUL <= seuil)."""
    if seuil < 0:
        raise ValueError("Le seuil doit être positif ou nul.")
    return np.asarray(rul) <= seuil
