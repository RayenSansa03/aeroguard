import pandas as pd

from aeroguard.data import COLONNES_T
from aeroguard.donnees_ml import colonnes_features, preparer_xy


def fausse_table():
    """2 features + toutes les colonnes 'réponses' qu'on ne doit JAMAIS mettre dans X."""
    return pd.DataFrame(
        {
            "dataset": "DS01",
            "moteur": ["DS01_1", "DS01_1", "DS01_2", "DS01_3"],
            "unit": [1, 1, 2, 3],
            "cycle": [1, 2, 1, 1],
            "T48_mean_montee": [0.1, 2.0, 0.2, 0.3],
            "Wf_max_croisiere": [0.0, 0.5, 0.1, 0.2],
            "hs": [1, 0, 1, 1],
            "RUL": [1, 0, 5, 7],
            "duree_s": [100, 110, 90, 95],
            "Fc": [1, 1, 2, 3],
            "mode": "HPT",
            "famille": "HPT",
            "groupe": ["train", "train", "val", "test"],
        }
    )


def test_seules_les_features_sont_gardees():
    assert colonnes_features(fausse_table()) == ["T48_mean_montee", "Wf_max_croisiere"]


def test_aucune_fuite_dans_X():
    X, _ = preparer_xy(fausse_table(), "train")
    interdites = {"hs", "RUL", "mode", "famille", "groupe", "unit", "cycle", *COLONNES_T}
    assert interdites.isdisjoint(X.columns)


def test_y_vaut_1_pour_les_vols_uses():
    _, y = preparer_xy(fausse_table(), "train")
    assert list(y) == [0, 1]  # hs = 1 → 0 (sain), hs = 0 → 1 (usé)


def test_un_groupe_a_la_fois():
    X, y = preparer_xy(fausse_table(), "val")
    assert len(X) == len(y) == 1
