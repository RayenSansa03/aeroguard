import pandas as pd

from aeroguard.donnees_ml import preparer_xy_famille
from aeroguard.metier import diagnostic_par_moteur


def faux_vols():
    return pd.DataFrame(
        {
            "groupe": ["train", "train", "train", "val"],
            "hs": [1, 0, 0, 0],  # 1 = sain, 0 = usé
            "famille": ["HPT", "HPT", "Fan", "LPT"],
            "T48_mean_montee": [0.1, 0.9, 0.2, 0.8],
            "P24_max_croisiere": [0.0, 0.1, -0.7, 0.2],
        }
    )


def test_garde_seulement_les_vols_uses_du_groupe():
    X, y = preparer_xy_famille(faux_vols(), "train")
    assert len(X) == 2
    assert list(y) == ["HPT", "Fan"]


def test_pas_de_fuite_dans_X():
    X, _ = preparer_xy_famille(faux_vols(), "train")
    assert list(X.columns) == ["T48_mean_montee", "P24_max_croisiere"]


def test_diagnostic_par_moteur_vote_majoritaire():
    moteurs = ["A", "A", "A", "B", "B"]
    predictions = ["HPT", "HPT", "LPT", "Fan", "Fan"]
    diag = diagnostic_par_moteur(moteurs, predictions)
    assert diag["A"] == "HPT"
    assert diag["B"] == "Fan"
