import pandas as pd

from aeroguard.features import ajouter_phase, features_par_vol


def faux_vol():
    """1 moteur, 2 vols de 7 secondes : monte, croisière, descend."""
    alt = [0, 5, 10, 10, 10, 5, 0]
    return pd.DataFrame(
        {
            "unit": 1,
            "cycle": [1] * 7 + [2] * 7,
            "alt": alt + alt,
            "S1": [1, 2, 3, 4, 5, 6, 7] + [10, 20, 30, 40, 50, 60, 70],
        }
    )


def test_phases_montee_croisiere_descente():
    df = ajouter_phase(faux_vol())
    attendu = ["montee", "montee", "croisiere", "croisiere", "croisiere", "descente", "descente"]
    assert list(df["phase"].astype(str))[:7] == attendu


def test_une_ligne_par_vol():
    df = ajouter_phase(faux_vol())
    feats = features_par_vol(df, capteurs=["S1"])
    assert len(feats) == 2


def test_noms_aplatis_et_valeurs():
    df = ajouter_phase(faux_vol())
    feats = features_par_vol(df, capteurs=["S1"])
    assert "S1_mean_croisiere" in feats.columns
    vol1 = feats[feats["cycle"] == 1].iloc[0]
    assert vol1["S1_mean_croisiere"] == 4  # moyenne de 3, 4, 5
    assert vol1["S1_max_montee"] == 2  # max de 1, 2
