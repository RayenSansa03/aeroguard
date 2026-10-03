import numpy as np
import pandas as pd
import pytest
from aeroguard.normalisation import ajuster_modele_sain, calculer_residus

W = ["alt", "TRA"]
S = ["S1"]


def faux_moteur(panne_apres=None, ajout=0.0):
    """30 vols × 50 secondes. S1 = 3*alt + 2*TRA² + 10 (+ 'ajout' après la panne)."""
    rng = np.random.default_rng(0)
    n = 30 * 50
    df = pd.DataFrame({
        "unit": 1,
        "cycle": np.repeat(np.arange(1, 31), 50),
        "alt": rng.uniform(0, 10, n),
        "TRA": rng.uniform(0, 5, n),
    })
    df["S1"] = 3 * df["alt"] + 2 * df["TRA"] ** 2 + 10
    if panne_apres is not None:
        df.loc[df["cycle"] > panne_apres, "S1"] += ajout
    return df


def test_moteur_sain_residu_nul():
    df = faux_moteur()
    modele = ajuster_modele_sain(df, W, S)
    res = calculer_residus(df, modele, W, S)
    assert res["S1"].abs().max() == pytest.approx(0, abs=1e-6)


def test_panne_visible_dans_le_residu():
    df = faux_moteur(panne_apres=20, ajout=5.0)
    modele = ajuster_modele_sain(df, W, S)          # n'apprend que sur les vols 1 à 15
    res = calculer_residus(df, modele, W, S)
    assert res.loc[res["cycle"] > 20, "S1"].mean() == pytest.approx(5.0, abs=1e-6)
    assert res.loc[res["cycle"] <= 15, "S1"].mean() == pytest.approx(0.0, abs=1e-6)


def test_garde_les_colonnes_identite():
    df = faux_moteur()
    res = calculer_residus(df, ajuster_modele_sain(df, W, S), W, S)
    assert list(res.columns) == ["unit", "cycle", "S1"]
    assert len(res) == len(df)