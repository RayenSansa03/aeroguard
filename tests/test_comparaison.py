import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from aeroguard.comparaison import (
    bootstrap_moteurs,
    comparer_a_reference,
    comptes_par_moteur,
    f1_macro_depuis_comptes,
    intervalle_confiance,
)


def petite_flotte(graine=0):
    """3 moteurs × 10 vols : les 4 premiers vols sains, les 6 suivants usés."""
    rng = np.random.default_rng(graine)
    moteurs = np.repeat(["M1", "M2", "M3"], 10)
    y = np.tile([0] * 4 + [1] * 6, 3)
    pred = np.where(rng.random(30) < 0.8, y, 1 - y)  # 80 % de bonnes réponses
    return moteurs, y, pred


def test_f1_macro_depuis_comptes_egal_sklearn():
    _, y, pred = petite_flotte()
    vp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    vn = int(((y == 0) & (pred == 0)).sum())
    attendu = f1_score(y, pred, average="macro", zero_division=0)
    assert f1_macro_depuis_comptes(vp, fp, fn, vn) == pytest.approx(attendu)


def test_f1_macro_depuis_comptes_sans_division_par_zero():
    # Aucun vol sain et aucune alerte « sain » : le F1 sain vaut 0, pas NaN
    assert f1_macro_depuis_comptes(5, 0, 0, 0) == pytest.approx(0.5)


def test_comptes_par_moteur():
    moteurs, y, pred = petite_flotte()
    comptes = comptes_par_moteur(moteurs, y, pred)
    assert list(comptes.index) == ["M1", "M2", "M3"]
    assert list(comptes.columns) == ["vp", "fp", "fn", "vn"]
    assert comptes.to_numpy().sum() == 30
    assert (comptes.sum(axis=1) == 10).all()


def test_bootstrap_forme_et_reproductible():
    moteurs, y, pred = petite_flotte()
    predictions = {"a": pred, "b": y}
    t1 = bootstrap_moteurs(moteurs, y, predictions, n_tirages=50, graine=1)
    t2 = bootstrap_moteurs(moteurs, y, predictions, n_tirages=50, graine=1)
    assert t1.shape == (50, 2)
    assert list(t1.columns) == ["a", "b"]
    pd.testing.assert_frame_equal(t1, t2)


def test_bootstrap_modele_parfait_vaut_1():
    moteurs, y, _ = petite_flotte()
    tirages = bootstrap_moteurs(moteurs, y, {"parfait": y}, n_tirages=20)
    assert np.allclose(tirages["parfait"], 1.0)


def test_bootstrap_moyenne_proche_du_score_global():
    moteurs, y, pred = petite_flotte()
    tirages = bootstrap_moteurs(moteurs, y, {"a": pred}, n_tirages=2000)
    score_global = f1_score(y, pred, average="macro")
    assert tirages["a"].mean() == pytest.approx(score_global, abs=0.05)


def test_bootstrap_longueurs_differentes():
    moteurs, y, pred = petite_flotte()
    with pytest.raises(ValueError):
        bootstrap_moteurs(moteurs, y, {"a": pred[:-1]})


@pytest.mark.parametrize("n_tirages, predictions", [(0, {"a": [0] * 30}), (10, {})])
def test_bootstrap_parametres_invalides(n_tirages, predictions):
    moteurs, y, _ = petite_flotte()
    with pytest.raises(ValueError):
        bootstrap_moteurs(moteurs, y, predictions, n_tirages=n_tirages)


def test_intervalle_confiance():
    bas, haut = intervalle_confiance(np.arange(101), niveau=0.95)
    assert bas == pytest.approx(2.5)
    assert haut == pytest.approx(97.5)


@pytest.mark.parametrize("niveau", [0, 1, 1.5])
def test_intervalle_confiance_niveau_invalide(niveau):
    with pytest.raises(ValueError):
        intervalle_confiance([0.1, 0.2], niveau=niveau)


def test_comparer_a_reference():
    tirages = pd.DataFrame({"ref": [0.80, 0.82, 0.84], "meilleur": [0.90, 0.91, 0.95]})
    bilan = comparer_a_reference(tirages, "ref")
    assert bilan.loc["ref", "ecart_moyen"] == pytest.approx(0.0)
    assert not bilan.loc["ref", "difference_prouvee"]
    assert bilan.loc["meilleur", "proba_meilleur"] == pytest.approx(1.0)
    assert bilan.loc["meilleur", "difference_prouvee"]


def test_comparer_a_reference_absente():
    with pytest.raises(ValueError):
        comparer_a_reference(pd.DataFrame({"a": [0.5]}), "b")
