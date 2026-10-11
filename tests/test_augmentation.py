import numpy as np
import pytest

from aeroguard.augmentation import gaussienne, plis_famille_rare, rogner, smote_simple


def test_smote_reste_entre_les_vols_reels():
    X = np.random.default_rng(0).normal(size=(30, 3))
    faux = smote_simple(X, 200, k=3, graine=1)
    assert faux.shape == (200, 3)
    assert (faux >= X.min(axis=0) - 1e-12).all() and (faux <= X.max(axis=0) + 1e-12).all()
    assert np.allclose(faux, smote_simple(X, 200, k=3, graine=1))


def test_smote_sur_une_droite_donne_des_points_de_la_droite():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    faux = smote_simple(X, 50, k=1)
    assert np.allclose(faux[:, 0], faux[:, 1])


def test_gaussienne_reproduit_moyenne_et_correlation():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    X = np.c_[a, 2 * a + 0.1 * rng.normal(size=5000)]
    faux = gaussienne(X, 5000, graine=3)
    assert faux.shape == (5000, 2)
    assert np.allclose(faux.mean(axis=0), X.mean(axis=0), atol=0.1)
    assert np.corrcoef(faux, rowvar=False)[0, 1] > 0.95


def test_entrees_invalides():
    with pytest.raises(ValueError):
        smote_simple(np.ones((1, 3)), 5)
    with pytest.raises(ValueError):
        gaussienne(np.ones((4, 2)), 0)
    with pytest.raises(ValueError):
        rogner(np.ones((3, 2)), np.ones((3, 4)))


def test_rogner_aux_limites_des_vrais_vols():
    reel = np.array([[0.0, 10.0], [1.0, 20.0]])
    faux = np.array([[-5.0, 15.0], [0.5, 99.0]])
    assert rogner(faux, reel).tolist() == [[0.0, 15.0], [0.5, 20.0]]


def test_plis_famille_rare():
    moteurs = np.repeat(np.arange(10), 3)
    familles = np.repeat(["fan"] * 3 + ["hpt"] * 7, 3)
    plis = plis_famille_rare(moteurs, familles, "fan", graine=0)
    assert len(plis) == 3
    for masque in plis:
        assert len(np.unique(moteurs[masque & (familles == "fan")])) == 1
    couverture = np.sum(plis, axis=0)
    assert (couverture == 1).all()  # chaque vol est évalué exactement une fois
    with pytest.raises(ValueError):
        plis_famille_rare(moteurs, familles, "absente")
