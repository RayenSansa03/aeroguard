import numpy as np
import pytest

from aeroguard.diversite import diagnostic_diversite, distances_plus_proche_voisin, est_effondre


def test_distances_plus_proche_voisin_dans_le_meme_ensemble():
    A = np.array([[0.0], [1.0], [3.0]])
    assert distances_plus_proche_voisin(A).tolist() == [1.0, 1.0, 2.0]


def test_distances_plus_proche_voisin_vers_un_autre_ensemble():
    A = np.array([[0.0, 0.0], [5.0, 0.0]])
    B = np.array([[3.0, 4.0], [5.0, 1.0]])
    assert np.allclose(distances_plus_proche_voisin(A, B), [5.0, 1.0])


def test_distances_entrees_invalides():
    with pytest.raises(ValueError):
        distances_plus_proche_voisin(np.ones((1, 3)))
    with pytest.raises(ValueError):
        distances_plus_proche_voisin(np.ones((3, 2)), np.ones((3, 4)))


def test_diagnostic_faux_identiques_aux_vrais():
    reel = np.random.default_rng(0).normal(size=(400, 5))
    diag = diagnostic_diversite(reel, reel.copy())
    assert diag["rapport_ecarts"] == pytest.approx(1.0)
    assert diag["rapport_ppv"] == pytest.approx(1.0)
    assert diag["couverture_queues"] == pytest.approx(1.0, abs=0.1)
    assert diag["part_doublons"] == 0.0
    assert not est_effondre(diag)


def test_diagnostic_mode_collapse_total():
    reel = np.random.default_rng(1).normal(size=(200, 4))
    faux = np.tile(reel[:1], (200, 1))  # toujours le même vol
    diag = diagnostic_diversite(reel, faux)
    assert diag["rapport_ecarts"] == pytest.approx(0.0)
    assert diag["part_features_ecrasees"] == 1.0
    assert diag["part_doublons"] == 1.0
    assert est_effondre(diag)


def test_diagnostic_faux_trop_sages_perdent_les_queues():
    rng = np.random.default_rng(2)
    reel = rng.normal(size=(2000, 3))
    faux = 0.3 * rng.normal(size=(2000, 3))  # même centre, 3 fois moins étalés
    diag = diagnostic_diversite(reel, faux)
    assert diag["rapport_ecarts"] == pytest.approx(0.3, abs=0.05)
    assert diag["couverture_queues"] < 0.05
    assert est_effondre(diag)


def test_diagnostic_entrees_invalides():
    with pytest.raises(ValueError):
        diagnostic_diversite(np.ones((5, 2)), np.ones((5, 3)))
    with pytest.raises(ValueError):
        diagnostic_diversite(np.ones((5, 2)), np.ones((5, 2)), quantile_queue=0.5)
