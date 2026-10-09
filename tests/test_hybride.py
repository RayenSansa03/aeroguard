import numpy as np
import pytest

from aeroguard.hybride import centrer_par_moteur, etat_hybride


def test_chaque_moteur_compare_a_ses_premiers_vols():
    X = np.array([[10.0], [12.0], [20.0], [100.0], [102.0], [110.0]])
    moteurs = ["A", "A", "A", "B", "B", "B"]
    cycles = [1, 2, 3, 1, 2, 3]
    centre = centrer_par_moteur(X, moteurs, cycles, n_premiers=2)
    # A : ligne de base 11 ; B : ligne de base 101 → la flotte (10 contre 100) disparaît
    assert np.allclose(centre.ravel(), [-1, 1, 9, -1, 1, 9])


def test_ordre_des_lignes_sans_importance():
    X = np.array([[20.0], [10.0], [12.0]])
    centre = centrer_par_moteur(X, ["A", "A", "A"], [3, 1, 2], n_premiers=2)
    assert np.allclose(centre.ravel(), [9, -1, 1])  # base = vols des cycles 1 et 2


def test_moteur_plus_court_que_n():
    centre = centrer_par_moteur(np.array([[4.0], [6.0]]), ["A", "A"], [1, 2], n_premiers=10)
    assert np.allclose(centre.ravel(), [-1, 1])  # base = les 2 vols disponibles


@pytest.mark.parametrize("n_premiers, moteurs", [(0, ["A", "A"]), (2, ["A"])])
def test_parametres_invalides(n_premiers, moteurs):
    with pytest.raises(ValueError):
        centrer_par_moteur(np.zeros((2, 1)), moteurs, [1, 2], n_premiers=n_premiers)


def test_les_quatre_etats():
    etats = etat_hybride([0, 0, 1, 1], [0, 1, 0, 1])
    assert list(etats) == ["normal", "anomalie inconnue", "usure connue", "usure confirmée"]


def test_formes_differentes():
    with pytest.raises(ValueError):
        etat_hybride([0, 1], [0])
