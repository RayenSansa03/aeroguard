"""Dérive des données : PSI, référence d'entraînement et rapport."""

import numpy as np
import pytest

from aeroguard.derive import construire_reference, psi, rapport_derive, statut

COLONNES = ["T24_moy", "T48_moy", "Wf_moy"]


@pytest.fixture
def reference():
    X = np.random.default_rng(0).normal(size=(5000, 3))
    return construire_reference(X, COLONNES)


def test_deciles_de_dix_pour_cent(reference):
    for proportions in reference["proportions"]:
        assert sum(proportions) == pytest.approx(1.0)
        assert proportions == pytest.approx([0.1] * 10, abs=0.01)


def test_psi_nul_pour_deux_repartitions_identiques():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_seuils_de_lecture():
    assert statut(0.05) == "stable"
    assert statut(0.15) == "moderate"
    assert statut(0.40) == "major"


def test_meme_distribution_stable(reference):
    X = np.random.default_rng(1).normal(size=(2000, 3))
    rapport = rapport_derive(reference, X)
    assert rapport["status"] == "stable"


def test_capteur_decale_detecte_seul(reference):
    X = np.random.default_rng(2).normal(size=(2000, 3))
    X[:, 1] += 1.0
    rapport = rapport_derive(reference, X)
    assert rapport["status"] == "major"
    assert rapport["n_major"] == 1
    assert rapport["features"][0]["feature"] == "T48_moy"


def test_capteur_mort_impute_detecte(reference):
    X = np.random.default_rng(3).normal(size=(2000, 3))
    X[:, 2] = 0.0  # capteur mort : toujours remplacé par la médiane
    rapport = rapport_derive(reference, X)
    assert rapport["features"][0]["feature"] == "Wf_moy"
    assert rapport["features"][0]["status"] == "major"


def test_trop_peu_de_vols(reference):
    rapport = rapport_derive(reference, np.zeros((10, 3)))
    assert rapport["status"] == "insufficient_data"
    assert rapport["features"] == []
