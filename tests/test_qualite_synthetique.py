import numpy as np
import pytest

from aeroguard.qualite_synthetique import (
    auc_separabilite,
    bilan_fidelite,
    mesurer_memorisation,
    resume_fidelite,
    score_classifieur,
    statistique_ks,
)


def test_statistique_ks_cas_simples():
    assert statistique_ks([1, 2, 3], [1, 2, 3]) == 0.0
    assert statistique_ks([0, 0, 0], [5, 5, 5]) == 1.0
    assert statistique_ks([0, 1], [1, 2]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        statistique_ks([], [1])


def test_bilan_fidelite_repere_la_feature_mal_reproduite():
    rng = np.random.default_rng(0)
    reel = rng.normal(size=(500, 3))
    faux = rng.normal(size=(500, 3))
    faux[:, 1] += 2.0  # feature décalée
    bilan = bilan_fidelite(reel, faux, colonnes=["a", "b", "c"])
    assert bilan.index[0] == "b"
    assert bilan.loc["b", "ecart_moyennes"] == pytest.approx(2.0, abs=0.2)
    assert bilan.loc["a", "ks"] < 0.15
    assert bilan.loc["a", "rapport_iqr"] == pytest.approx(1.0, abs=0.2)
    resume = resume_fidelite(bilan)
    assert resume["part_ks_eleve"] == pytest.approx(1 / 3)


def test_rapport_iqr_insensible_aux_valeurs_aberrantes():
    rng = np.random.default_rng(1)
    reel = rng.normal(size=(1000, 1))
    reel[0, 0] = 1000.0  # une seule valeur aberrante
    faux = rng.normal(size=(1000, 1))
    bilan = bilan_fidelite(reel, faux)
    assert bilan["rapport_iqr"].iloc[0] == pytest.approx(1.0, abs=0.15)


def test_bilan_fidelite_entrees_invalides():
    with pytest.raises(ValueError):
        bilan_fidelite(np.ones((5, 2)), np.ones((5, 3)))
    with pytest.raises(ValueError):
        bilan_fidelite(np.ones((5, 2)), np.ones((5, 2)), colonnes=["seule"])


def test_memorisation_detecte_les_copies():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(300, 4))
    reference = rng.normal(size=(100, 4))
    copies = train[:100] + 1e-6
    nouveaux = rng.normal(size=(100, 4))
    avec_copies = mesurer_memorisation(train, copies, reference)
    sans_copies = mesurer_memorisation(train, nouveaux, reference)
    assert avec_copies["part_quasi_copies"] == 1.0
    assert avec_copies["rapport_dcr"] < 0.01
    assert sans_copies["part_quasi_copies"] < 0.1
    assert sans_copies["rapport_dcr"] == pytest.approx(1.0, abs=0.3)


def test_auc_separabilite():
    rng = np.random.default_rng(3)
    reel = rng.normal(size=(200, 3))
    pareil = rng.normal(size=(200, 3))
    different = rng.normal(loc=3.0, size=(200, 3))
    assert auc_separabilite(reel, pareil) == pytest.approx(0.5, abs=0.1)
    assert auc_separabilite(reel, different) > 0.95


def test_score_classifieur():
    rng = np.random.default_rng(4)
    X = np.r_[rng.normal(size=(100, 2)), rng.normal(loc=4.0, size=(100, 2))]
    y = np.r_[np.zeros(100), np.ones(100)]
    scores = score_classifieur(X[::2], y[::2], X[1::2], y[1::2])
    assert scores["pr_auc"] > 0.95
    assert scores["exactitude_equilibree"] > 0.9
