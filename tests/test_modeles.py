import numpy as np
import pytest

from aeroguard.modeles import REGLAGES_XGB, creer_xgboost, seuil_optimal


def test_seuil_optimal_separation_parfaite():
    y = [0, 0, 0, 1, 1, 1]
    probas = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    seuil, score = seuil_optimal(y, probas)
    assert 0.3 < seuil <= 0.7
    assert score == pytest.approx(1.0)


def test_seuil_optimal_trouve_un_seuil_bas():
    # les usés ont des probabilités faibles mais toujours au-dessus des sains
    y = [0, 0, 1, 1]
    probas = [0.05, 0.10, 0.20, 0.25]
    seuil, score = seuil_optimal(y, probas)
    assert seuil <= 0.20
    assert score == pytest.approx(1.0)


def test_seuil_optimal_seuils_personnalises():
    seuil, _ = seuil_optimal([0, 1], [0.2, 0.8], seuils=np.array([0.5]))
    assert seuil == 0.5


def test_creer_xgboost_reglages():
    pytest.importorskip("xgboost")
    modele = creer_xgboost(device="cpu", max_depth=6)
    params = modele.get_params()
    assert params["device"] == "cpu"
    assert params["max_depth"] == 6  # remplacé
    assert params["learning_rate"] == REGLAGES_XGB["learning_rate"]  # par défaut
