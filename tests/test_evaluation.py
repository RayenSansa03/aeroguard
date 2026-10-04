import pandas as pd
import pytest

from aeroguard.evaluation import ajouter_au_leaderboard, appliquer_seuil, taux_par_classe


def test_seuil_par_defaut():
    assert list(appliquer_seuil([0.2, 0.5, 0.8])) == [0, 1, 1]


def test_seuil_bas_et_haut():
    probas = [0.2, 0.4, 0.8]
    assert list(appliquer_seuil(probas, seuil=0.3)) == [0, 1, 1]
    assert list(appliquer_seuil(probas, seuil=0.9)) == [0, 0, 0]


def test_seuil_invalide():
    with pytest.raises(ValueError):
        appliquer_seuil([0.5], seuil=1.5)


def test_taux_par_classe():
    scores = taux_par_classe(y_vrai=[0, 0, 1, 1], y_pred=[0, 1, 1, 1])
    assert scores["exactitude"] == pytest.approx(0.75)
    assert scores["uses_detectes"] == pytest.approx(1.0)
    assert scores["sains_reconnus"] == pytest.approx(0.5)


def test_leaderboard_ajoute_une_ligne_a_chaque_fois(tmp_path):
    chemin = tmp_path / "leaderboard.csv"
    ajouter_au_leaderboard(chemin, "bete", {"exactitude": 0.70})
    ajouter_au_leaderboard(chemin, "logistique", {"exactitude": 0.82})
    tableau = pd.read_csv(chemin)
    assert list(tableau["modele"]) == ["bete", "logistique"]
