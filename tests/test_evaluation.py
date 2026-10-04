import pandas as pd
import pytest

from aeroguard.evaluation import (
    ajouter_au_leaderboard,
    appliquer_seuil,
    metriques,
    taux_par_classe,
)

# Exemple du cours : 7 vols (0 = sain, 1 = usé)
Y_VRAI = [0, 0, 0, 1, 1, 1, 1]
Y_PRED = [0, 1, 0, 1, 1, 0, 1]


def test_seuil_par_defaut():
    assert list(appliquer_seuil([0.2, 0.5, 0.9])) == [0, 1, 1]


def test_seuil_bas_et_haut():
    probas = [0.2, 0.4, 0.7]
    assert list(appliquer_seuil(probas, seuil=0.3)) == [0, 1, 1]
    assert list(appliquer_seuil(probas, seuil=0.8)) == [0, 0, 0]


def test_seuil_invalide():
    with pytest.raises(ValueError):
        appliquer_seuil([0.5], seuil=1.5)


def test_taux_par_classe():
    taux = taux_par_classe(Y_VRAI, Y_PRED)
    assert taux["exactitude"] == pytest.approx(5 / 7)
    assert taux["uses_detectes"] == pytest.approx(3 / 4)
    assert taux["sains_reconnus"] == pytest.approx(2 / 3)


def test_leaderboard(tmp_path):
    chemin = tmp_path / "leaderboard.csv"
    ajouter_au_leaderboard(chemin, "bete", {"exactitude": 0.70})
    ajouter_au_leaderboard(chemin, "knn", {"exactitude": 0.80})
    table = pd.read_csv(chemin)
    assert list(table["modele"]) == ["bete", "knn"]


def test_matrice_de_confusion():
    s = metriques(Y_VRAI, Y_PRED)
    assert (s["vn"], s["fp"], s["fn"], s["vp"]) == (2, 1, 1, 3)


def test_precision_rappel_f1():
    s = metriques(Y_VRAI, Y_PRED)
    assert s["precision"] == pytest.approx(0.75)
    assert s["rappel"] == pytest.approx(0.75)
    assert s["f1"] == pytest.approx(0.75)


def test_f1_macro():
    s = metriques(Y_VRAI, Y_PRED)
    assert s["f1_macro"] == pytest.approx((0.75 + 2 / 3) / 2)


def test_modele_jamais_use():
    s = metriques(Y_VRAI, [0] * 7)
    assert s["rappel"] == 0
    assert s["precision"] == 0


def test_pr_auc_parfaite():
    probas = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95]
    s = metriques(Y_VRAI, appliquer_seuil(probas), probas=probas)
    assert s["pr_auc"] == pytest.approx(1.0)
