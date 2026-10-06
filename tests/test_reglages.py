import pytest

from aeroguard.modeles import proposer_reglages_xgb


def faux_essai():
    optuna = pytest.importorskip("optuna")
    return optuna.trial.FixedTrial(
        {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "reg_lambda": 1.0,
        }
    )


def test_proposer_reglages_renvoie_les_7_reglages():
    reglages = proposer_reglages_xgb(faux_essai())
    assert set(reglages) == {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }


def test_proposer_reglages_garde_les_valeurs():
    reglages = proposer_reglages_xgb(faux_essai())
    assert reglages["max_depth"] == 3
    assert reglages["learning_rate"] == pytest.approx(0.05)
