import pytest

from aeroguard.explication import decrire_feature, importances_triees


def test_decrire_feature():
    assert decrire_feature("T48_mean_croisiere") == "T48 moyenne en croisière"
    assert decrire_feature("Nc_std_montee") == "Nc variation en montée"


def test_decrire_feature_invalide():
    with pytest.raises(ValueError):
        decrire_feature("hs")
    with pytest.raises(ValueError):
        decrire_feature("T48_min_croisiere")


def test_importances_triees_ordre():
    colonnes = ["T48_mean_croisiere", "Wf_max_montee", "Nc_std_descente"]
    table = importances_triees([0.2, 0.5, 0.3], colonnes)
    assert list(table["feature"]) == ["Wf_max_montee", "Nc_std_descente", "T48_mean_croisiere"]


def test_importances_triees_n():
    colonnes = ["T48_mean_croisiere", "Wf_max_montee", "Nc_std_descente"]
    table = importances_triees([0.2, 0.5, 0.3], colonnes, n=2)
    assert len(table) == 2
    assert table.loc[0, "description"] == "Wf maximum en montée"


def test_importances_longueurs_differentes():
    with pytest.raises(ValueError):
        importances_triees([0.1, 0.9], ["T48_mean_croisiere"])
