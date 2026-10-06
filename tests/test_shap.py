import pytest

from aeroguard.explication import expliquer_vol, importance_par_capteur

COLONNES = ["T48_mean_montee", "T48_max_croisiere", "Nc_std_descente", "P24_mean_montee"]


def test_importance_par_capteur_additionne():
    resultat = importance_par_capteur([0.5, 0.25, 0.1, 0.05], COLONNES)
    assert resultat["T48"] == pytest.approx(0.75)  # 0,5 + 0,25
    assert resultat.index[0] == "T48"  # le plus important en premier
    assert len(resultat) == 3


def test_importance_par_capteur_longueurs_differentes():
    with pytest.raises(ValueError):
        importance_par_capteur([0.1], COLONNES)


def test_expliquer_vol_trie_par_valeur_absolue():
    table = expliquer_vol([0.3, -2.0, 0.1, 1.0], COLONNES, n=2)
    assert list(table["feature"]) == ["T48_max_croisiere", "P24_mean_montee"]
    assert list(table["sens"]) == ["contre", "pour"]
    assert table.loc[0, "description"] == "T48 maximum en croisière"
