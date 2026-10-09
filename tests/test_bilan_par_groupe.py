import pandas as pd
import pytest

from aeroguard.metier import bilan_par_groupe, evaluer_alerte_score


def flotte():
    infos = pd.DataFrame(
        {
            "moteur": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "cycle": [1, 2, 3, 4] * 3,
            "RUL": [3, 2, 1, 0] * 3,
        }
    )
    y = [0, 0, 1, 1] * 3
    scores = [0.1, 0.2, 9.0, 9.5, 0.1, 0.3, 0.2, 0.4, 7.0, 0.2, 8.0, 9.0]
    familles = ["fan"] * 4 + ["turbine"] * 8
    return familles, y, scores, infos


def test_une_ligne_par_groupe_triee():
    familles, y, scores, infos = flotte()
    bilan = bilan_par_groupe(familles, y, scores, seuil=5.0, infos=infos, k=1)
    assert list(bilan.index) == ["fan", "turbine"]
    assert bilan.loc["fan", "moteurs"] == 1
    assert bilan.loc["turbine", "moteurs"] == 2


def test_identique_au_bilan_du_sous_groupe():
    familles, y, scores, infos = flotte()
    bilan = bilan_par_groupe(familles, y, scores, seuil=5.0, infos=infos, k=1)
    attendu = evaluer_alerte_score(y[4:], scores[4:], 5.0, infos.iloc[4:], k=1)
    assert bilan.loc["turbine"].to_dict() == pytest.approx(attendu)
    assert bilan.loc["turbine", "moteurs_detectes"] == 1  # B jamais prévenu, C oui
    assert bilan.loc["turbine", "fausses_alarmes_total"] == 1  # le pic de C au 1er vol


def test_longueurs_differentes():
    familles, y, scores, infos = flotte()
    with pytest.raises(ValueError):
        bilan_par_groupe(familles[:-1], y, scores, seuil=5.0, infos=infos)
