import numpy as np
import pandas as pd
import pytest

from aeroguard.metier import (
    alertes_confirmees,
    avance_alerte,
    bilan_flotte,
    fausses_alarmes,
    resume_flotte,
)

# Exemple du cours : 10 vols, usé à partir du 5e
ALERTES = [0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
Y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
RUL = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_alertes_confirmees_k2():
    assert list(alertes_confirmees([1, 1, 0, 1, 1, 1], k=2)) == [0, 1, 0, 0, 1, 1]


def test_alertes_confirmees_k_invalide():
    with pytest.raises(ValueError):
        alertes_confirmees([1, 0], k=0)


def test_avance_k1():
    assert avance_alerte(ALERTES, Y, RUL, k=1) == 4


def test_avance_k2_un_vol_de_retard():
    assert avance_alerte(ALERTES, Y, RUL, k=2) == 3


def test_moteur_rate():
    assert np.isnan(avance_alerte([0] * 10, Y, RUL))


def test_fausses_alarmes_k1_et_k2():
    assert fausses_alarmes(ALERTES, Y, k=1) == 1
    assert fausses_alarmes(ALERTES, Y, k=2) == 0


def test_bilan_et_resume_flotte():
    m1 = pd.DataFrame({"unit": "A", "cycle": range(1, 11), "RUL": RUL, "y": Y, "alerte": ALERTES})
    m2 = pd.DataFrame({"unit": "B", "cycle": range(1, 11), "RUL": RUL, "y": Y, "alerte": [0] * 10})
    table = pd.concat([m2, m1]).sample(frac=1, random_state=0)  # ordre mélangé exprès
    bilan = bilan_flotte(table, k=1)
    assert list(bilan["moteur"]) == ["A", "B"]
    assert bilan.loc[0, "avance"] == 4
    resume = resume_flotte(bilan)
    assert resume["moteurs_detectes"] == 1
    assert resume["fausses_alarmes_total"] == 1
