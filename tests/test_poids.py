import numpy as np
import pytest
from sklearn.utils.class_weight import compute_class_weight

from aeroguard.donnees_ml import poids_classes


def test_poids_exemple_simple():
    # 1 sain, 3 usés → sain = 4/(2×1) = 2 ; usé = 4/(2×3) ≈ 0,667
    poids = poids_classes([0, 1, 1, 1])
    assert poids[0] == pytest.approx(2.0)
    assert poids[1] == pytest.approx(4 / 6)


def test_classe_rare_plus_lourde():
    poids = poids_classes([0, 1, 1, 1])
    assert poids[0] > poids[1]


def test_classes_equilibrees_poids_un():
    poids = poids_classes([0, 0, 1, 1])
    assert poids == {0: 1.0, 1: 1.0}


def test_les_deux_classes_pesent_autant():
    y = np.array([0] * 30 + [1] * 70)
    poids = poids_classes(y)
    assert poids[0] * 30 == pytest.approx(poids[1] * 70)


def test_identique_a_sklearn():
    y = np.array([0] * 11 + [1] * 26)
    attendu = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    poids = poids_classes(y)
    assert [poids[0], poids[1]] == pytest.approx(list(attendu))


def test_classe_absente():
    with pytest.raises(ValueError):
        poids_classes([1, 1, 1])
