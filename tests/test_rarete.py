import numpy as np
import pytest

from aeroguard.rarete import bilan_classes, ratio_desequilibre, zone_critique


def test_bilan_classes_trie_de_la_plus_rare_a_la_plus_frequente():
    table = bilan_classes(["sain"] * 6 + ["hpt"] * 3 + ["fan"])
    assert list(table.index) == ["fan", "hpt", "sain"]
    assert list(table["effectif"]) == [1, 3, 6]


def test_bilan_classes_proportions_somment_a_un():
    table = bilan_classes([0, 0, 0, 1])
    assert table["proportion"].sum() == pytest.approx(1.0)
    assert table.loc[1, "proportion"] == pytest.approx(0.25)


def test_bilan_classes_vide():
    with pytest.raises(ValueError):
        bilan_classes([])


def test_ratio_desequilibre():
    assert ratio_desequilibre(["a"] * 9 + ["b"] * 3) == pytest.approx(3.0)
    assert ratio_desequilibre(["a", "b"]) == pytest.approx(1.0)


def test_zone_critique():
    rul = np.array([50, 11, 10, 3, 0])
    assert zone_critique(rul).tolist() == [False, False, True, True, True]
    assert zone_critique(rul, seuil=3).tolist() == [False, False, False, True, True]


def test_zone_critique_seuil_negatif():
    with pytest.raises(ValueError):
        zone_critique([1, 2], seuil=-1)
