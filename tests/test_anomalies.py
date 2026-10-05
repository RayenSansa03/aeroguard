import numpy as np
import pytest

from aeroguard.anomalies import alertes, score_anomalie, seuil_percentile


class FauxModele:
    """Imite un modèle scikit-learn : score_samples renvoie plus grand = plus normal."""

    def score_samples(self, X):
        return np.array([-0.4, -0.6, -0.5])


def test_score_anomalie_inverse_le_signe():
    scores = score_anomalie(FauxModele(), X=None)
    assert list(scores) == pytest.approx([0.4, 0.6, 0.5])


def test_score_anomalie_le_plus_bizarre_a_le_plus_grand_score():
    scores = score_anomalie(FauxModele(), X=None)
    assert scores.argmax() == 1


def test_seuil_percentile():
    sains = np.arange(1, 101)  # 1, 2, ..., 100
    assert seuil_percentile(sains, 95) == pytest.approx(95.05)


def test_seuil_percentile_invalide():
    with pytest.raises(ValueError):
        seuil_percentile([1, 2, 3], 100)
    with pytest.raises(ValueError):
        seuil_percentile([1, 2, 3], 0)


def test_alertes_strictement_au_dessus():
    assert list(alertes([0.2, 0.5, 0.9], seuil=0.5)) == [0, 0, 1]
