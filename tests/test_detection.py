"""Tests automatiques des fonctions de détection."""

import pytest

from aeroguard.detection import ecart_absolu, score_z, detecter_anomalies, etat_moteur


def test_ecart_absolu_toujours_positif():
    resultat = ecart_absolu([510, 530], 520)
    assert list(resultat) == [10, 10]


def test_score_z_valeur_normale_donne_zero():
    z = score_z([520], [518, 522, 519, 521, 520])
    assert z[0] == pytest.approx(0)


def test_score_z_valeur_haute_donne_score_positif():
    z = score_z([600], [518, 522, 519, 521, 520])
    assert z[0] == pytest.approx(56.57, abs=0.01)


def test_detecter_anomalies_trop_haut_et_trop_bas():
    resultat = detecter_anomalies([0, 56.6, -35.4])
    assert list(resultat) == [False, True, True]


def test_etat_moteur_trois_niveaux():
    assert etat_moteur(12) == "ok"
    assert etat_moteur(55) == "surveiller"
    assert etat_moteur(87) == "urgent"


def test_etat_moteur_cas_limites():
    assert etat_moteur(40) == "ok"
    assert etat_moteur(80) == "surveiller"


def test_score_z_capteur_bloque_leve_une_erreur():
    with pytest.raises(ValueError):
        score_z([520, 530], [520, 520, 520])