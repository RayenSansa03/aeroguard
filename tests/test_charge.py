"""Tests du test de charge (faux service qui dort, sans réseau)."""

import threading
import time

import pytest

from aeroguard.charge import executer_charge, resume_latences


def test_resume_latences():
    resume = resume_latences([10.0, 20.0, 30.0, 40.0])
    assert resume["n"] == 4
    assert resume["p50"] == pytest.approx(25.0)
    assert resume["moyenne"] == pytest.approx(25.0)
    assert resume["p95"] == pytest.approx(38.5)


def test_resume_latences_vide():
    assert resume_latences([])["p50"] is None


def test_chaque_moteur_garde_l_ordre_de_ses_vols():
    recus = {}
    verrou = threading.Lock()

    def envoyer(vol):
        moteur, cycle = vol
        with verrou:
            recus.setdefault(moteur, []).append(cycle)

    sequences = [[(m, c) for c in range(1, 6)] for m in ("A", "B", "C")]
    resultat = executer_charge(sequences, envoyer, concurrence=3)
    assert resultat["n"] == 15
    assert resultat["erreurs"] == 0
    assert all(cycles == [1, 2, 3, 4, 5] for cycles in recus.values())


def test_les_echecs_sont_comptes_a_part():
    def envoyer(vol):
        if vol % 2:
            raise RuntimeError("503")

    resultat = executer_charge([[1, 2, 3, 4]], envoyer, concurrence=1)
    assert resultat["n"] == 2
    assert resultat["erreurs"] == 2


def test_la_concurrence_augmente_le_debit_quand_le_service_attend():
    sequences = [[0] * 5 for _ in range(4)]

    def envoyer(vol):
        time.sleep(0.02)

    seul = executer_charge(sequences, envoyer, concurrence=1)
    a_quatre = executer_charge(sequences, envoyer, concurrence=4)
    assert a_quatre["debit"] > 2 * seul["debit"]
