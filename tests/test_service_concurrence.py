"""Concurrence du service : moteurs différents en parallèle, un même moteur dans l'ordre."""

import threading
import time

import numpy as np

from aeroguard.service import ServiceDecision


def service_avec_compteur():
    """Service dont le faux modèle compte combien d'appels se chevauchent."""
    etat = {"en_cours": 0, "max": 0}
    verrou = threading.Lock()

    def predire_proba(x):
        with verrou:
            etat["en_cours"] += 1
            etat["max"] = max(etat["max"], etat["en_cours"])
        time.sleep(0.05)
        with verrou:
            etat["en_cours"] -= 1
        return 0.1

    service = ServiceDecision(
        colonnes=["T24_a", "T30_a"],
        predire_proba=predire_proba,
        erreurs_ae=lambda x: np.zeros(2),
        centrer=lambda X, cycles: X,
        medianes=np.zeros(2),
    )
    return service, etat


def lancer_en_parallele(fonctions):
    threads = [threading.Thread(target=f) for f in fonctions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


MESURES = {"T24_a": 1.0, "T30_a": 2.0}


def test_moteurs_differents_traites_en_parallele():
    service, etat = service_avec_compteur()
    lancer_en_parallele([lambda m=f"M{i}": service.traiter_vol(m, 1, MESURES) for i in range(4)])
    assert etat["max"] >= 2
    assert len(service.etat_flotte()) == 4


def test_un_meme_moteur_reste_sequentiel():
    service, etat = service_avec_compteur()

    def envoyer(cycle):
        try:
            service.traiter_vol("M1", cycle, MESURES)
        except ValueError:
            pass  # cycle arrivé après un cycle plus récent : rejeté, c'est voulu

    lancer_en_parallele([lambda c=c: envoyer(c) for c in range(1, 5)])
    assert etat["max"] == 1


def test_etat_flotte_pendant_les_envois():
    service, _ = service_avec_compteur()
    lancer_en_parallele(
        [lambda m=f"M{i}": service.traiter_vol(m, 1, MESURES) for i in range(8)]
        + [service.etat_flotte for _ in range(8)]
    )
    assert len(service.etat_flotte()) == 8
