"""Dépôt d'historique : contrat de stockage et écriture « tout ou rien » du service."""

import numpy as np
import pytest

from aeroguard.depot import DepotMemoire, HistoriqueMoteur
from aeroguard.service import ServiceDecision


@pytest.fixture
def depot():
    return DepotMemoire()


def historique_exemple(cycle=1):
    derniere = {"moteur": "M1", "cycle": cycle, "vols_recus": 1, "niveau": "OK"}
    return HistoriqueMoteur(
        cycles=[cycle],
        vols=[np.array([1.0, 2.0])],
        probas=[0.1],
        erreurs=[0.0],
        recommandations=[derniere],
        derniere=derniere,
    )


def test_lire_un_moteur_inconnu(depot):
    assert depot.lire("M1") is None


def test_ecrire_puis_lire(depot):
    depot.ecrire("M1", historique_exemple())
    lu = depot.lire("M1")
    assert lu.cycles == [1]
    np.testing.assert_array_equal(lu.vols[0], [1.0, 2.0])
    assert lu.derniere["niveau"] == "OK"


def test_supprimer(depot):
    depot.ecrire("M1", historique_exemple())
    assert depot.supprimer("M1") is True
    assert depot.supprimer("M1") is False
    assert depot.lire("M1") is None


def test_dernieres(depot):
    depot.ecrire("M1", historique_exemple())
    depot.ecrire("M2", historique_exemple(cycle=5))
    assert {m: d["cycle"] for m, d in depot.dernieres().items()} == {"M1": 1, "M2": 5}


def test_un_verrou_par_moteur(depot):
    assert depot.verrou("M1") is depot.verrou("M1")
    assert depot.verrou("M1") is not depot.verrou("M2")


def test_echec_du_modele_ne_laisse_pas_d_historique_partiel():
    def autoencodeur_en_panne(x):
        raise RuntimeError("modèle indisponible")

    service = ServiceDecision(
        colonnes=["T24_a", "T30_a"],
        predire_proba=lambda x: 0.1,
        erreurs_ae=autoencodeur_en_panne,
        centrer=lambda X, cycles: X,
        medianes=np.zeros(2),
        n_premiers=1,
    )
    mesures = {"T24_a": 1.0, "T30_a": 2.0}
    service.traiter_vol("M1", 1, mesures)  # ligne de base : l'autoencodeur n'est pas appelé
    with pytest.raises(RuntimeError):
        service.traiter_vol("M1", 2, mesures)

    assert service.depot.lire("M1").cycles == [1]
    assert [r["cycle"] for r in service.historique("M1")] == [1]
