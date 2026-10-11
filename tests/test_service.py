"""Tests du service de décision en ligne (faux modèles, sans XGBoost ni TensorFlow)."""

import numpy as np
import pytest

from aeroguard.service import ServiceDecision, capteurs_en_cause, charger_service

COLONNES = ["T48_moy", "T50_moy", "Nf_moy"]
SAIN = {"T48_moy": 1.0, "T50_moy": 1.0, "Nf_moy": 1.0}
USE = {"T48_moy": 20.0, "T50_moy": 15.0, "Nf_moy": 1.0}


class FauxModeles:
    """Faux modèles : usure si T48 > 10, erreur AE = écart absolu à la ligne de base."""

    def __init__(self):
        self.dernier_x = None

    def predire_proba(self, x):
        self.dernier_x = x
        return 0.99 if x[0, 0] > 10 else 0.10

    @staticmethod
    def erreurs_ae(x_centre):
        return np.abs(x_centre[0])

    @staticmethod
    def centrer(X, cycles):
        return X - X[:2].mean(axis=0)


@pytest.fixture
def modeles():
    return FauxModeles()


@pytest.fixture
def service(modeles):
    return ServiceDecision(
        colonnes=COLONNES,
        predire_proba=modeles.predire_proba,
        erreurs_ae=modeles.erreurs_ae,
        centrer=modeles.centrer,
        medianes=np.array([1.0, 1.0, 1.0]),
        seuil_ae=0.5,
        k_ae=2,
        n_premiers=2,
    )


def test_moteur_sain_reste_ok(service):
    for cycle in range(1, 7):
        reco = service.traiter_vol("M1", cycle, SAIN)
    assert reco["niveau"] == "OK"
    assert reco["vols_recus"] == 6


def test_usure_durable_passe_urgent(service):
    for cycle in range(1, 4):
        service.traiter_vol("M1", cycle, SAIN)
    for cycle in range(4, 10):
        reco = service.traiter_vol("M1", cycle, USE)
    assert reco["niveau"] == "URGENT"
    assert reco["proba_usure"] == pytest.approx(0.99)


def test_autoencodeur_muet_pendant_ligne_de_base(service):
    for cycle in (1, 2):
        reco = service.traiter_vol("M1", cycle, USE)
        assert reco["erreur_reconstruction"] == 0.0


def test_mesure_manquante_nan_pour_xgb_mediane_pour_ae(service, modeles):
    service.traiter_vol("M1", 1, {"T48_moy": 1.0, "T50_moy": 1.0})
    assert np.isnan(modeles.dernier_x[0, 2])
    assert service.depot.lire("M1").vols[0][2] == 1.0


def test_feature_inconnue_refusee(service):
    with pytest.raises(ValueError, match="inconnues"):
        service.traiter_vol("M1", 1, {"capteur_fantome": 3.0})


def test_cycle_non_croissant_refuse(service):
    service.traiter_vol("M1", 5, SAIN)
    with pytest.raises(ValueError, match="supérieur"):
        service.traiter_vol("M1", 5, SAIN)


def test_moteurs_independants_et_flotte_triee(service):
    for cycle in range(1, 10):
        service.traiter_vol("A", cycle, SAIN)
        service.traiter_vol("B", cycle, SAIN if cycle < 4 else USE)
    flotte = service.etat_flotte()
    assert [ligne["moteur"] for ligne in flotte] == ["B", "A"]
    assert service.derniere("A")["niveau"] == "OK"
    assert service.derniere("inconnu") is None


def test_reinitialiser_oublie_le_moteur(service):
    service.traiter_vol("M1", 1, SAIN)
    assert service.reinitialiser("M1") is True
    assert service.reinitialiser("M1") is False
    assert service.traiter_vol("M1", 1, SAIN)["vols_recus"] == 1


def test_capteurs_en_cause_sans_doublon():
    colonnes = ["T48_moy", "T50_moy", "T50_max"]
    assert capteurs_en_cause([0.1, 3.0, 2.0], colonnes) == ("T50", "T48")


def test_charger_service_signale_fichiers_manquants(tmp_path):
    with pytest.raises(FileNotFoundError, match="xgb_binaire_v3.json"):
        charger_service(tmp_path)


def test_historique_garde_chaque_recommandation(service):
    for cycle in range(1, 4):
        service.traiter_vol("M1", cycle, SAIN)
    assert [reco["cycle"] for reco in service.historique("M1")] == [1, 2, 3]
    assert service.historique("inconnu") is None
