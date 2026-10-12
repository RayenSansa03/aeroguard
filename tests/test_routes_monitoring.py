"""Route /monitoring/derive et fenêtre des vols récents."""

import json

import numpy as np
from fastapi.testclient import TestClient

from aeroguard import api, routes_flotte, routes_monitoring
from aeroguard.derive import FICHIER_REFERENCE, construire_reference
from aeroguard.service import ServiceDecision

COLONNES = ["T24_moy", "T48_moy"]
REFERENCE = construire_reference(np.random.default_rng(0).normal(size=(5000, 2)), COLONNES)


class FauxService:
    def __init__(self, vols, n_moteurs=10):
        self.vols = vols
        self.n_moteurs = n_moteurs

    def vols_recents(self, limite):
        return self.vols[-limite:]

    def etat_flotte(self):
        return [{"moteur": f"M{i}"} for i in range(self.n_moteurs)]


def appeler(vols, url="/monitoring/derive"):
    api.app.dependency_overrides[routes_flotte.obtenir_service] = lambda: FauxService(vols)
    api.app.dependency_overrides[routes_monitoring.obtenir_reference] = lambda: REFERENCE
    try:
        return TestClient(api.app).get(url)
    finally:
        api.app.dependency_overrides.clear()


def test_flotte_stable():
    vols = np.random.default_rng(1).normal(size=(1000, 2))
    assert appeler(vols).json()["status"] == "stable"


def test_capteur_biaise_detecte():
    vols = np.random.default_rng(2).normal(size=(1000, 2))
    vols[:, 1] += 1.0
    rapport = appeler(vols).json()
    assert rapport["status"] == "major"
    assert rapport["features"][0]["feature"] == "T48_moy"


def test_limite_trop_petite_refusee():
    assert appeler(np.zeros((100, 2)), "/monitoring/derive?limite=10").status_code == 422


def test_reference_absente_503(monkeypatch, tmp_path):
    monkeypatch.setenv("AEROGUARD_MODELES", str(tmp_path))
    routes_monitoring.obtenir_reference.cache_clear()
    api.app.dependency_overrides[routes_flotte.obtenir_service] = lambda: FauxService(None)
    try:
        reponse = TestClient(api.app).get("/monitoring/derive")
    finally:
        api.app.dependency_overrides.clear()
        routes_monitoring.obtenir_reference.cache_clear()
    assert reponse.status_code == 503


def test_reference_lue_depuis_le_dossier(monkeypatch, tmp_path):
    (tmp_path / FICHIER_REFERENCE).write_text(json.dumps(REFERENCE), encoding="utf-8")
    monkeypatch.setenv("AEROGUARD_MODELES", str(tmp_path))
    routes_monitoring.obtenir_reference.cache_clear()
    try:
        assert routes_monitoring.obtenir_reference()["colonnes"] == COLONNES
    finally:
        routes_monitoring.obtenir_reference.cache_clear()


def service_reel():
    return ServiceDecision(
        colonnes=COLONNES,
        predire_proba=lambda x: 0.1,
        erreurs_ae=lambda x: np.zeros(2),
        centrer=lambda X, cycles: X,
        medianes=np.zeros(2),
    )


def test_vols_recents_tous_moteurs():
    service = service_reel()
    assert service.vols_recents().shape == (0, 2)
    service.traiter_vol("M1", 1, {"T24_moy": 1.0, "T48_moy": 2.0})
    service.traiter_vol("M1", 2, {"T24_moy": 3.0, "T48_moy": None})
    service.traiter_vol("M2", 1, {"T24_moy": 5.0, "T48_moy": 6.0})
    vols = service.vols_recents()
    assert vols.shape == (3, 2)
    assert vols[1, 1] == 0.0  # mesure manquante imputée à la médiane
    assert service.vols_recents(limite=2).shape == (2, 2)


def test_un_seul_moteur_pas_de_fausse_alarme():
    vols = np.random.default_rng(4).normal(size=(500, 2)) + 3.0
    api.app.dependency_overrides[routes_flotte.obtenir_service] = lambda: FauxService(vols, 1)
    api.app.dependency_overrides[routes_monitoring.obtenir_reference] = lambda: REFERENCE
    try:
        rapport = TestClient(api.app).get("/monitoring/derive").json()
    finally:
        api.app.dependency_overrides.clear()
    assert rapport["status"] == "insufficient_engines"
    assert rapport["n_engines"] == 1
