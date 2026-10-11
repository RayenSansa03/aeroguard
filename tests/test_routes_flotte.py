"""Tests des routes /flotte avec un faux service (sans modèles)."""

import pytest
from fastapi.testclient import TestClient

from aeroguard.api import app
from aeroguard.routes_flotte import obtenir_service

VOL = {"moteur": "M1", "cycle": 1, "mesures": {"T48_moy": 1.0, "Nf_moy": None}}


class FauxService:
    def __init__(self):
        self.recus = []

    def traiter_vol(self, moteur, cycle, mesures):
        if "inconnue" in mesures:
            raise ValueError("Features inconnues : ['inconnue']")
        self.recus.append((moteur, cycle, mesures))
        return {"moteur": moteur, "cycle": cycle, "niveau": "OK"}

    def derniere(self, moteur):
        return {"moteur": moteur, "niveau": "OK"} if moteur == "M1" else None

    def etat_flotte(self):
        return [{"moteur": "M1", "niveau": "OK"}]

    def reinitialiser(self, moteur):
        return moteur == "M1"


@pytest.fixture
def faux_service():
    service = FauxService()
    app.dependency_overrides[obtenir_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_envoyer_un_vol(client, faux_service):
    reponse = client.post("/flotte/vols", json=VOL)
    assert reponse.status_code == 200
    assert reponse.json()["niveau"] == "OK"
    assert faux_service.recus == [("M1", 1, {"T48_moy": 1.0, "Nf_moy": None})]


def test_vol_invalide_refuse_par_pydantic(client, faux_service):
    reponse = client.post("/flotte/vols", json={**VOL, "cycle": 0})
    assert reponse.status_code == 422
    assert faux_service.recus == []


def test_erreur_metier_renvoie_422(client, faux_service):
    reponse = client.post("/flotte/vols", json={**VOL, "mesures": {"inconnue": 1.0}})
    assert reponse.status_code == 422
    assert "inconnues" in reponse.json()["detail"]


def test_consulter_et_lister_les_moteurs(client, faux_service):
    assert client.get("/flotte/moteurs").json() == [{"moteur": "M1", "niveau": "OK"}]
    assert client.get("/flotte/moteurs/M1").status_code == 200
    assert client.get("/flotte/moteurs/M9").status_code == 404


def test_reinitialiser_un_moteur(client, faux_service):
    assert client.delete("/flotte/moteurs/M1").json()["statut"] == "reinitialise"
    assert client.delete("/flotte/moteurs/M9").status_code == 404


def test_modeles_absents_renvoie_503(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AEROGUARD_MODELES", str(tmp_path))
    obtenir_service.cache_clear()
    reponse = client.post("/flotte/vols", json=VOL)
    assert reponse.status_code == 503
    assert "manquants" in reponse.json()["detail"]
    obtenir_service.cache_clear()


def test_health_ne_depend_pas_des_modeles(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AEROGUARD_MODELES", str(tmp_path))
    obtenir_service.cache_clear()
    assert client.get("/health").status_code == 200
