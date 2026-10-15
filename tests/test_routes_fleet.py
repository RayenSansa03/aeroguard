"""Tests de l'API publique en anglais (/fleet, /monitoring/drift) avec un faux service."""

import pytest
from fastapi.testclient import TestClient

from aeroguard.api import app
from aeroguard.routes_flotte import erreur_anglaise, message_anglais, obtenir_service

VOL = {"engine": "M1", "cycle": 1, "measurements": {"T48_moy": 1.0, "Nf_moy": None}}

RESULTAT = {
    "moteur": "M1",
    "cycle": 1,
    "vols_recus": 1,
    "niveau": "A_SURVEILLER",
    "etat": "usure connue",
    "proba_usure": 0.9,
    "erreur_reconstruction": 0.1,
    "composant": None,
    "capteurs": [],
    "message": "Usure connue détectée (probabilité 90%).",
}


class FauxService:
    def __init__(self):
        self.recus = []

    def traiter_vol(self, moteur, cycle, mesures):
        if "inconnue" in mesures:
            raise ValueError("Features inconnues : ['inconnue']")
        self.recus.append((moteur, cycle, mesures))
        return {**RESULTAT, "moteur": moteur, "cycle": cycle}

    def derniere(self, moteur):
        return RESULTAT if moteur == "M1" else None

    def historique(self, moteur):
        return [RESULTAT] if moteur == "M1" else None

    def etat_flotte(self):
        return [{"moteur": "M1", "vols_recus": 1, "niveau": "URGENT", "cycle": 1}]

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


def test_envoyer_un_vol_en_anglais(client, faux_service):
    reponse = client.post("/fleet/flights", json=VOL)
    assert reponse.status_code == 200
    donnees = reponse.json()
    assert donnees["engine"] == "M1"
    assert donnees["level"] == "WATCH"
    assert donnees["state"] == "known wear"
    assert donnees["wear_probability"] == 0.9
    assert donnees["message"] == "Known wear detected (probability 90%)."
    assert "niveau" not in donnees
    assert faux_service.recus == [("M1", 1, {"T48_moy": 1.0, "Nf_moy": None})]


def test_vol_invalide_refuse(client, faux_service):
    assert client.post("/fleet/flights", json={**VOL, "cycle": 0}).status_code == 422
    assert faux_service.recus == []


def test_erreur_metier_en_anglais(client, faux_service):
    reponse = client.post("/fleet/flights", json={**VOL, "measurements": {"inconnue": 1.0}})
    assert reponse.status_code == 422
    assert reponse.json()["detail"] == "Unknown features: ['inconnue']"


def test_lister_consulter_historique(client, faux_service):
    assert client.get("/fleet/engines").json() == [
        {"engine": "M1", "level": "URGENT", "cycle": 1, "flights_received": 1}
    ]
    assert client.get("/fleet/engines/M1").json()["level"] == "WATCH"
    assert len(client.get("/fleet/engines/M1/history").json()) == 1
    inconnu = client.get("/fleet/engines/M9")
    assert inconnu.status_code == 404
    assert inconnu.json()["detail"] == "Unknown engine: M9"
    assert client.get("/fleet/engines/M9/history").status_code == 404


def test_reinitialiser_en_anglais(client, faux_service):
    assert client.delete("/fleet/engines/M1").json() == {"engine": "M1", "status": "reset"}
    assert client.delete("/fleet/engines/M9").status_code == 404


def test_ancienne_api_toujours_disponible(client, faux_service):
    vol = {"moteur": "M1", "cycle": 2, "mesures": {}}
    assert client.post("/flotte/vols", json=vol).status_code == 200


def test_swagger_ne_montre_que_l_api_anglaise(client):
    chemins = client.get("/openapi.json").json()["paths"]
    assert "/fleet/flights" in chemins
    assert not any(chemin.startswith("/flotte") for chemin in chemins)


def test_messages_anglais_par_etat():
    assert message_anglais({"etat": "normal"}) == "No confirmed alert: normal operation."
    assert message_anglais({"etat": "anomalie inconnue", "capteurs": ["T48", "T50"]}) == (
        "Unknown anomaly confirmed: inspect the engine (sensors: T48, T50)."
    )
    assert message_anglais({"etat": "usure confirmée", "composant": "HPT"}) == (
        "Wear confirmed by both models: priority maintenance, likely component: HPT."
    )


def test_erreur_de_cycle_traduite():
    texte = "Le cycle 2 doit être supérieur au dernier cycle reçu (3) pour le moteur M1"
    assert erreur_anglaise(texte) == (
        "Cycle 2 must be greater than the last cycle received (3) for engine M1"
    )
    assert erreur_anglaise("autre erreur") == "autre erreur"


def test_swagger_monitoring_en_anglais(client):
    chemins = client.get("/openapi.json").json()["paths"]
    assert "/monitoring/drift" in chemins
    assert "/monitoring/derive" not in chemins
    parametres = [p["name"] for p in chemins["/monitoring/drift"]["get"]["parameters"]]
    assert "limit" in parametres
