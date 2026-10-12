"""Monitoring : logs JSON, temps de réponse, décisions journalisées, santé sans chargement."""

import json
import logging
import sys

from fastapi.testclient import TestClient

from aeroguard import api, routes_flotte
from aeroguard.journalisation import FormatJSON


def enregistrement(message="vol traité", exc_info=None):
    return logging.LogRecord("aeroguard.test", logging.INFO, __file__, 1, message, None, exc_info)


def test_une_ligne_json_par_evenement():
    record = enregistrement()
    record.champs = {"engine": "M1", "cycle": 3}
    ligne = json.loads(FormatJSON().format(record))
    assert ligne["level"] == "INFO"
    assert ligne["message"] == "vol traité"
    assert ligne["engine"] == "M1"
    assert ligne["timestamp"].endswith("+00:00")


def test_exception_incluse_dans_le_log():
    try:
        raise RuntimeError("Triton injoignable")
    except RuntimeError:
        record = enregistrement("échec", exc_info=sys.exc_info())
    ligne = json.loads(FormatJSON().format(record))
    assert "Triton injoignable" in ligne["exception"]


def test_requete_mesuree_et_journalisee(caplog):
    with caplog.at_level(logging.INFO, logger="aeroguard.http"):
        reponse = TestClient(api.app).get("/")
    assert float(reponse.headers["X-Response-Time-ms"]) >= 0
    log = next(r for r in caplog.records if r.name == "aeroguard.http")
    assert log.champs["path"] == "/"
    assert log.champs["status"] == 200


def test_health_mesure_mais_pas_journalise(caplog):
    with caplog.at_level(logging.INFO, logger="aeroguard.http"):
        reponse = TestClient(api.app).get("/health")
    assert "X-Response-Time-ms" in reponse.headers
    assert not [r for r in caplog.records if r.name == "aeroguard.http"]


def test_health_indique_le_mode_sans_charger_les_modeles(monkeypatch):
    monkeypatch.delenv("AEROGUARD_TRITON_URL", raising=False)
    monkeypatch.setenv("AEROGUARD_INFERENCE", "onnx")
    routes_flotte.obtenir_service.cache_clear()
    corps = TestClient(api.app).get("/health").json()
    assert corps["inference"] == "onnx"
    assert corps["modeles_charges"] is False


def test_mode_triton_prioritaire(monkeypatch):
    monkeypatch.setenv("AEROGUARD_TRITON_URL", "http://triton:8000")
    assert api.mode_inference() == "triton"


class FauxService:
    def traiter_vol(self, moteur, cycle, mesures):
        return {"moteur": moteur, "cycle": cycle, "niveau": "URGENT", "proba_usure": 0.97}


def test_chaque_decision_est_journalisee(caplog):
    api.app.dependency_overrides[routes_flotte.obtenir_service] = FauxService
    try:
        with caplog.at_level(logging.INFO, logger="aeroguard.decision"):
            vol = {"moteur": "M1", "cycle": 5, "mesures": {}}
            TestClient(api.app).post("/flotte/vols", json=vol)
    finally:
        api.app.dependency_overrides.clear()
    log = next(r for r in caplog.records if r.name == "aeroguard.decision")
    assert log.champs == {
        "engine": "M1",
        "cycle": 5,
        "decision": "URGENT",
        "wear_probability": 0.97,
    }
