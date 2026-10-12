"""La version annoncée par /health vient du build (tag Git), sinon du package."""

from importlib.metadata import version

from fastapi.testclient import TestClient

from aeroguard import api


def test_version_injectee_au_build(monkeypatch):
    monkeypatch.setenv("AEROGUARD_VERSION", "7.0.0-rc.2")
    reponse = TestClient(api.app).get("/health")
    assert reponse.json()["version"] == "7.0.0-rc.2"


def test_version_du_package_hors_docker(monkeypatch):
    monkeypatch.delenv("AEROGUARD_VERSION", raising=False)
    assert api.version_aeroguard() == version("aeroguard")
