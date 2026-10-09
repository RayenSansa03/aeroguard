from importlib.metadata import PackageNotFoundError

from fastapi.testclient import TestClient

from aeroguard import api

client = TestClient(api.app)


def test_accueil_indique_la_documentation():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert reponse.json()["documentation"] == "/docs"


def test_health_repond_ok():
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["statut"] == "ok"


def test_health_lit_l_adresse_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert client.get("/health").json()["mlflow"] == "http://mlflow:5000"


def test_health_sans_mlflow(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert client.get("/health").json()["mlflow"] == "non configuré"


def test_version_inconnue_si_package_absent(monkeypatch):
    def absent(_nom):
        raise PackageNotFoundError

    monkeypatch.setattr(api, "version", absent)
    assert api.version_aeroguard() == "inconnue"


def test_route_inexistante():
    assert client.get("/nexiste-pas").status_code == 404
