"""Santé du système : client de monitoring (sans réseau) et mise en forme."""

import httpx

from aeroguard.sante_systeme import (
    ClientMonitoring,
    lecture_statut,
    lignes_derive,
    rapport_markdown,
)

RAPPORT = {
    "status": "major",
    "n_flights": 907,
    "n_engines": 21,
    "n_major": 1,
    "features": [
        {"feature": "T48_mean_montee", "psi": 0.6034, "status": "major"},
        {"feature": "P40_mean_montee", "psi": 0.12, "status": "moderate"},
    ],
}


def fausse_api(request):
    if request.url.path == "/health":
        corps = {"statut": "ok", "version": "7.0.0", "inference": "onnx", "modeles_charges": True}
        return httpx.Response(200, json=corps, headers={"X-Response-Time-ms": "1.5"})
    assert request.url.params["limite"] == "300"
    return httpx.Response(200, json=RAPPORT)


def client():
    transport = httpx.MockTransport(fausse_api)
    session = httpx.Client(base_url="http://test", transport=transport)
    return ClientMonitoring("http://test", session)


def test_sante_avec_temps_de_reponse():
    sante = client().sante()
    assert sante["version"] == "7.0.0"
    assert sante["server_ms"] == 1.5
    assert sante["round_trip_ms"] >= 0


def test_derive_transmet_la_limite():
    assert client().derive(limite=300)["n_engines"] == 21


def test_lecture_des_statuts():
    assert lecture_statut("major")[1] == "🔴"
    assert lecture_statut("statut_inconnu") == ("statut_inconnu", "⚪", "")


def test_lignes_derive():
    assert lignes_derive(RAPPORT, n=1) == [
        {"Feature": "T48_mean_montee", "PSI": 0.6034, "Status": "major"}
    ]


def test_rapport_markdown_complet():
    texte = rapport_markdown(RAPPORT, "https://api.example")
    assert "🔴 AeroGuard drift report: Major drift" in texte
    assert "- Engines: 21" in texte
    assert "| T48_mean_montee | 0.603 | major |" in texte


def test_rapport_markdown_sans_features():
    rapport = {"status": "insufficient_engines", "n_engines": 1, "features": []}
    texte = rapport_markdown(rapport, "https://api.example")
    assert "Not enough engines" in texte
    assert "| Feature |" not in texte
