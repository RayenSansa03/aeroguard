"""Tests des outils du tableau de bord (faux serveur httpx, sans Streamlit)."""

import httpx
import pytest

from aeroguard.tableau import (
    LIBELLES,
    MESSAGES,
    ClientFlotte,
    compter_niveaux,
    tableau_flotte,
    tableau_historique,
)

FLOTTE = [
    {"moteur": "B", "niveau": "URGENT", "cycle": 60, "vols_recus": 60},
    {"moteur": "A", "niveau": "OK", "cycle": 12, "vols_recus": 12},
]
HISTORIQUE = [
    {"cycle": 2, "proba_usure": 0.2, "erreur_reconstruction": 0.0, "niveau": "OK", "etat": "x"},
    {"cycle": 1, "proba_usure": 0.1, "erreur_reconstruction": 0.0, "niveau": "OK", "etat": "x"},
]


def faux_api(requete):
    chemin = requete.url.path
    if chemin == "/health":
        return httpx.Response(200, json={"statut": "ok"})
    if chemin == "/flotte/moteurs":
        return httpx.Response(200, json=FLOTTE)
    if chemin == "/flotte/moteurs/A/historique":
        return httpx.Response(200, json=HISTORIQUE)
    if chemin == "/flotte/moteurs/PANNE/historique":
        return httpx.Response(500)
    return httpx.Response(404, json={"detail": "Moteur inconnu"})


@pytest.fixture
def client():
    transport = httpx.MockTransport(faux_api)
    return ClientFlotte(client=httpx.Client(transport=transport, base_url="http://api"))


def test_lecture_de_la_flotte_et_de_l_historique(client):
    assert client.disponible() is True
    assert client.flotte() == FLOTTE
    assert len(client.historique("A")) == 2


def test_moteur_inconnu_donne_un_historique_vide(client):
    assert client.historique("Z") == []


def test_erreur_serveur_remonte(client):
    with pytest.raises(httpx.HTTPStatusError):
        client.historique("PANNE")


def test_api_injoignable():
    def refuse(requete):
        raise httpx.ConnectError("refusé", request=requete)

    transport = httpx.MockTransport(refuse)
    client = ClientFlotte(client=httpx.Client(transport=transport, base_url="http://api"))
    assert client.disponible() is False


def test_client_par_defaut_vise_le_port_8000():
    assert str(ClientFlotte().client.base_url) == "http://127.0.0.1:8000"


def test_compter_niveaux_garde_les_trois_niveaux():
    assert compter_niveaux(FLOTTE) == {"URGENT": 1, "A_SURVEILLER": 0, "OK": 1}
    assert compter_niveaux([]) == {"URGENT": 0, "A_SURVEILLER": 0, "OK": 0}


def test_tableau_flotte_en_anglais():
    tableau = tableau_flotte(FLOTTE)
    assert list(tableau["Status"]) == ["Urgent", "OK"]
    colonnes = ["Engine", "Status", "Last cycle", "Flights received", "niveau"]
    assert list(tableau.columns) == colonnes


def test_tableau_historique_trie_par_cycle_et_garde_l_essentiel():
    tableau = tableau_historique(HISTORIQUE)
    assert list(tableau["cycle"]) == [1, 2]
    colonnes = ["cycle", "proba_usure", "erreur_reconstruction", "niveau", "status"]
    assert list(tableau.columns) == colonnes
    assert list(tableau["status"]) == ["OK", "OK"]


def test_chaque_niveau_a_un_libelle_et_un_message():
    assert set(LIBELLES) == set(MESSAGES) == {"OK", "A_SURVEILLER", "URGENT"}
    assert LIBELLES["A_SURVEILLER"] == "Watch"
