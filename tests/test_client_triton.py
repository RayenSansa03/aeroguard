"""Tests du client Triton avec un faux serveur (httpx.MockTransport)."""

import json

import httpx
import numpy as np
import pytest

from aeroguard.client_triton import ClientTriton


def faux_triton(requete):
    """Imite Triton : décode l'en-tête JSON et les octets, renvoie une sortie JSON."""
    if requete.url.path.endswith("/ready"):
        return httpx.Response(200 if "inconnu" not in requete.url.path else 404)
    taille = int(requete.headers["Inference-Header-Content-Length"])
    contenu = requete.content
    entete = json.loads(contenu[:taille])
    entree = entete["inputs"][0]
    X = np.frombuffer(contenu[taille:], dtype=np.float32).reshape(entree["shape"])
    if "xgb" in requete.url.path:
        proba = np.where(np.isnan(X[:, 0]), 0.5, 0.9)
        sortie = np.stack([1 - proba, proba], axis=1)
    else:
        sortie = X * 2
    nom = entete["outputs"][0]["name"]
    donnees = sortie.ravel().tolist()
    corps = {"outputs": [{"name": nom, "shape": list(sortie.shape), "data": donnees}]}
    return httpx.Response(200, json=corps)


@pytest.fixture
def client():
    transport = httpx.MockTransport(faux_triton)
    return ClientTriton(client=httpx.Client(transport=transport, base_url="http://triton"))


def test_probas_usure_prend_la_colonne_classe_1(client):
    X = np.ones((3, 4))
    np.testing.assert_allclose(client.probas_usure(X), [0.9, 0.9, 0.9], rtol=1e-6)


def test_les_nan_traversent_le_reseau_intacts(client):
    X = np.array([[np.nan, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(client.probas_usure(X), [0.5, 0.9], rtol=1e-6)


def test_reconstruction_garde_la_forme(client):
    Xn = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_allclose(client.reconstruction(Xn), Xn * 2)


def test_un_seul_vol_est_accepte(client):
    assert client.reconstruction(np.ones(3)).shape == (1, 3)


def test_pret_serveur_et_modele(client):
    assert client.pret() is True
    assert client.pret("xgb_alerte") is True
    assert client.pret("inconnu") is False


def test_serveur_injoignable_pas_pret():
    def refuse(requete):
        raise httpx.ConnectError("refusé", request=requete)

    transport = httpx.MockTransport(refuse)
    client = ClientTriton(client=httpx.Client(transport=transport, base_url="http://triton"))
    assert client.pret() is False


def test_erreur_triton_remonte():
    transport = httpx.MockTransport(lambda requete: httpx.Response(400, json={"error": "x"}))
    client = ClientTriton(client=httpx.Client(transport=transport, base_url="http://triton"))
    with pytest.raises(httpx.HTTPStatusError):
        client.probas_usure(np.ones((1, 2)))


def test_client_par_defaut_vise_le_port_8000():
    assert str(ClientTriton().client.base_url) == "http://127.0.0.1:8000"
