"""Tests du rejeu de vols (données factices, faux serveur httpx)."""

import json

import httpx
import numpy as np
import pandas as pd

from aeroguard import rejeu


def fausses_donnees():
    return pd.DataFrame(
        {
            "moteur": ["M2", "M1", "M1", "M2", "M1"],
            "cycle": [1, 3, 1, 2, 2],
            "T48": [10.0, 30.0, 10.0, np.nan, 20.0],
        }
    )


def test_vols_par_moteur_trie_les_cycles(monkeypatch):
    vols = fausses_donnees()
    monkeypatch.setattr(rejeu, "preparer_xy", lambda v, partie: (v[["T48"]], None))
    par_moteur = rejeu.vols_par_moteur(vols)
    cycles, X = par_moteur["M1"]
    assert list(cycles) == [1, 2, 3]
    assert list(X["T48"]) == [10.0, 20.0, 30.0]
    assert sorted(par_moteur) == ["M1", "M2"]


def test_en_mesures_remplace_nan_par_none():
    ligne = pd.Series({"T48": np.nan, "T50": 2.5})
    assert rejeu.en_mesures(ligne) == {"T48": None, "T50": 2.5}


def faux_client(recus):
    def repondre(requete):
        if requete.method == "DELETE":
            recus.append(("DELETE", requete.url.path))
            return httpx.Response(404)
        vol = json.loads(requete.content)
        recus.append(("POST", vol["cycle"]))
        return httpx.Response(200, json={"cycle": vol["cycle"], "niveau": "OK"})

    return httpx.Client(transport=httpx.MockTransport(repondre), base_url="http://api")


def test_envoyer_vols_reinitialise_puis_envoie_dans_l_ordre():
    recus = []
    X = pd.DataFrame({"T48": [1.0, 2.0, 3.0]})
    recos = rejeu.envoyer_vols(faux_client(recus), "M1", np.array([1, 2, 3]), X)
    assert recus == [("DELETE", "/flotte/moteurs/M1"), ("POST", 1), ("POST", 2), ("POST", 3)]
    assert [r["cycle"] for r in recos] == [1, 2, 3]


def test_envoyer_vols_peut_s_arreter_en_cours_de_vie():
    recus = []
    X = pd.DataFrame({"T48": [1.0, 2.0, 3.0]})
    recos = rejeu.envoyer_vols(faux_client(recus), "M1", np.array([1, 2, 3]), X, jusqu_a=2)
    assert len(recos) == 2
