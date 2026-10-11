"""Tests du simulateur de flotte (faux moteurs, fausse API)."""

import numpy as np
import pandas as pd

from aeroguard.simulateur import MoteurSimule, Simulateur


def moteur(nom, vie=6):
    """Moteur factice : la mesure « usure » vaut la RUL, l'API dira URGENT si RUL <= 3."""
    rul = np.arange(vie - 1, -1, -1)
    return MoteurSimule(nom, np.arange(1, vie + 1), pd.DataFrame({"usure": rul}), rul)


def fausse_api(seuil=3):
    envois = []

    def envoyer(nom, cycle, mesures):
        envois.append((nom, cycle))
        return {"niveau": "URGENT" if mesures["usure"] <= seuil else "OK"}

    return envoyer, envois


def simuler(simulateur):
    evenements = []
    while not simulateur.termine():
        evenements += simulateur.tour()
    return evenements


def test_alerte_puis_maintenance_apres_le_delai():
    envoyer, _ = fausse_api()
    m = moteur("M1")
    evenements = simuler(Simulateur([m], envoyer, delai_atelier=1))
    assert m.rul_alerte == 3
    assert m.issue == "maintenance"
    assert m.rul_sortie == 2
    assert [e["evenement"] for e in evenements] == ["alerte", "maintenance"]


def test_delai_nul_maintenance_immediate():
    envoyer, _ = fausse_api()
    m = moteur("M1")
    simuler(Simulateur([m], envoyer, delai_atelier=0))
    assert m.issue == "maintenance"
    assert m.rul_sortie == 3


def test_sans_alerte_le_moteur_tombe_en_panne():
    envoyer, _ = fausse_api(seuil=-1)
    m = moteur("M1")
    simuler(Simulateur([m], envoyer))
    assert m.rul_alerte is None
    assert m.issue == "panne"
    assert m.rul_sortie == 0


def test_delai_trop_long_donne_une_panne():
    envoyer, _ = fausse_api()
    m = moteur("M1")
    simuler(Simulateur([m], envoyer, delai_atelier=10))
    assert m.rul_alerte == 3
    assert m.issue == "panne"


def test_nombre_de_moteurs_en_vol_limite_et_remplacement():
    envoyer, envois = fausse_api()
    reinitialises = []
    moteurs = [moteur("A"), moteur("B"), moteur("C")]
    simulateur = Simulateur(moteurs, envoyer, reinitialises.append, delai_atelier=0, en_vol_max=2)
    simulateur.tour()
    assert {nom for nom, _ in envois} == {"A", "B"}
    simuler(simulateur)
    assert {nom for nom, _ in envois} == {"A", "B", "C"}
    assert reinitialises.count("C") == 2


def test_bilan_operationnel():
    envoyer, _ = fausse_api()
    moteurs = [moteur("A"), moteur("B", vie=2)]
    simulateur = Simulateur(moteurs, envoyer, delai_atelier=1)
    simuler(simulateur)
    bilan = simulateur.bilan()
    assert bilan["moteurs"] == 2
    assert bilan["pannes_evitees"] == 1
    assert bilan["pannes_en_service"] == 1
    assert bilan["avance_mediane"] == 2.0
    assert bilan["vols_restants_moyens"] == 2.0


def test_alerte_au_dernier_vol_reste_une_panne():
    envoyer, _ = fausse_api(seuil=0)
    m = moteur("M1")
    simuler(Simulateur([m], envoyer, delai_atelier=0))
    assert m.rul_alerte == 0
    assert m.issue == "panne"


def test_bilan_vide():
    simulateur = Simulateur([], fausse_api()[0])
    assert simulateur.termine()
    assert simulateur.bilan()["avance_mediane"] is None
