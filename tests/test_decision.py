import json

import pytest

from aeroguard.decision import NIVEAUX, decider
from aeroguard.hybride import ETATS

SAIN_XGB = [0.1, 0.2, 0.1, 0.2, 0.1]
SAIN_AE = [0.2, 0.3, 0.2, 0.3, 0.2]
USE_XGB = [0.5, 0.9, 0.85, 0.9, 0.9]
FORT_AE = [0.7, 0.8, 0.9, 0.8, 0.9]


def test_moteur_normal():
    reco = decider(SAIN_XGB, SAIN_AE)
    assert (reco.niveau, reco.etat) == ("OK", ETATS[0])
    assert reco.composant is None and reco.capteurs == ()


def test_usure_connue_a_surveiller_avec_composant():
    reco = decider(USE_XGB, SAIN_AE, composant="HPT", capteurs=("T48",))
    assert (reco.niveau, reco.etat) == ("A_SURVEILLER", ETATS[2])
    assert reco.composant == "HPT"
    assert reco.capteurs == ()  # pas d'anomalie : pas de capteurs affichés
    assert "HPT" in reco.message


def test_usure_connue_tres_probable_est_urgente():
    reco = decider([0.9, 0.95, 0.97], [0.1, 0.1, 0.1])
    assert reco.niveau == "URGENT"


def test_anomalie_inconnue_a_surveiller_avec_capteurs():
    reco = decider(SAIN_XGB, FORT_AE, composant="HPT", capteurs=("P21", "P15"))
    assert (reco.niveau, reco.etat) == ("A_SURVEILLER", ETATS[1])
    assert reco.capteurs == ("P21", "P15")
    assert reco.composant is None  # XGBoost n'a rien confirmé
    assert "P21" in reco.message


def test_anomalie_inconnue_tres_forte_est_urgente():
    reco = decider(SAIN_XGB, [2.0, 2.5, 3.0, 2.8, 3.1])
    assert reco.niveau == "URGENT"


def test_deux_modeles_d_accord_urgent():
    reco = decider(USE_XGB, FORT_AE, composant="fan")
    assert (reco.niveau, reco.etat) == ("URGENT", ETATS[3])
    assert "fan" in reco.message


def test_confirmation_sur_k_vols():
    # seulement 2 des 3 derniers vols au-dessus du seuil XGBoost : pas confirmé
    assert decider([0.9, 0.5, 0.9, 0.9], [0.1] * 4).niveau == "OK"
    # historique trop court pour confirmer
    assert decider([0.9, 0.9], [0.9, 0.9]).niveau == "OK"


def test_seuils_aux_limites():
    # XGBoost : >= seuil compte ; autoencodeur : il faut strictement > seuil
    assert decider([0.81] * 3, [0.1] * 3).etat == ETATS[2]
    assert decider([0.1] * 5, [0.618] * 5).etat == ETATS[0]


def test_en_dict_serialisable_en_json():
    donnees = decider(SAIN_XGB, FORT_AE, capteurs=("P21",)).en_dict()
    assert donnees["capteurs"] == ["P21"]
    assert donnees["niveau"] in NIVEAUX
    json.dumps(donnees)


def test_entrees_invalides():
    with pytest.raises(ValueError):
        decider([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        decider([], [])
    with pytest.raises(ValueError):
        decider([1.2], [0.1])
    with pytest.raises(ValueError):
        decider([0.1], [0.1], k_xgb=0)
