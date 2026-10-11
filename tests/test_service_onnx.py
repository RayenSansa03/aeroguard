"""Mode ONNX : vrais petits modèles fabriqués dans le test, sans TensorFlow ni Triton."""

import json

import numpy as np
import onnx
import pytest
import xgboost
from onnx import TensorProto, helper

from aeroguard import routes_flotte, service
from aeroguard.service import FICHIER_ONNX, FICHIERS_MODELES, charger_service_onnx

COLONNES = ["T24_moy", "T30_moy", "T48_moy", "Wf_moy"]


def ecrire_modeles(dossier, n_premiers=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(COLONNES)))
    y = (X[:, 0] > 0).astype(int)
    modele = xgboost.XGBClassifier(n_estimators=5, max_depth=2)
    modele.fit(X, y)
    modele.save_model(dossier / FICHIERS_MODELES["alerte"])

    # Autoencodeur factice : reconstruction = 0.5 * entrée.
    poids = helper.make_tensor(
        "poids", TensorProto.FLOAT, [4, 4], (0.5 * np.eye(4)).flatten().tolist()
    )
    graphe = helper.make_graph(
        [helper.make_node("MatMul", ["entree", "poids"], ["sortie"])],
        "autoencodeur_test",
        [helper.make_tensor_value_info("entree", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("sortie", TensorProto.FLOAT, [None, 4])],
        initializer=[poids],
    )
    onnx.save(
        helper.make_model(graphe, opset_imports=[helper.make_opsetid("", 17)], ir_version=8),
        dossier / FICHIER_ONNX,
    )

    np.savez(
        dossier / FICHIERS_MODELES["stats"],
        colonnes=np.array(COLONNES),
        moyennes=np.zeros(4),
        ecarts=np.ones(4),
        medianes_imputation=np.zeros(4),
    )
    config = {"seuil": 0.618, "k": 5, "n_premiers": n_premiers}
    (dossier / FICHIERS_MODELES["config"]).write_text(json.dumps(config), encoding="utf-8")
    return modele


def test_fichier_manquant(tmp_path):
    with pytest.raises(FileNotFoundError, match="manquants"):
        charger_service_onnx(tmp_path)


def test_proba_identique_a_xgbclassifier(tmp_path):
    modele = ecrire_modeles(tmp_path)
    service = charger_service_onnx(tmp_path)
    mesures = dict(zip(COLONNES, [0.7, -0.2, 0.1, 0.3], strict=True))
    resultat = service.traiter_vol("M1", 1, mesures)
    attendu = modele.predict_proba(np.array([[0.7, -0.2, 0.1, 0.3]]))[0, 1]
    assert resultat["proba_usure"] == pytest.approx(attendu, abs=1e-6)


def test_erreur_autoencodeur_apres_la_ligne_de_base(tmp_path):
    ecrire_modeles(tmp_path, n_premiers=3)
    service = charger_service_onnx(tmp_path)
    for cycle in range(1, 6):
        mesures = dict(zip(COLONNES, [0.1 * cycle] * 4, strict=True))
        service.traiter_vol("M1", cycle, mesures)
    historique = service.depot.lire("M1")
    assert historique.erreurs[:3] == [0.0, 0.0, 0.0]
    assert historique.erreurs[-1] > 0


def test_mode_onnx_choisi_par_variable(monkeypatch):
    monkeypatch.delenv("AEROGUARD_TRITON_URL", raising=False)
    monkeypatch.setenv("AEROGUARD_INFERENCE", "onnx")
    monkeypatch.setattr(service, "charger_service_onnx", lambda dossier: "service-onnx")
    routes_flotte.obtenir_service.cache_clear()
    try:
        assert routes_flotte.obtenir_service() == "service-onnx"
    finally:
        routes_flotte.obtenir_service.cache_clear()
