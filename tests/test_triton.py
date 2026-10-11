"""Tests de la préparation du dépôt Triton (sans Triton ni ONNX)."""

import json

import pytest

from aeroguard.triton import (
    adapter_xgboost_pour_fil,
    config_fil,
    config_onnx,
    ecrire_modele_fil,
    ecrire_modele_onnx,
)


def modele_xgboost(cats=None, base_score="5E-1", arbres=1, best_iteration=None):
    """Squelette du JSON XGBoost, avec ou sans les nouveautés de XGBoost 3."""
    interne = {
        "trees": [{"id": i} for i in range(arbres)],
        "tree_info": [0] * arbres,
        "iteration_indptr": list(range(arbres + 1)),
        "gbtree_model_param": {"num_trees": str(arbres)},
    }
    if cats is not None:
        interne["cats"] = cats
    learner = {
        "gradient_booster": {"model": interne, "name": "gbtree"},
        "learner_model_param": {"base_score": base_score, "num_feature": "3"},
    }
    if best_iteration is not None:
        learner["attributes"] = {"best_iteration": str(best_iteration)}
    return {"learner": learner}


def ecrire_json(chemin, contenu):
    chemin.write_text(json.dumps(contenu), encoding="utf-8")
    return chemin


def test_config_fil_decrit_un_xgboost_sur_cpu():
    config = config_fil("xgb_alerte", 42)
    assert 'backend: "fil"' in config
    assert "dims: [ 42 ]" in config
    assert '"xgboost_json"' in config
    assert "KIND_CPU" in config
    assert '"threshold"' in config


def test_config_onnx_entree_et_sortie_de_meme_taille():
    config = config_onnx("autoencodeur", 42)
    assert 'backend: "onnxruntime"' in config
    assert config.count("dims: [ 42 ]") == 2
    assert "dynamic_batching" in config


def test_adapter_retire_cats_vide_et_garde_les_arbres(tmp_path):
    vide = {"enc": [], "feature_segments": [], "sorted_idx": []}
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(cats=vide))
    adapter_xgboost_pour_fil(source, tmp_path / "fil.json")
    resultat = json.loads((tmp_path / "fil.json").read_text(encoding="utf-8"))
    interne = resultat["learner"]["gradient_booster"]["model"]
    assert "cats" not in interne
    assert interne["trees"] == [{"id": 0}]


def test_adapter_accepte_un_modele_xgboost_2(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost())
    adapter_xgboost_pour_fil(source, tmp_path / "fil.json")
    resultat = json.loads((tmp_path / "fil.json").read_text(encoding="utf-8"))
    assert resultat == modele_xgboost()


def test_adapter_refuse_les_variables_categorielles(tmp_path):
    remplie = {"enc": [[0, 1]], "feature_segments": [0], "sorted_idx": [0]}
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(cats=remplie))
    with pytest.raises(ValueError, match="catégorielles"):
        adapter_xgboost_pour_fil(source, tmp_path / "fil.json")


def test_adapter_transforme_base_score_liste_en_nombre(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(base_score="[7.0505923E-1]"))
    adapter_xgboost_pour_fil(source, tmp_path / "fil.json")
    resultat = json.loads((tmp_path / "fil.json").read_text(encoding="utf-8"))
    assert resultat["learner"]["learner_model_param"]["base_score"] == "7.0505923E-1"


def test_adapter_refuse_base_score_multiple(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(base_score="[1E-1,2E-1]"))
    with pytest.raises(ValueError, match="plusieurs valeurs"):
        adapter_xgboost_pour_fil(source, tmp_path / "fil.json")


def test_adapter_coupe_les_arbres_apres_best_iteration(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(arbres=10, best_iteration=6))
    assert adapter_xgboost_pour_fil(source, tmp_path / "fil.json") == 3
    resultat = json.loads((tmp_path / "fil.json").read_text(encoding="utf-8"))
    interne = resultat["learner"]["gradient_booster"]["model"]
    assert [arbre["id"] for arbre in interne["trees"]] == list(range(7))
    assert len(interne["tree_info"]) == 7
    assert interne["iteration_indptr"] == list(range(8))
    assert interne["gbtree_model_param"]["num_trees"] == "7"


def test_adapter_sans_arret_precoce_garde_tous_les_arbres(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(arbres=5))
    assert adapter_xgboost_pour_fil(source, tmp_path / "fil.json") == 0


def test_ecrire_modele_fil_respecte_la_structure_triton(tmp_path):
    source = ecrire_json(tmp_path / "xgb.json", modele_xgboost(cats={"enc": []}))
    dossier = ecrire_modele_fil(tmp_path / "depot", "xgb_alerte", source, 3)
    assert (dossier / "1" / "xgboost.json").exists()
    assert 'name: "xgb_alerte"' in (dossier / "config.pbtxt").read_text(encoding="utf-8")


def test_ecrire_modele_onnx_respecte_la_structure_triton(tmp_path):
    class FauxOnnx:
        def SerializeToString(self):  # noqa: N802 (nom imposé par protobuf)
            return b"onnx"

    dossier = ecrire_modele_onnx(tmp_path / "depot", "autoencodeur", FauxOnnx(), 3)
    assert (dossier / "1" / "model.onnx").read_bytes() == b"onnx"
    assert 'name: "autoencodeur"' in (dossier / "config.pbtxt").read_text(encoding="utf-8")
