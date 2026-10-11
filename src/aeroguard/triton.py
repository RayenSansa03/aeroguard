"""Dépôt de modèles Triton : XGBoost via le backend FIL, autoencodeur via ONNX Runtime."""

import json
from pathlib import Path


def config_fil(nom, nombre_features, max_batch=1024):
    """Configuration Triton d'un classifieur XGBoost binaire servi par FIL (sur CPU)."""
    return f"""name: "{nom}"
backend: "fil"
max_batch_size: {max_batch}
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {nombre_features} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }}
]
instance_group [ {{ kind: KIND_CPU }} ]
parameters [
  {{ key: "model_type" value: {{ string_value: "xgboost_json" }} }},
  {{ key: "output_class" value: {{ string_value: "true" }} }},
  {{ key: "predict_proba" value: {{ string_value: "true" }} }},
  {{ key: "threshold" value: {{ string_value: "0.5" }} }}
]
dynamic_batching {{ }}
"""


def config_onnx(nom, nombre_features, max_batch=1024):
    """Configuration Triton d'un autoencodeur ONNX (entrée et sortie de même taille, sur CPU)."""
    return f"""name: "{nom}"
backend: "onnxruntime"
max_batch_size: {max_batch}
input [
  {{
    name: "entree"
    data_type: TYPE_FP32
    dims: [ {nombre_features} ]
  }}
]
output [
  {{
    name: "reconstruction"
    data_type: TYPE_FP32
    dims: [ {nombre_features} ]
  }}
]
instance_group [ {{ kind: KIND_CPU }} ]
dynamic_batching {{ }}
"""


def couper_a_best_iteration(modele):
    """Ne garde que les arbres jusqu'à best_iteration, comme le fait XGBoost à la prédiction.

    Avec l'arrêt précoce, le fichier contient aussi les arbres construits après le meilleur
    point. XGBoost les ignore, mais Treelite/FIL les additionnerait tous.
    """
    meilleur = modele["learner"].get("attributes", {}).get("best_iteration")
    if meilleur is None:
        return 0
    interne = modele["learner"]["gradient_booster"]["model"]
    indptr = interne.get("iteration_indptr")
    garder = int(indptr[int(meilleur) + 1]) if indptr else int(meilleur) + 1
    retires = len(interne["trees"]) - garder
    interne["trees"] = interne["trees"][:garder]
    interne["tree_info"] = interne["tree_info"][:garder]
    interne["gbtree_model_param"]["num_trees"] = str(garder)
    if indptr:
        interne["iteration_indptr"] = indptr[: int(meilleur) + 2]
    return retires


def adapter_xgboost_pour_fil(chemin_source, chemin_cible):
    """Rend un JSON de XGBoost 3 lisible par le Treelite (ancien) de Triton 24.08."""
    modele = json.loads(Path(chemin_source).read_text(encoding="utf-8"))
    interne = modele["learner"]["gradient_booster"]["model"]
    categories = interne.pop("cats", None) or {}
    if any(categories.values()):
        raise ValueError("Le modèle utilise des variables catégorielles : « cats » est nécessaire.")
    parametres = modele["learner"]["learner_model_param"]
    base = str(parametres.get("base_score", "")).strip()
    if base.startswith("["):
        valeurs = [v for v in base.strip("[]").split(",") if v.strip()]
        if len(valeurs) != 1:
            raise ValueError(f"base_score à plusieurs valeurs non supporté : {base}")
        parametres["base_score"] = valeurs[0].strip()
    retires = couper_a_best_iteration(modele)
    Path(chemin_cible).write_text(json.dumps(modele, separators=(",", ":")), encoding="utf-8")
    return retires


def ecrire_modele_fil(depot, nom, chemin_xgb, nombre_features):
    """Écrit le modèle XGBoost (JSON adapté à FIL) et sa configuration dans le dépôt Triton."""
    dossier = Path(depot) / nom
    (dossier / "1").mkdir(parents=True, exist_ok=True)
    adapter_xgboost_pour_fil(chemin_xgb, dossier / "1" / "xgboost.json")
    (dossier / "config.pbtxt").write_text(config_fil(nom, nombre_features), encoding="utf-8")
    return dossier


def ecrire_modele_onnx(depot, nom, modele_onnx, nombre_features):
    """Écrit le modèle ONNX et sa configuration dans le dépôt Triton."""
    dossier = Path(depot) / nom
    (dossier / "1").mkdir(parents=True, exist_ok=True)
    (dossier / "1" / "model.onnx").write_bytes(modele_onnx.SerializeToString())
    (dossier / "config.pbtxt").write_text(config_onnx(nom, nombre_features), encoding="utf-8")
    return dossier
