"""Créer les modèles du projet et choisir leur seuil de décision."""

import numpy as np
from sklearn.metrics import f1_score

REGLAGES_XGB = {
    "n_estimators": 2000,  # maximum : l'early stopping s'arrêtera avant
    "learning_rate": 0.05,  # petits pas (leçon 3.1)
    "max_depth": 4,  # petits arbres
    "subsample": 0.8,  # chaque arbre voit 80 % des vols
    "colsample_bytree": 0.8,  # et 80 % des features
    "tree_method": "hist",
    "eval_metric": "aucpr",  # PR-AUC suivie sur la validation
    "early_stopping_rounds": 50,
    "random_state": 42,
}


def creer_xgboost(device="cuda", **reglages):
    """XGBoost binaire avec les réglages du projet ; `reglages` remplace les valeurs par défaut."""
    from xgboost import XGBClassifier

    return XGBClassifier(**{**REGLAGES_XGB, **reglages, "device": device})


def seuil_optimal(y_vrai, probas, seuils=None):
    """Le seuil qui maximise le F1 macro, et ce F1 macro."""
    probas = np.asarray(probas)
    if seuils is None:
        seuils = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [
        f1_score(y_vrai, (probas >= s).astype(int), average="macro", zero_division=0)
        for s in seuils
    ]
    meilleur = int(np.argmax(scores))
    return float(seuils[meilleur]), float(scores[meilleur])
