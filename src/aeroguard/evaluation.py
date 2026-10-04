"""Évaluer les modèles : seuil, scores par classe, métriques et leaderboard."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def appliquer_seuil(probas, seuil=0.5):
    """1 (usé) si la probabilité est >= seuil, sinon 0 (sain)."""
    if not 0 <= seuil <= 1:
        raise ValueError("Le seuil doit être entre 0 et 1.")
    return (np.asarray(probas) >= seuil).astype(int)


def taux_par_classe(y_vrai, y_pred):
    """Exactitude, part des usés détectés, part des sains reconnus."""
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    return {
        "exactitude": float((y_vrai == y_pred).mean()),
        "uses_detectes": float((y_pred[y_vrai == 1] == 1).mean()),
        "sains_reconnus": float((y_pred[y_vrai == 0] == 0).mean()),
    }


def ajouter_au_leaderboard(chemin, modele, scores):
    """Ajoute une ligne (nom du modèle + scores) au fichier CSV, en le créant si besoin."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    ligne = pd.DataFrame([{"modele": modele, **scores}])
    if chemin.exists():
        ligne = pd.concat([pd.read_csv(chemin), ligne], ignore_index=True)
    ligne.to_csv(chemin, index=False)
    return ligne


def metriques(y_vrai, y_pred, probas=None):
    """Matrice de confusion, précision, rappel, F1, F1 macro et (si probas) PR-AUC."""
    vn, fp, fn, vp = confusion_matrix(y_vrai, y_pred, labels=[0, 1]).ravel()
    scores = {
        "precision": float(precision_score(y_vrai, y_pred, zero_division=0)),
        "rappel": float(recall_score(y_vrai, y_pred, zero_division=0)),
        "f1": float(f1_score(y_vrai, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro", zero_division=0)),
        "vn": int(vn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(vp),
    }
    if probas is not None:
        scores["pr_auc"] = float(average_precision_score(y_vrai, probas))
    return scores
