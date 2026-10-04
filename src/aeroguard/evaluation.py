"""Évaluer les modèles : seuil de décision, scores par classe, leaderboard."""

from pathlib import Path

import numpy as np
import pandas as pd


def appliquer_seuil(probas, seuil=0.5):
    """1 (usé) si la probabilité est >= seuil, sinon 0 (sain)."""
    if not 0 <= seuil <= 1:
        raise ValueError("Le seuil doit être entre 0 et 1.")
    return (np.asarray(probas, dtype=float) >= seuil).astype(int)


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
    ligne = pd.DataFrame([{"modele": modele, **scores}])
    if chemin.exists():
        ligne = pd.concat([pd.read_csv(chemin), ligne], ignore_index=True)
    ligne.to_csv(chemin, index=False)
    return ligne
