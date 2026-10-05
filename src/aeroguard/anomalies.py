"""Détection non supervisée : score d'anomalie, seuil par percentile, alertes."""

import numpy as np


def score_anomalie(modele, X):
    """Score où plus grand = plus anormal (scikit-learn renvoie l'inverse)."""
    return -np.asarray(modele.score_samples(X))


def seuil_percentile(scores_sains, pourcentage=95):
    """Seuil = percentile des scores des vols sains (ex. 95 → ~5 % de fausses alarmes)."""
    if not 0 < pourcentage < 100:
        raise ValueError("Le pourcentage doit être strictement entre 0 et 100.")
    return float(np.percentile(scores_sains, pourcentage))


def alertes(scores, seuil):
    """1 (alerte) si le score dépasse strictement le seuil, sinon 0."""
    return (np.asarray(scores) > seuil).astype(int)
