"""Augmenter une famille rare avec des vols synthétiques : SMOTE, gaussienne, rognage (module 6)."""

import numpy as np


def _verifier(X, n):
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X doit être un tableau 2D avec au moins 2 vols.")
    if n < 1:
        raise ValueError("n doit être supérieur ou égal à 1.")
    return X


def smote_simple(X, n, k=5, graine=0):
    """SMOTE : n nouveaux vols, chacun pris au hasard sur le segment entre un vol et un voisin."""
    X = _verifier(X, n)
    k = min(k, len(X) - 1)
    carres = (X**2).sum(axis=1)[:, None] + (X**2).sum(axis=1)[None, :] - 2 * X @ X.T
    np.fill_diagonal(carres, np.inf)
    voisins = np.argsort(carres, axis=1)[:, :k]

    rng = np.random.default_rng(graine)
    depart = rng.integers(0, len(X), n)
    arrivee = voisins[depart, rng.integers(0, k, n)]
    position = rng.random((n, 1))
    return X[depart] + position * (X[arrivee] - X[depart])


def gaussienne(X, n, graine=0):
    """n vols tirés d'une loi normale de même moyenne et mêmes corrélations que X."""
    X = _verifier(X, n)
    rng = np.random.default_rng(graine)
    return rng.multivariate_normal(X.mean(axis=0), np.cov(X, rowvar=False), size=n)


def rogner(X_faux, X_reel):
    """Ramène chaque feature des faux vols entre le min et le max observés sur les vrais vols."""
    reel = np.asarray(X_reel, dtype="float64")
    faux = np.asarray(X_faux, dtype="float64")
    if reel.ndim != 2 or faux.ndim != 2 or reel.shape[1] != faux.shape[1]:
        raise ValueError("X_faux et X_reel doivent être 2D avec le même nombre de features.")
    return np.clip(faux, reel.min(axis=0), reel.max(axis=0))


def plis_famille_rare(moteurs, familles, famille_rare, graine=0):
    """Validation croisée par moteur : un pli par moteur de la famille rare.

    Chaque pli évalue exactement un moteur de la famille rare, plus une part des autres
    moteurs (répartis au hasard). Renvoie la liste des masques d'évaluation (un par pli).
    """
    moteurs = np.asarray(moteurs)
    familles = np.asarray(familles)
    rares = np.unique(moteurs[familles == famille_rare])
    if len(rares) < 2:
        raise ValueError("Il faut au moins 2 moteurs de la famille rare.")
    autres = np.unique(moteurs[familles != famille_rare])
    np.random.default_rng(graine).shuffle(autres)
    return [
        np.isin(moteurs, np.r_[[moteur], autres[pli :: len(rares)]])
        for pli, moteur in enumerate(rares)
    ]
