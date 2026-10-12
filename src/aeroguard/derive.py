"""Dérive des données : les vols reçus ressemblent-ils encore à ceux de l'entraînement ?

Indice PSI (Population Stability Index) par feature : la référence découpe chaque feature
en déciles d'entraînement, puis on compare la part des vols récents dans chaque tranche.
Les clés du rapport sont en anglais : c'est une interface lue par des outils de monitoring.
"""

from __future__ import annotations

import numpy as np

SEUIL_MODEREE = 0.10
SEUIL_FORTE = 0.25
MINIMUM_VOLS = 50
FICHIER_REFERENCE = "reference_derive.json"


def _proportions(valeurs: np.ndarray, bords: np.ndarray) -> np.ndarray:
    """Part des valeurs dans chaque tranche délimitée par les bords intérieurs."""
    tranches = np.searchsorted(bords, valeurs, side="right")
    comptes = np.bincount(tranches, minlength=len(bords) + 1)
    return comptes / max(len(valeurs), 1)


def construire_reference(X, colonnes: list[str], n_tranches: int = 10) -> dict:
    """Déciles et proportions de chaque feature sur les données d'entraînement."""
    X = np.asarray(X, dtype="float64")
    quantiles = np.linspace(0, 1, n_tranches + 1)[1:-1]
    bords, proportions = [], []
    for j in range(X.shape[1]):
        colonne = X[:, j][~np.isnan(X[:, j])]
        interieurs = np.unique(np.quantile(colonne, quantiles))
        bords.append(interieurs.tolist())
        proportions.append(_proportions(colonne, interieurs).tolist())
    return {
        "colonnes": list(colonnes),
        "bords": bords,
        "proportions": proportions,
        "n_reference": int(len(X)),
    }


def psi(attendu, observe, epsilon: float = 1e-4) -> float:
    """Population Stability Index entre deux répartitions par tranche."""
    attendu = np.clip(np.asarray(attendu, dtype="float64"), epsilon, None)
    observe = np.clip(np.asarray(observe, dtype="float64"), epsilon, None)
    return float(np.sum((observe - attendu) * np.log(observe / attendu)))


def statut(valeur: float) -> str:
    """Lecture d'un PSI selon les seuils usuels de l'industrie."""
    if valeur >= SEUIL_FORTE:
        return "major"
    if valeur >= SEUIL_MODEREE:
        return "moderate"
    return "stable"


def rapport_derive(reference: dict, X, minimum: int = MINIMUM_VOLS) -> dict:
    """Compare des vols récents à la référence d'entraînement, feature par feature."""
    X = np.asarray(X, dtype="float64")
    if len(X) < minimum:
        return {
            "status": "insufficient_data",
            "n_flights": int(len(X)),
            "minimum": minimum,
            "features": [],
        }
    lignes = []
    for j, nom in enumerate(reference["colonnes"]):
        colonne = X[:, j][~np.isnan(X[:, j])]
        observe = _proportions(colonne, np.asarray(reference["bords"][j]))
        valeur = psi(reference["proportions"][j], observe)
        lignes.append({"feature": nom, "psi": round(valeur, 4), "status": statut(valeur)})
    lignes.sort(key=lambda ligne: ligne["psi"], reverse=True)
    n_major = sum(ligne["status"] == "major" for ligne in lignes)
    n_moderate = sum(ligne["status"] == "moderate" for ligne in lignes)
    global_ = "major" if n_major else "moderate" if n_moderate else "stable"
    return {
        "status": global_,
        "n_flights": int(len(X)),
        "n_major": n_major,
        "n_moderate": n_moderate,
        "features": lignes,
    }
