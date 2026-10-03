"""Fonctions de détection d'anomalies pour AeroGuard."""

import numpy as np


def ecart_absolu(mesures, reference):
    """Écart entre chaque mesure et la référence, toujours positif."""
    return np.abs(np.asarray(mesures, dtype=float) - reference)


def score_z(mesures, donnees_saines):
    """À combien d'écarts-types chaque mesure est de la moyenne des données saines."""
    mesures = np.asarray(mesures, dtype=float)
    donnees_saines = np.asarray(donnees_saines, dtype=float)
    moyenne = donnees_saines.mean()
    ecart_type = donnees_saines.std()
    if ecart_type == 0:
        raise ValueError(
            "Écart-type nul dans les données saines (capteur bloqué ?) : "
            "impossible de calculer un score z."
        )
    return (mesures - moyenne) / ecart_type


def detecter_anomalies(scores, seuil=3.0):
    """True si le score dépasse le seuil, dans un sens ou dans l'autre."""
    return np.abs(np.asarray(scores, dtype=float)) > seuil


def etat_moteur(ecart, seuil_surveiller=40, seuil_urgent=80):
    """Classe un écart en 'ok', 'surveiller' ou 'urgent'."""
    if ecart > seuil_urgent:
        return "urgent"
    if ecart > seuil_surveiller:
        return "surveiller"
    return "ok"


def confirmer_alarmes(alarmes, k=3):
    """Alarme confirmée seulement après k alarmes de suite."""
    if k < 1:
        raise ValueError("k doit être au moins 1.")
    alarmes = np.asarray(alarmes, dtype=bool)
    confirmees = np.zeros(len(alarmes), dtype=bool)
    compteur = 0
    for i, alarme in enumerate(alarmes):
        compteur = compteur + 1 if alarme else 0
        confirmees[i] = compteur >= k
    return confirmees


def premiere_alarme(alarmes, k=3):
    """Position de la 1re alarme confirmée, ou None s'il n'y en a pas."""
    positions = np.flatnonzero(confirmer_alarmes(alarmes, k))
    return int(positions[0]) if len(positions) > 0 else None