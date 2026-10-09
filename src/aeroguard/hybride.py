"""Système hybride : ligne de base par moteur, et alerte supervisée + anomalie (module 5)."""

import numpy as np

ETATS = ("normal", "anomalie inconnue", "usure connue", "usure confirmée")


def centrer_par_moteur(X, moteurs, cycles, n_premiers=10):
    """Soustrait à chaque vol la moyenne des `n_premiers` vols de SON moteur (sa ligne de base).

    Les premiers vols d'un moteur sont supposés sains (moteur neuf ou révisé) : le résultat mesure
    « ce qui a changé depuis que CE moteur était neuf », et efface la signature de sa flotte.
    """
    if n_premiers < 1:
        raise ValueError("n_premiers doit être au moins 1.")
    valeurs = np.asarray(X, dtype="float64")
    moteurs = np.asarray(moteurs)
    cycles = np.asarray(cycles)
    if not len(valeurs) == len(moteurs) == len(cycles):
        raise ValueError("X, moteurs et cycles doivent avoir la même longueur.")
    centre = np.empty_like(valeurs)
    for moteur in np.unique(moteurs):
        lignes = np.where(moteurs == moteur)[0]
        premiers = lignes[np.argsort(cycles[lignes], kind="stable")[:n_premiers]]
        centre[lignes] = valeurs[lignes] - valeurs[premiers].mean(axis=0)
    return centre


def etat_hybride(alerte_supervisee, alerte_anomalie):
    """Combine deux alertes confirmées (0/1) en un état lisible pour le mécanicien.

    supervisée 0 + anomalie 0 → normal ; 0 + 1 → anomalie inconnue ;
    1 + 0 → usure connue ; 1 + 1 → usure confirmée.
    """
    supervisee = np.asarray(alerte_supervisee).astype(int)
    anomalie = np.asarray(alerte_anomalie).astype(int)
    if supervisee.shape != anomalie.shape:
        raise ValueError("Les deux alertes doivent avoir la même forme.")
    return np.array(ETATS, dtype=object)[supervisee * 2 + anomalie]
