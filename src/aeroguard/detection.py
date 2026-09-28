"""Fonctions de base pour détecter des anomalies dans les mesures des moteurs."""

import numpy as np


def ecart_absolu(mesures, reference):
    """Renvoie l'écart absolu entre chaque mesure et une valeur de référence.

    Exemple : ecart_absolu([510, 530], 520) -> array([10., 10.])
    """
    mesures = np.asarray(mesures, dtype=float)
    return np.abs(mesures - reference)


def score_z(mesures, donnees_saines):
    """Renvoie le score z de chaque mesure, par rapport à des données SAINES.
    """
    mesures = np.asarray(mesures, dtype=float)
    donnees_saines = np.asarray(donnees_saines, dtype=float)
    moyenne = donnees_saines.mean()
    ecart_type = donnees_saines.std()
    if ecart_type == 0:                                        
        raise ValueError(                                      
            "Écart-type nul dans les données saines "          
            "(capteur bloqué ?) : score z impossible."         
        )                                                      
    return (mesures - moyenne) / ecart_type


def detecter_anomalies(scores, seuil=3.0):
    """Renvoie True pour chaque score dont la valeur absolue dépasse le seuil.
    """
    scores = np.asarray(scores, dtype=float)
    return np.abs(scores) > seuil


def etat_moteur(ecart, seuil_surveiller=40, seuil_urgent=80):
    """Renvoie l'état d'un moteur selon son écart : "urgent", "surveiller" ou "ok".
    """
    if ecart > seuil_urgent:
        return "urgent"
    elif ecart > seuil_surveiller:
        return "surveiller"
    else:
        return "ok"