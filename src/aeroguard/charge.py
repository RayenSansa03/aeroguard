"""Test de charge : des clients envoient des vols en parallèle, on mesure latence et débit."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def resume_latences(latences_ms):
    """Médiane (p50), queue de distribution (p95, p99) et moyenne, en millisecondes."""
    if len(latences_ms) == 0:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "moyenne": None}
    valeurs = np.asarray(latences_ms, dtype=float)
    p50, p95, p99 = np.percentile(valeurs, [50, 95, 99])
    return {
        "n": len(valeurs),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "moyenne": float(valeurs.mean()),
    }


def executer_charge(sequences, envoyer, concurrence):
    """Envoie chaque séquence (les vols d'un moteur, dans l'ordre) depuis un même client.

    `concurrence` clients travaillent en même temps. Seules les requêtes réussies comptent
    dans les latences ; les échecs sont comptés à part.
    """

    def jouer(sequence):
        latences, erreurs = [], 0
        for vol in sequence:
            debut = time.perf_counter()
            try:
                envoyer(vol)
            except Exception:
                erreurs += 1
                continue
            latences.append((time.perf_counter() - debut) * 1000)
        return latences, erreurs

    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrence) as pool:
        resultats = list(pool.map(jouer, sequences))
    duree = time.perf_counter() - debut
    latences = [valeur for lot, _ in resultats for valeur in lot]
    return {
        **resume_latences(latences),
        "erreurs": sum(erreurs for _, erreurs in resultats),
        "duree_s": duree,
        "debit": len(latences) / duree if duree > 0 else 0.0,
    }
