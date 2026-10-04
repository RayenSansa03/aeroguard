"""Petits outils transverses."""
import time


def chrono(fonction, repetitions=3):
    """Lance la fonction plusieurs fois et retourne le MEILLEUR temps (secondes)."""
    temps = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        temps.append(time.perf_counter() - debut)
    return min(temps)