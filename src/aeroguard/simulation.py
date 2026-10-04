"""Simulateur d'une flotte de moteurs d'avion qui s'usent."""

import numpy as np
import pandas as pd


def simuler_moteur(unit, n_vols, debut_usure, rng):
    """Simule tous les vols d'un moteur, du vol 1 jusqu'à la panne."""
    cycles = np.arange(1, n_vols + 1)
    usure = np.clip(cycles - debut_usure, 0, None)

    t48 = 520 + rng.normal(0, 5, n_vols) + 0.05 * usure**2
    p30 = 45 + rng.normal(0, 1, n_vols) - 0.005 * usure**2

    return pd.DataFrame(
        {
            "unit": unit,
            "cycle": cycles,
            "T48": t48,
            "P30": p30,
            "hs": (cycles <= debut_usure).astype(int),
            "rul": n_vols - cycles,
        }
    )


def simuler_flotte(n_moteurs=10, seed=42):
    """Simule une flotte complète et renvoie un seul DataFrame (une ligne par vol)."""
    rng = np.random.default_rng(seed)
    moteurs = []
    for unit in range(1, n_moteurs + 1):
        n_vols = int(rng.integers(80, 121))
        debut_usure = int(n_vols * rng.uniform(0.5, 0.7))
        moteurs.append(simuler_moteur(unit, n_vols, debut_usure, rng))
    return pd.concat(moteurs, ignore_index=True)


if __name__ == "__main__":
    flotte = simuler_flotte()
    flotte.to_parquet("data/processed/flotte_simulee.parquet", index=False)
    print(flotte.head())
    print(f"\n{flotte['unit'].nunique()} moteurs, {len(flotte)} vols au total")
    print(flotte.groupby("hs")[["T48", "P30"]].mean().round(1))
