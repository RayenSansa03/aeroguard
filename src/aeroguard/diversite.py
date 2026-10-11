"""Mesurer la diversité de faux vols : détecter le mode collapse d'un GAN (module 6)."""

import numpy as np


def distances_plus_proche_voisin(A, B=None):
    """Distance de chaque ligne de A à sa plus proche voisine dans B (ou dans A, sans elle-même)."""
    A = np.asarray(A, dtype="float64")
    cible = A if B is None else np.asarray(B, dtype="float64")
    if A.ndim != 2 or cible.ndim != 2 or A.shape[1] != cible.shape[1]:
        raise ValueError("A et B doivent être des tableaux 2D avec le même nombre de features.")
    if B is None and len(A) < 2:
        raise ValueError("Il faut au moins 2 lignes pour chercher un voisin dans A.")
    carres = (A**2).sum(axis=1)[:, None] + (cible**2).sum(axis=1)[None, :] - 2 * A @ cible.T
    distances = np.sqrt(np.clip(carres, 0, None))
    if B is None:
        np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def diagnostic_diversite(X_reel, X_faux, quantile_queue=0.05, seuil_doublon=1e-3):
    """Compare la diversité des faux vols à celle des vrais (mêmes features, même échelle).

    - rapport_ecarts : écart-type moyen faux / vrais (1 = autant de variété)
    - part_features_ecrasees : part des features dont la variété est divisée par 2 ou plus
    - couverture_queues : valeurs extrêmes produites / attendues (1 = queues bien reproduites)
    - rapport_ppv : distance au plus proche voisin, faux / vrais (proche de 0 = faux collés)
    - part_doublons : part des faux vols quasi identiques à un autre faux vol
    """
    reel = np.asarray(X_reel, dtype="float64")
    faux = np.asarray(X_faux, dtype="float64")
    if reel.ndim != 2 or faux.ndim != 2 or reel.shape[1] != faux.shape[1]:
        raise ValueError("X_reel et X_faux doivent être 2D avec le même nombre de features.")
    if not 0 < quantile_queue < 0.5:
        raise ValueError("quantile_queue doit être strictement entre 0 et 0,5.")

    rapports = faux.std(axis=0) / np.clip(reel.std(axis=0), 1e-8, None)
    bas, haut = np.quantile(reel, [quantile_queue, 1 - quantile_queue], axis=0)
    part_extremes = ((faux < bas) | (faux > haut)).mean()
    ppv_faux = distances_plus_proche_voisin(faux)
    ppv_reels = distances_plus_proche_voisin(reel)
    return {
        "rapport_ecarts": float(rapports.mean()),
        "part_features_ecrasees": float((rapports < 0.5).mean()),
        "couverture_queues": float(part_extremes / (2 * quantile_queue)),
        "rapport_ppv": float(np.median(ppv_faux) / max(np.median(ppv_reels), 1e-12)),
        "part_doublons": float((ppv_faux < seuil_doublon).mean()),
    }


def est_effondre(diagnostic, seuil=0.5, part_doublons_max=0.05):
    """True si le GAN montre un mode collapse : variété ou espacement divisés par 2, ou doublons."""
    return bool(
        diagnostic["rapport_ecarts"] < seuil
        or diagnostic["rapport_ppv"] < seuil
        or diagnostic["part_doublons"] > part_doublons_max
    )
