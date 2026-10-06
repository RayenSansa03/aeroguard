"""Métriques métier par moteur : alertes confirmées, avance d'alerte, fausses alarmes."""

import numpy as np
import pandas as pd


def alertes_confirmees(alertes, k=1):
    """1 seulement si les k derniers vols (celui-ci compris) ont tous sonné."""
    if k < 1:
        raise ValueError("k doit être au moins 1.")
    serie = pd.Series(np.asarray(alertes, dtype=int))
    return (serie.rolling(k, min_periods=k).sum() == k).astype(int).to_numpy()


def avance_alerte(alertes, y, rul, k=1):
    """RUL au 1er vol usé avec alerte confirmée ; NaN si le moteur est raté."""
    conf = alertes_confirmees(alertes, k)
    vrais = np.where((conf == 1) & (np.asarray(y) == 1))[0]
    if len(vrais) == 0:
        return np.nan
    return float(np.asarray(rul)[vrais[0]])


def fausses_alarmes(alertes, y, k=1):
    """Nombre de vols sains avec une alerte confirmée."""
    conf = alertes_confirmees(alertes, k)
    return int(((conf == 1) & (np.asarray(y) == 0)).sum())


def bilan_flotte(table, col_moteur="unit", col_ordre="cycle", col_rul="RUL", k=1):
    """Une ligne par moteur : avance d'alerte et nombre de fausses alarmes.

    `table` doit contenir les colonnes moteur, ordre, RUL, `y` (1 = usé) et `alerte` (0/1).
    """
    lignes = []
    for moteur, vols in table.sort_values(col_ordre).groupby(col_moteur, sort=True):
        lignes.append(
            {
                "moteur": moteur,
                "avance": avance_alerte(vols["alerte"], vols["y"], vols[col_rul], k),
                "fausses_alarmes": fausses_alarmes(vols["alerte"], vols["y"], k),
            }
        )
    return pd.DataFrame(lignes)


def resume_flotte(bilan):
    """Résumé de la flotte : moteurs détectés, avance moyenne et médiane, fausses alarmes."""
    return {
        "moteurs": int(len(bilan)),
        "moteurs_detectes": int(bilan["avance"].notna().sum()),
        "avance_moyenne": float(bilan["avance"].mean()),
        "avance_mediane": float(bilan["avance"].median()),
        "moteurs_avec_fausse_alarme": int((bilan["fausses_alarmes"] > 0).sum()),
        "fausses_alarmes_total": int(bilan["fausses_alarmes"].sum()),
    }


def diagnostic_par_moteur(moteurs, predictions):
    """Pour chaque moteur, la famille prédite le plus souvent sur ses vols (vote majoritaire)."""
    table = pd.DataFrame({"moteur": list(moteurs), "prediction": list(predictions)})
    return table.groupby("moteur")["prediction"].agg(lambda s: s.value_counts().idxmax())


def evaluer_alerte(
    y_vrai, probas, seuil, infos, k=3, col_moteur="moteur", col_ordre="cycle", col_rul="RUL"
):
    """Bilan complet d'une alerte : scores par vol + bilan par moteur (avance, fausses alarmes).

    `infos` contient une ligne par vol, dans le même ordre que `y_vrai` et `probas`,
    avec les colonnes moteur, ordre (cycle) et RUL.
    """
    from aeroguard.evaluation import appliquer_seuil, metriques

    alerte = appliquer_seuil(probas, seuil)
    scores = metriques(y_vrai, alerte, probas)
    table = infos[[col_moteur, col_ordre, col_rul]].copy()
    table["y"] = list(y_vrai)
    table["alerte"] = alerte
    bilan = bilan_flotte(table, col_moteur, col_ordre, col_rul, k=k)
    return {**scores, **resume_flotte(bilan)}
