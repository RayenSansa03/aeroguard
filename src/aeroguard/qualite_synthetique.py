"""Évaluer des faux vols : fidélité, mémorisation et utilité (TSTR) (module 6)."""

import numpy as np
import pandas as pd

from aeroguard.diversite import distances_plus_proche_voisin


def statistique_ks(a, b):
    """Distance de Kolmogorov-Smirnov : écart maximal entre les deux courbes cumulées (0 à 1)."""
    a = np.sort(np.asarray(a, dtype="float64"))
    b = np.sort(np.asarray(b, dtype="float64"))
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Les deux échantillons doivent contenir au moins une valeur.")
    valeurs = np.concatenate([a, b])
    cumul_a = np.searchsorted(a, valeurs, side="right") / len(a)
    cumul_b = np.searchsorted(b, valeurs, side="right") / len(b)
    return float(np.max(np.abs(cumul_a - cumul_b)))


def _ecart_interquartile(X):
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)
    return q3 - q1


def bilan_fidelite(X_reel, X_faux, colonnes=None):
    """Une ligne par feature : KS, écart des moyennes (en écarts-types réels), rapport des IQR.

    Le rapport des écarts interquartiles (IQR) mesure l'étalement sans être faussé
    par quelques valeurs aberrantes, contrairement à l'écart-type.
    Trié de la feature la moins bien reproduite (KS le plus grand) à la mieux reproduite.
    """
    reel = np.asarray(X_reel, dtype="float64")
    faux = np.asarray(X_faux, dtype="float64")
    if reel.ndim != 2 or faux.ndim != 2 or reel.shape[1] != faux.shape[1]:
        raise ValueError("X_reel et X_faux doivent être 2D avec le même nombre de features.")
    colonnes = list(colonnes) if colonnes is not None else [f"f{j}" for j in range(reel.shape[1])]
    if len(colonnes) != reel.shape[1]:
        raise ValueError("Il faut un nom de colonne par feature.")

    ecarts_reels = np.clip(reel.std(axis=0), 1e-8, None)
    bilan = pd.DataFrame(
        {
            "ks": [statistique_ks(reel[:, j], faux[:, j]) for j in range(reel.shape[1])],
            "ecart_moyennes": np.abs(faux.mean(axis=0) - reel.mean(axis=0)) / ecarts_reels,
            "rapport_iqr": _ecart_interquartile(faux)
            / np.clip(_ecart_interquartile(reel), 1e-8, None),
        },
        index=pd.Index(colonnes, name="feature"),
    )
    return bilan.sort_values("ks", ascending=False)


def resume_fidelite(bilan, seuil_ks=0.2):
    """Résumé du bilan : KS médian, part des features mal reproduites, moyennes et étalement."""
    return {
        "ks_median": float(bilan["ks"].median()),
        "part_ks_eleve": float((bilan["ks"] > seuil_ks).mean()),
        "ecart_moyennes_median": float(bilan["ecart_moyennes"].median()),
        "rapport_iqr_median": float(bilan["rapport_iqr"].median()),
    }


def mesurer_memorisation(X_train, X_faux, X_reference, quantile=0.01):
    """Le GAN a-t-il recopié des vols du train ?

    - rapport_dcr : distance médiane faux → train / distance médiane référence → train
      (référence = vrais vols jamais vus par le GAN ; proche de 0 = faux collés au train)
    - part_quasi_copies : part des faux plus proches d'un vol du train que ne le sont
      les vols du train entre eux (seuil = quantile 1 %) ; attendu ≈ `quantile` sans copie
    """
    train = np.asarray(X_train, dtype="float64")
    dcr_faux = distances_plus_proche_voisin(X_faux, train)
    dcr_reference = distances_plus_proche_voisin(X_reference, train)
    seuil = np.quantile(distances_plus_proche_voisin(train), quantile)
    return {
        "dcr_faux_median": float(np.median(dcr_faux)),
        "dcr_reference_median": float(np.median(dcr_reference)),
        "rapport_dcr": float(np.median(dcr_faux) / max(np.median(dcr_reference), 1e-12)),
        "part_quasi_copies": float((dcr_faux < seuil).mean()),
    }


def _classifieur(graine):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=graine))


def auc_separabilite(X_reel, X_faux, graine=42, plis=5):
    """AUC d'un classifieur qui essaie de séparer vrais et faux (0,5 = impossible à distinguer)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.vstack([np.asarray(X_reel, dtype="float64"), np.asarray(X_faux, dtype="float64")])
    y = np.r_[np.zeros(len(X_reel)), np.ones(len(X_faux))]
    validation = StratifiedKFold(n_splits=plis, shuffle=True, random_state=graine)
    scores = cross_val_score(_classifieur(graine), X, y, cv=validation, scoring="roc_auc")
    return float(scores.mean())


def score_classifieur(X_app, y_app, X_eval, y_eval, graine=42):
    """Entraîne une régression logistique et la note sur des données d'évaluation (vraies)."""
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    modele = _classifieur(graine).fit(np.asarray(X_app), np.asarray(y_app))
    probas = modele.predict_proba(np.asarray(X_eval))[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_eval, probas)),
        "exactitude_equilibree": float(
            balanced_accuracy_score(y_eval, (probas >= 0.5).astype(int))
        ),
    }
