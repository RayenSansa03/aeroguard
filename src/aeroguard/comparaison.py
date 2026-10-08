"""Comparer des modèles honnêtement : bootstrap par moteur et intervalles de confiance."""

import numpy as np
import pandas as pd


def f1_macro_depuis_comptes(vp, fp, fn, vn):
    """F1 macro (moyenne du F1 « usé » et du F1 « sain ») à partir des 4 comptes."""
    vp, fp, fn, vn = (np.asarray(c, dtype="float64") for c in (vp, fp, fn, vn))
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_use = np.where(2 * vp + fp + fn > 0, 2 * vp / (2 * vp + fp + fn), 0.0)
        f1_sain = np.where(2 * vn + fn + fp > 0, 2 * vn / (2 * vn + fn + fp), 0.0)
    return (f1_use + f1_sain) / 2


def comptes_par_moteur(moteurs, y_vrai, y_pred):
    """Tableau (une ligne par moteur) des comptes vp, fp, fn, vn."""
    table = pd.DataFrame(
        {
            "moteur": np.asarray(moteurs),
            "vp": (np.asarray(y_vrai) == 1) & (np.asarray(y_pred) == 1),
            "fp": (np.asarray(y_vrai) == 0) & (np.asarray(y_pred) == 1),
            "fn": (np.asarray(y_vrai) == 1) & (np.asarray(y_pred) == 0),
            "vn": (np.asarray(y_vrai) == 0) & (np.asarray(y_pred) == 0),
        }
    )
    return table.groupby("moteur", sort=True)[["vp", "fp", "fn", "vn"]].sum()


def bootstrap_moteurs(moteurs, y_vrai, predictions, n_tirages=1000, graine=42):
    """F1 macro de chaque modèle sur `n_tirages` flottes tirées au hasard (moteurs avec remise).

    `predictions` est un dictionnaire {nom du modèle: alertes 0/1}, une valeur par vol.
    Tous les modèles sont jugés sur les MÊMES tirages : la comparaison est appariée.
    Renvoie un DataFrame : une ligne par tirage, une colonne par modèle.
    """
    if n_tirages < 1:
        raise ValueError("n_tirages doit être au moins 1.")
    if not predictions:
        raise ValueError("Il faut au moins un modèle dans `predictions`.")
    n_vols = len(y_vrai)
    if len(moteurs) != n_vols or any(len(p) != n_vols for p in predictions.values()):
        raise ValueError("moteurs, y_vrai et chaque prédiction doivent avoir la même longueur.")

    comptes = {
        nom: comptes_par_moteur(moteurs, y_vrai, pred).to_numpy()
        for nom, pred in predictions.items()
    }
    n_moteurs = len(next(iter(comptes.values())))
    rng = np.random.default_rng(graine)
    tirages = rng.integers(0, n_moteurs, size=(n_tirages, n_moteurs))
    # Combien de fois chaque moteur est tiré, pour chaque tirage : (n_tirages, n_moteurs)
    multiplicites = np.stack([np.bincount(t, minlength=n_moteurs) for t in tirages])

    resultats = {}
    for nom, c in comptes.items():
        totaux = multiplicites @ c  # (n_tirages, 4) : vp, fp, fn, vn de chaque flotte tirée
        resultats[nom] = f1_macro_depuis_comptes(*totaux.T)
    return pd.DataFrame(resultats)


def intervalle_confiance(scores, niveau=0.95):
    """Intervalle des percentiles : (borne basse, borne haute) qui contient `niveau` des scores."""
    if not 0 < niveau < 1:
        raise ValueError("Le niveau doit être strictement entre 0 et 1.")
    alpha = (1 - niveau) / 2 * 100
    bas, haut = np.percentile(np.asarray(scores, dtype="float64"), [alpha, 100 - alpha])
    return float(bas), float(haut)


def comparer_a_reference(tirages, reference, niveau=0.95):
    """Pour chaque modèle : écart moyen à la référence, son intervalle, et P(meilleur).

    Si l'intervalle de l'écart contient 0, la différence n'est pas prouvée.
    """
    if reference not in tirages.columns:
        raise ValueError(f"Le modèle de référence « {reference} » est absent des tirages.")
    lignes = []
    for nom in tirages.columns:
        ecarts = tirages[nom] - tirages[reference]
        bas, haut = intervalle_confiance(ecarts, niveau)
        lignes.append(
            {
                "modele": nom,
                "f1_macro_moyen": float(tirages[nom].mean()),
                "ecart_moyen": float(ecarts.mean()),
                "ecart_bas": bas,
                "ecart_haut": haut,
                "proba_meilleur": float((ecarts > 0).mean()),
                "difference_prouvee": bool(bas > 0 or haut < 0),
            }
        )
    return pd.DataFrame(lignes).set_index("modele")
