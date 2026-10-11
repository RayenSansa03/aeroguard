"""Moteur de décision : une recommandation par moteur, à partir de XGBoost et de l'autoencodeur."""

from dataclasses import asdict, dataclass

import numpy as np

from aeroguard.hybride import ETATS

NIVEAUX = ("OK", "A_SURVEILLER", "URGENT")
SEUIL_XGB, K_XGB = 0.81, 3  # réglages figés de la v3
SEUIL_AE, K_AE = 0.618, 5  # autoencodeur centré de la v5


@dataclass(frozen=True)
class Recommandation:
    """Ce que le système dit au mécanicien pour un moteur."""

    niveau: str
    etat: str
    proba_usure: float
    erreur_reconstruction: float
    composant: str | None
    capteurs: tuple
    message: str

    def en_dict(self):
        """Version prête pour le JSON (l'API du module 7)."""
        donnees = asdict(self)
        donnees["capteurs"] = list(self.capteurs)
        return donnees


def _confirmee(valeurs, seuil, k, strict):
    """True si les k derniers vols dépassent le seuil (strictement ou non)."""
    derniers = valeurs[-k:]
    if len(derniers) < k:
        return False
    return bool((derniers > seuil).all() if strict else (derniers >= seuil).all())


def decider(
    probas_xgb,
    erreurs_ae,
    seuil_xgb=SEUIL_XGB,
    k_xgb=K_XGB,
    seuil_ae=SEUIL_AE,
    k_ae=K_AE,
    proba_urgente=0.95,
    facteur_urgent_ae=2.0,
    composant=None,
    capteurs=(),
):
    """Recommandation pour UN moteur, à partir de ses derniers vols (du plus ancien au plus récent).

    - XGBoost confirme une usure connue si ses k_xgb dernières probabilités sont >= seuil_xgb.
    - L'autoencodeur confirme une anomalie si ses k_ae dernières erreurs sont > seuil_ae.
    - Niveau : OK (rien), A_SURVEILLER (un seul modèle, signal modéré), URGENT (les deux
      modèles, ou un seul avec un signal très fort).
    """
    probas = np.atleast_1d(np.asarray(probas_xgb, dtype="float64"))
    erreurs = np.atleast_1d(np.asarray(erreurs_ae, dtype="float64"))
    if len(probas) == 0 or len(probas) != len(erreurs):
        raise ValueError("Il faut autant de probabilités que d'erreurs, et au moins un vol.")
    if ((probas < 0) | (probas > 1)).any():
        raise ValueError("Les probabilités XGBoost doivent être entre 0 et 1.")
    if k_xgb < 1 or k_ae < 1:
        raise ValueError("k_xgb et k_ae doivent être supérieurs ou égaux à 1.")

    supervise = _confirmee(probas, seuil_xgb, k_xgb, strict=False)
    anomalie = _confirmee(erreurs, seuil_ae, k_ae, strict=True)
    index = 2 * int(supervise) + int(anomalie)
    proba, erreur = float(probas[-1]), float(erreurs[-1])
    nom_composant = f", composant probable : {composant}" if supervise and composant else ""
    liste_capteurs = (
        f" (capteurs en cause : {', '.join(capteurs)})" if anomalie and capteurs else ""
    )

    if index == 0:
        niveau, message = "OK", "Aucune alerte confirmée : fonctionnement normal."
    elif index == 1:
        niveau = "URGENT" if erreur >= facteur_urgent_ae * seuil_ae else "A_SURVEILLER"
        message = f"Anomalie inconnue confirmée sur {k_ae} vols : inspection{liste_capteurs}."
    elif index == 2:
        niveau = "URGENT" if proba >= proba_urgente else "A_SURVEILLER"
        message = f"Usure connue détectée (probabilité {proba:.0%}){nom_composant}."
    else:
        niveau = "URGENT"
        message = f"Usure confirmée par les deux modèles : intervention prioritaire{nom_composant}."

    return Recommandation(
        niveau=niveau,
        etat=ETATS[index],
        proba_usure=proba,
        erreur_reconstruction=erreur,
        composant=composant if supervise else None,
        capteurs=tuple(capteurs) if anomalie else (),
        message=message,
    )
