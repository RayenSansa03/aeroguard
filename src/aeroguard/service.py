"""Service de décision en ligne : historique des vols par moteur et modèles champions."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aeroguard.decision import K_AE, SEUIL_AE, decider

FICHIERS_MODELES = {
    "alerte": "xgb_binaire_v3.json",
    "anomalie": "autoencodeur_centre_v5.keras",
    "stats": "autoencodeur_centre_v5_stats.npz",
    "config": "autoencodeur_centre_v5_alerte.json",
}


def capteurs_en_cause(erreurs_vol, colonnes, nombre=2):
    """Capteurs dont les features sont le plus mal reconstruites sur ce vol."""
    capteurs = []
    for j in np.argsort(np.asarray(erreurs_vol))[::-1]:
        capteur = colonnes[j].split("_")[0]
        if capteur not in capteurs:
            capteurs.append(capteur)
        if len(capteurs) == nombre:
            break
    return tuple(capteurs)


@dataclass
class HistoriqueMoteur:
    """Ce que le service garde en mémoire pour un moteur."""

    cycles: list = field(default_factory=list)
    vols: list = field(default_factory=list)
    probas: list = field(default_factory=list)
    erreurs: list = field(default_factory=list)
    recommandations: list = field(default_factory=list)
    derniere: dict | None = None


class ServiceDecision:
    """Transforme chaque vol reçu en recommandation de maintenance.

    Les modèles sont injectés sous forme de fonctions : les tests utilisent des
    faux modèles légers, la production utilise les vrais modèles champions.
    """

    def __init__(
        self,
        colonnes: list[str],
        predire_proba: Callable[[np.ndarray], float],
        erreurs_ae: Callable[[np.ndarray], np.ndarray],
        centrer: Callable[[np.ndarray, np.ndarray], np.ndarray],
        medianes: np.ndarray,
        seuil_ae: float = SEUIL_AE,
        k_ae: int = K_AE,
        n_premiers: int = 10,
    ):
        self.colonnes = list(colonnes)
        self.predire_proba = predire_proba
        self.erreurs_ae = erreurs_ae
        self.centrer = centrer
        self.medianes = np.asarray(medianes, dtype="float64")
        self.seuil_ae = seuil_ae
        self.k_ae = k_ae
        self.n_premiers = n_premiers
        self._moteurs: dict[str, HistoriqueMoteur] = {}
        self._verrous: dict[str, threading.Lock] = {}
        self._registre = threading.Lock()

    def _verrou_moteur(self, moteur: str) -> threading.Lock:
        """Verrou propre à un moteur : les vols d'un même moteur restent ordonnés,
        les moteurs différents sont traités en parallèle."""
        with self._registre:
            return self._verrous.setdefault(moteur, threading.Lock())

    def _vecteur(self, mesures: Mapping[str, float | None]) -> np.ndarray:
        inconnues = sorted(set(mesures) - set(self.colonnes))
        if inconnues:
            raise ValueError(f"Features inconnues : {inconnues}")
        valeurs = [mesures.get(c) for c in self.colonnes]
        return np.array([np.nan if v is None else v for v in valeurs], dtype="float64")

    def traiter_vol(self, moteur: str, cycle: int, mesures: Mapping[str, float | None]) -> dict:
        """Ajoute un vol à l'historique du moteur et renvoie la recommandation."""
        x_brut = self._vecteur(mesures)
        with self._verrou_moteur(moteur):
            with self._registre:
                historique = self._moteurs.setdefault(moteur, HistoriqueMoteur())
            if historique.cycles and cycle <= historique.cycles[-1]:
                raise ValueError(
                    f"Le cycle {cycle} doit être supérieur au dernier cycle reçu "
                    f"({historique.cycles[-1]}) pour le moteur {moteur}"
                )
            proba = float(self.predire_proba(x_brut[None, :]))
            x_impute = np.where(np.isnan(x_brut), self.medianes, x_brut)
            historique.cycles.append(int(cycle))
            historique.vols.append(x_impute)

            # Pendant les vols de la ligne de base, l'autoencodeur ne peut pas encore juger.
            if len(historique.vols) <= self.n_premiers:
                erreurs_vol = np.zeros(len(self.colonnes))
            else:
                X_centre = self.centrer(np.vstack(historique.vols), np.array(historique.cycles))
                erreurs_vol = np.asarray(self.erreurs_ae(X_centre[-1][None, :]), dtype="float64")
            erreur = float(erreurs_vol.mean())

            historique.probas.append(proba)
            historique.erreurs.append(erreur)
            recommandation = decider(
                np.array(historique.probas),
                np.array(historique.erreurs),
                seuil_ae=self.seuil_ae,
                k_ae=self.k_ae,
                capteurs=capteurs_en_cause(erreurs_vol, self.colonnes),
            )
            resultat = {
                "moteur": moteur,
                "cycle": int(cycle),
                "vols_recus": len(historique.cycles),
                **recommandation.en_dict(),
            }
            historique.recommandations.append(resultat)
            historique.derniere = resultat
            return resultat

    def derniere(self, moteur: str) -> dict | None:
        """Dernière recommandation d'un moteur, ou None si le moteur est inconnu."""
        historique = self._moteurs.get(moteur)
        return historique.derniere if historique else None

    def historique(self, moteur: str) -> list[dict] | None:
        """Toutes les recommandations d'un moteur, vol après vol, ou None s'il est inconnu."""
        historique = self._moteurs.get(moteur)
        return list(historique.recommandations) if historique else None

    def _copie_moteurs(self) -> list[tuple[str, HistoriqueMoteur]]:
        """Instantané du registre, pour le parcourir sans bloquer les autres requêtes."""
        with self._registre:
            return list(self._moteurs.items())

    def etat_flotte(self) -> list[dict]:
        """Une ligne de synthèse par moteur, les plus urgents en premier."""
        ordre = {"URGENT": 0, "A_SURVEILLER": 1, "OK": 2}
        lignes = [
            {
                "moteur": moteur,
                "vols_recus": len(h.cycles),
                "niveau": h.derniere["niveau"],
                "cycle": h.derniere["cycle"],
            }
            for moteur, h in self._copie_moteurs()
            if h.derniere is not None
        ]
        return sorted(lignes, key=lambda ligne: (ordre.get(ligne["niveau"], 3), ligne["moteur"]))

    def reinitialiser(self, moteur: str) -> bool:
        """Oublie un moteur (par exemple après une révision). False s'il était inconnu."""
        with self._verrou_moteur(moteur), self._registre:
            return self._moteurs.pop(moteur, None) is not None


def charger_service(dossier: str | Path) -> ServiceDecision:
    """Construit le service à partir des fichiers des modèles champions."""
    dossier = Path(dossier)
    manquants = [nom for nom in FICHIERS_MODELES.values() if not (dossier / nom).exists()]
    if manquants:
        raise FileNotFoundError(f"Fichiers de modèles manquants dans {dossier} : {manquants}")
    return _charger_modeles(dossier)  # pragma: no cover


def charger_service_triton(dossier: str | Path, url: str, client=None) -> ServiceDecision:
    """Construit le service en déléguant les deux modèles à Triton.

    L'API ne garde que la logique (imputation, centrage, standardisation, décision) : ni
    TensorFlow ni XGBoost ne sont nécessaires, seulement les statistiques et la configuration.
    """
    from aeroguard.autoencodeurs import standardiser
    from aeroguard.hybride import centrer_par_moteur

    dossier = Path(dossier)
    requis = [FICHIERS_MODELES["stats"], FICHIERS_MODELES["config"]]
    manquants = [nom for nom in requis if not (dossier / nom).exists()]
    if manquants:
        raise FileNotFoundError(f"Fichiers de modèles manquants dans {dossier} : {manquants}")
    if client is None:
        from aeroguard.client_triton import ClientTriton

        client = ClientTriton(url)
    for modele in ("xgb_alerte", "autoencodeur"):
        if not client.pret(modele):
            raise ConnectionError(f"Triton n'est pas prêt pour {modele} ({url})")

    stats = np.load(dossier / FICHIERS_MODELES["stats"], allow_pickle=True)
    config = json.loads((dossier / FICHIERS_MODELES["config"]).read_text(encoding="utf-8"))
    moyennes, ecarts = stats["moyennes"], stats["ecarts"]

    def predire_proba(x):
        return client.probas_usure(x)[0]

    def erreurs_ae(x_centre):
        # Même calcul que autoencodeurs.erreur_reconstruction (par_feature=True), en float32.
        xn = standardiser(x_centre, moyennes, ecarts)
        return ((xn - client.reconstruction(xn)) ** 2)[0]

    def centrer(X, cycles):
        moteurs = np.zeros(len(cycles), dtype=int)
        return centrer_par_moteur(X, moteurs, cycles, n_premiers=config["n_premiers"])

    return ServiceDecision(
        colonnes=[str(c) for c in stats["colonnes"]],
        predire_proba=predire_proba,
        erreurs_ae=erreurs_ae,
        centrer=centrer,
        medianes=stats["medianes_imputation"],
        seuil_ae=config["seuil"],
        k_ae=config["k"],
        n_premiers=config["n_premiers"],
    )


def _charger_modeles(dossier: Path) -> ServiceDecision:  # pragma: no cover
    # Imports ML lourds ici : l'API démarre sans eux (par exemple /health dans Docker).
    import pandas as pd
    import xgboost
    from tensorflow import keras

    from aeroguard.autoencodeurs import standardiser
    from aeroguard.hybride import centrer_par_moteur

    stats = np.load(dossier / FICHIERS_MODELES["stats"], allow_pickle=True)
    config = json.loads((dossier / FICHIERS_MODELES["config"]).read_text(encoding="utf-8"))
    colonnes = [str(c) for c in stats["colonnes"]]

    xgb = xgboost.XGBClassifier()
    xgb.load_model(str(dossier / FICHIERS_MODELES["alerte"]))
    xgb.set_params(device="cpu")
    autoencodeur = keras.models.load_model(str(dossier / FICHIERS_MODELES["anomalie"]))

    def predire_proba(x):
        return xgb.predict_proba(pd.DataFrame(x, columns=colonnes))[0, 1]

    def erreurs_ae(x_centre):

        xn = standardiser(x_centre, stats["moyennes"], stats["ecarts"])
        reconstruction = autoencodeur(xn, training=False).numpy()
        return ((xn - reconstruction) ** 2)[0]

    def centrer(X, cycles):
        moteurs = np.zeros(len(cycles), dtype=int)
        return centrer_par_moteur(X, moteurs, cycles, n_premiers=config["n_premiers"])

    return ServiceDecision(
        colonnes=colonnes,
        predire_proba=predire_proba,
        erreurs_ae=erreurs_ae,
        centrer=centrer,
        medianes=stats["medianes_imputation"],
        seuil_ae=config["seuil"],
        k_ae=config["k"],
        n_premiers=config["n_premiers"],
    )
