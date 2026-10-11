"""Service de décision en ligne : historique des vols par moteur et modèles champions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from aeroguard.decision import K_AE, SEUIL_AE, decider
from aeroguard.depot import DepotHistorique, DepotMemoire, HistoriqueMoteur

FICHIERS_MODELES = {
    "alerte": "xgb_binaire_v3.json",
    "anomalie": "autoencodeur_centre_v5.keras",
    "stats": "autoencodeur_centre_v5_stats.npz",
    "config": "autoencodeur_centre_v5_alerte.json",
}
FICHIER_ONNX = "autoencodeur_centre_v5.onnx"


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


class ServiceDecision:
    """Transforme chaque vol reçu en recommandation de maintenance.

    Les modèles sont injectés sous forme de fonctions : les tests utilisent des
    faux modèles légers, la production utilise les vrais modèles champions.
    L'historique des moteurs est confié à un dépôt (en mémoire par défaut).
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
        depot: DepotHistorique | None = None,
    ):
        self.colonnes = list(colonnes)
        self.predire_proba = predire_proba
        self.erreurs_ae = erreurs_ae
        self.centrer = centrer
        self.medianes = np.asarray(medianes, dtype="float64")
        self.seuil_ae = seuil_ae
        self.k_ae = k_ae
        self.n_premiers = n_premiers
        self.depot = depot if depot is not None else DepotMemoire()

    def _vecteur(self, mesures: Mapping[str, float | None]) -> np.ndarray:
        inconnues = sorted(set(mesures) - set(self.colonnes))
        if inconnues:
            raise ValueError(f"Features inconnues : {inconnues}")
        valeurs = [mesures.get(c) for c in self.colonnes]
        return np.array([np.nan if v is None else v for v in valeurs], dtype="float64")

    def traiter_vol(self, moteur: str, cycle: int, mesures: Mapping[str, float | None]) -> dict:
        """Ajoute un vol à l'historique du moteur et renvoie la recommandation.

        Tout est calculé sur des copies puis écrit en une fois : si un modèle échoue,
        l'historique reste exactement dans son état précédent.
        """
        x_brut = self._vecteur(mesures)
        with self.depot.verrou(moteur):
            historique = self.depot.lire(moteur) or HistoriqueMoteur()
            if historique.cycles and cycle <= historique.cycles[-1]:
                raise ValueError(
                    f"Le cycle {cycle} doit être supérieur au dernier cycle reçu "
                    f"({historique.cycles[-1]}) pour le moteur {moteur}"
                )
            proba = float(self.predire_proba(x_brut[None, :]))
            x_impute = np.where(np.isnan(x_brut), self.medianes, x_brut)
            cycles = [*historique.cycles, int(cycle)]
            vols = [*historique.vols, x_impute]

            # Pendant les vols de la ligne de base, l'autoencodeur ne peut pas encore juger.
            if len(vols) <= self.n_premiers:
                erreurs_vol = np.zeros(len(self.colonnes))
            else:
                X_centre = self.centrer(np.vstack(vols), np.array(cycles))
                erreurs_vol = np.asarray(self.erreurs_ae(X_centre[-1][None, :]), dtype="float64")

            probas = [*historique.probas, proba]
            erreurs = [*historique.erreurs, float(erreurs_vol.mean())]
            recommandation = decider(
                np.array(probas),
                np.array(erreurs),
                seuil_ae=self.seuil_ae,
                k_ae=self.k_ae,
                capteurs=capteurs_en_cause(erreurs_vol, self.colonnes),
            )
            resultat = {
                "moteur": moteur,
                "cycle": int(cycle),
                "vols_recus": len(cycles),
                **recommandation.en_dict(),
            }
            self.depot.ecrire(
                moteur,
                HistoriqueMoteur(
                    cycles=cycles,
                    vols=vols,
                    probas=probas,
                    erreurs=erreurs,
                    recommandations=[*historique.recommandations, resultat],
                    derniere=resultat,
                ),
            )
            return resultat

    def derniere(self, moteur: str) -> dict | None:
        """Dernière recommandation d'un moteur, ou None si le moteur est inconnu."""
        historique = self.depot.lire(moteur)
        return historique.derniere if historique else None

    def historique(self, moteur: str) -> list[dict] | None:
        """Toutes les recommandations d'un moteur, vol après vol, ou None s'il est inconnu."""
        historique = self.depot.lire(moteur)
        return list(historique.recommandations) if historique else None

    def etat_flotte(self) -> list[dict]:
        """Une ligne de synthèse par moteur, les plus urgents en premier."""
        ordre = {"URGENT": 0, "A_SURVEILLER": 1, "OK": 2}
        lignes = [
            {
                "moteur": moteur,
                "vols_recus": derniere["vols_recus"],
                "niveau": derniere["niveau"],
                "cycle": derniere["cycle"],
            }
            for moteur, derniere in self.depot.dernieres().items()
        ]
        return sorted(lignes, key=lambda ligne: (ordre.get(ligne["niveau"], 3), ligne["moteur"]))

    def reinitialiser(self, moteur: str) -> bool:
        """Oublie un moteur (par exemple après une révision). False s'il était inconnu."""
        with self.depot.verrou(moteur):
            return self.depot.supprimer(moteur)


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


def charger_service_onnx(dossier: str | Path) -> ServiceDecision:
    """Construit le service avec ONNX Runtime (autoencodeur) et XGBoost CPU (alerte).

    Mode léger destiné à l'hébergement : ni TensorFlow ni Triton, mêmes calculs que les
    autres modes. Un seul thread par appel : le parallélisme vient des requêtes.
    """
    import onnxruntime as ort
    import xgboost

    from aeroguard.autoencodeurs import standardiser
    from aeroguard.hybride import centrer_par_moteur

    dossier = Path(dossier)
    requis = [
        FICHIERS_MODELES["alerte"],
        FICHIER_ONNX,
        FICHIERS_MODELES["stats"],
        FICHIERS_MODELES["config"],
    ]
    manquants = [nom for nom in requis if not (dossier / nom).exists()]
    if manquants:
        raise FileNotFoundError(f"Fichiers de modèles manquants dans {dossier} : {manquants}")

    stats = np.load(dossier / FICHIERS_MODELES["stats"], allow_pickle=True)
    config = json.loads((dossier / FICHIERS_MODELES["config"]).read_text(encoding="utf-8"))
    moyennes, ecarts = stats["moyennes"], stats["ecarts"]

    booster = xgboost.Booster()
    booster.load_model(str(dossier / FICHIERS_MODELES["alerte"]))
    booster.set_param({"nthread": 1})
    # Même nombre d'arbres que XGBClassifier.predict_proba après early stopping.
    meilleure = booster.attr("best_iteration")
    plage = (0, int(meilleure) + 1) if meilleure is not None else (0, 0)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(dossier / FICHIER_ONNX), options, providers=["CPUExecutionProvider"]
    )
    entree = session.get_inputs()[0].name

    def predire_proba(x):
        return float(booster.inplace_predict(x, iteration_range=plage)[0])

    def erreurs_ae(x_centre):
        xn = standardiser(x_centre, moyennes, ecarts).astype(np.float32)
        reconstruction = session.run(None, {entree: xn})[0]
        return ((xn - reconstruction) ** 2)[0]

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
