"""Routes HTTP de la flotte.

Deux routeurs partagent le même service :
- routeur_fleet (/fleet/...) : l'API publique, entièrement en anglais, visible dans Swagger ;
- routeur (/flotte/...) : l'ancienne API en français, gardée pour la compatibilité
  (tableau de bord, scripts, CD) mais cachée de Swagger et marquée obsolète.
"""

import logging
import os
import re
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

routeur_fleet = APIRouter(prefix="/fleet", tags=["Fleet"])
routeur = APIRouter(prefix="/flotte", tags=["flotte"], include_in_schema=False, deprecated=True)
journal_decisions = logging.getLogger("aeroguard.decision")

NIVEAUX_EN = {"OK": "OK", "A_SURVEILLER": "WATCH", "URGENT": "URGENT"}
ETATS_EN = {
    "normal": "normal",
    "anomalie inconnue": "unknown anomaly",
    "usure connue": "known wear",
    "usure confirmée": "confirmed wear",
}


# ---------------------------------------------------------------- modèles publics (anglais)
class Flight(BaseModel):
    """One flight sent by the aircraft."""

    engine: str = Field(min_length=1, max_length=50, description="Engine identifier")
    cycle: int = Field(ge=1, description="Flight number of this engine (must increase)")
    measurements: dict[str, float | None] = Field(
        description="Flight features by name; null = missing sensor (replaced by the median)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"engine": "DEMO1", "cycle": 1, "measurements": {"T48_mean_montee": 5.9}}]
        }
    }


class Decision(BaseModel):
    """Maintenance recommendation for one engine, after its latest flight."""

    engine: str
    cycle: int | None = None
    flights_received: int | None = Field(default=None, description="Flights received so far")
    level: str = Field(description="OK, WATCH or URGENT")
    state: str = Field(description="normal, unknown anomaly, known wear or confirmed wear")
    wear_probability: float | None = Field(default=None, description="XGBoost probability")
    reconstruction_error: float | None = Field(default=None, description="Autoencoder error")
    component: str | None = None
    sensors: list[str] = Field(default=[], description="Sensors to inspect first")
    message: str


class EngineSummary(BaseModel):
    """One line per engine in the fleet overview."""

    engine: str
    level: str = Field(description="OK, WATCH or URGENT")
    cycle: int | None = None
    flights_received: int | None = None


class ResetResult(BaseModel):
    engine: str
    status: str


# ---------------------------------------------------------------- service partagé
@lru_cache(maxsize=1)
def obtenir_service():
    """Charge le service une seule fois, au premier appel (chargement paresseux).

    Si AEROGUARD_HF_DEPOT est défini, les modèles sont d'abord téléchargés depuis ce dépôt
    Hugging Face (révision AEROGUARD_HF_REVISION, token HF_TOKEN) dans AEROGUARD_MODELES.

    Mode d'inférence, par ordre de priorité :
    - AEROGUARD_TRITON_URL défini : les modèles tournent dans Triton ;
    - AEROGUARD_INFERENCE=onnx : ONNX Runtime et XGBoost CPU dans l'API (hébergement léger) ;
    - sinon : TensorFlow et XGBoost dans l'API (développement).
    """
    dossier = os.environ.get("AEROGUARD_MODELES", "local/models")
    url_triton = os.environ.get("AEROGUARD_TRITON_URL")
    inference = os.environ.get("AEROGUARD_INFERENCE", "local")
    depot_hf = os.environ.get("AEROGUARD_HF_DEPOT")
    try:
        if depot_hf:
            from aeroguard import modeles_hf

            modeles_hf.telecharger_modeles(
                depot_hf, dossier, revision=os.environ.get("AEROGUARD_HF_REVISION")
            )

        from aeroguard.service import (
            charger_service,
            charger_service_onnx,
            charger_service_triton,
        )

        if url_triton:
            return charger_service_triton(dossier, url_triton)
        if inference == "onnx":
            return charger_service_onnx(dossier)
        return charger_service(dossier)
    except (FileNotFoundError, ImportError, ConnectionError) as erreur:
        raise HTTPException(status_code=503, detail=f"Models unavailable: {erreur}") from erreur


Service = Annotated[Any, Depends(obtenir_service)]
Moteur = Annotated[str, Path(description="Engine identifier, for example DS02_14")]


# ---------------------------------------------------------------- traduction vers l'anglais
def message_anglais(resultat: dict) -> str:
    """Message de la recommandation en anglais, reconstruit à partir de l'état."""
    etat = resultat.get("etat", "normal")
    composant = resultat.get("composant")
    capteurs = resultat.get("capteurs") or []
    suffixe_composant = f", likely component: {composant}" if composant else ""
    if etat == "anomalie inconnue":
        suffixe_capteurs = f" (sensors: {', '.join(capteurs)})" if capteurs else ""
        return f"Unknown anomaly confirmed: inspect the engine{suffixe_capteurs}."
    if etat == "usure connue":
        proba = resultat.get("proba_usure") or 0.0
        return f"Known wear detected (probability {proba:.0%}){suffixe_composant}."
    if etat == "usure confirmée":
        return f"Wear confirmed by both models: priority maintenance{suffixe_composant}."
    return "No confirmed alert: normal operation."


def decision_anglaise(resultat: dict) -> dict:
    """Traduit une recommandation du service (clés françaises) vers le contrat public."""
    return {
        "engine": resultat.get("moteur"),
        "cycle": resultat.get("cycle"),
        "flights_received": resultat.get("vols_recus"),
        "level": NIVEAUX_EN.get(resultat.get("niveau"), resultat.get("niveau")),
        "state": ETATS_EN.get(resultat.get("etat", "normal"), resultat.get("etat")),
        "wear_probability": resultat.get("proba_usure"),
        "reconstruction_error": resultat.get("erreur_reconstruction"),
        "component": resultat.get("composant"),
        "sensors": list(resultat.get("capteurs") or []),
        "message": message_anglais(resultat),
    }


def resume_anglais(ligne: dict) -> dict:
    """Traduit une ligne de etat_flotte() vers le contrat public."""
    return {
        "engine": ligne.get("moteur"),
        "level": NIVEAUX_EN.get(ligne.get("niveau"), ligne.get("niveau")),
        "cycle": ligne.get("cycle"),
        "flights_received": ligne.get("vols_recus"),
    }


def erreur_anglaise(texte: str) -> str:
    """Traduit les erreurs métier connues du service ; les autres passent telles quelles."""
    features = re.match(r"Features inconnues : (.*)", texte)
    if features:
        return f"Unknown features: {features.group(1)}"
    cycle = re.match(
        r"Le cycle (\d+) doit être supérieur au dernier cycle reçu \((\d+)\) "
        r"pour le moteur (.+)",
        texte,
    )
    if cycle:
        nouveau, dernier, moteur = cycle.groups()
        return (
            f"Cycle {nouveau} must be greater than the last cycle received ({dernier}) "
            f"for engine {moteur}"
        )
    return texte


def journaliser_decision(moteur: str, cycle: int, resultat: dict) -> None:
    journal_decisions.info(
        "decision",
        extra={
            "champs": {
                "engine": moteur,
                "cycle": cycle,
                "decision": resultat.get("niveau"),
                "wear_probability": resultat.get("proba_usure"),
            }
        },
    )


# ---------------------------------------------------------------- API publique : /fleet
@routeur_fleet.post(
    "/flights",
    response_model=Decision,
    summary="Send one flight and get the maintenance decision",
    description="Adds the flight to the engine history and returns the recommendation. "
    "Flights of one engine must arrive in increasing cycle order.",
)
def recevoir_vol_fleet(vol: Flight, service: Service) -> dict:
    try:
        resultat = service.traiter_vol(vol.engine, vol.cycle, vol.measurements)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=erreur_anglaise(str(erreur))) from erreur
    journaliser_decision(vol.engine, vol.cycle, resultat)
    return decision_anglaise(resultat)


@routeur_fleet.get(
    "/engines",
    response_model=list[EngineSummary],
    summary="List engines, most urgent first",
)
def lister_moteurs_fleet(service: Service) -> list[dict]:
    return [resume_anglais(ligne) for ligne in service.etat_flotte()]


@routeur_fleet.get(
    "/engines/{engine}",
    response_model=Decision,
    summary="Get the latest decision for one engine",
)
def consulter_moteur_fleet(engine: Moteur, service: Service) -> dict:
    derniere = service.derniere(engine)
    if derniere is None:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine}")
    return decision_anglaise(derniere)


@routeur_fleet.get(
    "/engines/{engine}/history",
    response_model=list[Decision],
    summary="Get the decision history of one engine",
    description="All recommendations of the engine, flight after flight.",
)
def historique_moteur_fleet(engine: Moteur, service: Service) -> list[dict]:
    historique = service.historique(engine)
    if historique is None:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine}")
    return [decision_anglaise(ligne) for ligne in historique]


@routeur_fleet.delete(
    "/engines/{engine}",
    response_model=ResetResult,
    summary="Reset one engine (after maintenance)",
    description="Forgets the engine history, for example after a workshop visit.",
)
def reinitialiser_moteur_fleet(engine: Moteur, service: Service) -> dict:
    if not service.reinitialiser(engine):
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine}")
    return {"engine": engine, "status": "reset"}


# ---------------------------------------------------------------- ancienne API : /flotte
class Vol(BaseModel):
    """Un vol envoyé par l'avion : identifiant du moteur, cycle et mesures."""

    moteur: str = Field(min_length=1, max_length=50)
    cycle: int = Field(ge=1)
    mesures: dict[str, float | None]


@routeur.post("/vols")
def recevoir_vol(vol: Vol, service: Service) -> dict:
    """Ajoute un vol à l'historique du moteur et renvoie la recommandation."""
    try:
        resultat = service.traiter_vol(vol.moteur, vol.cycle, vol.mesures)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    journaliser_decision(vol.moteur, vol.cycle, resultat)
    return resultat


@routeur.get("/moteurs")
def lister_moteurs(service: Service) -> list[dict]:
    """Synthèse de la flotte, moteurs les plus urgents en premier."""
    return service.etat_flotte()


@routeur.get("/moteurs/{moteur}")
def consulter_moteur(moteur: str, service: Service) -> dict:
    """Dernière recommandation d'un moteur."""
    derniere = service.derniere(moteur)
    if derniere is None:
        raise HTTPException(status_code=404, detail=f"Moteur inconnu : {moteur}")
    return derniere


@routeur.get("/moteurs/{moteur}/historique")
def historique_moteur(moteur: str, service: Service) -> list[dict]:
    """Toutes les recommandations d'un moteur, vol après vol (pour le tableau de bord)."""
    historique = service.historique(moteur)
    if historique is None:
        raise HTTPException(status_code=404, detail=f"Moteur inconnu : {moteur}")
    return historique


@routeur.delete("/moteurs/{moteur}")
def reinitialiser_moteur(moteur: str, service: Service) -> dict:
    """Oublie l'historique d'un moteur, par exemple après une révision."""
    if not service.reinitialiser(moteur):
        raise HTTPException(status_code=404, detail=f"Moteur inconnu : {moteur}")
    return {"moteur": moteur, "statut": "reinitialise"}
