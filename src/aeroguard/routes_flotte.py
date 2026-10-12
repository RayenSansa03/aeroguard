"""Routes HTTP de la flotte : envoyer un vol, consulter ou réinitialiser un moteur."""

import logging
import os
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

routeur = APIRouter(prefix="/flotte", tags=["flotte"])
journal_decisions = logging.getLogger("aeroguard.decision")


class Vol(BaseModel):
    """Un vol envoyé par l'avion : identifiant du moteur, cycle et mesures."""

    moteur: str = Field(min_length=1, max_length=50)
    cycle: int = Field(ge=1)
    mesures: dict[str, float | None]


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
        raise HTTPException(status_code=503, detail=f"Modèles indisponibles : {erreur}") from erreur


Service = Annotated[Any, Depends(obtenir_service)]


@routeur.post("/vols")
def recevoir_vol(vol: Vol, service: Service) -> dict:
    """Ajoute un vol à l'historique du moteur et renvoie la recommandation."""
    try:
        resultat = service.traiter_vol(vol.moteur, vol.cycle, vol.mesures)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    journal_decisions.info(
        "decision",
        extra={
            "champs": {
                "engine": vol.moteur,
                "cycle": vol.cycle,
                "decision": resultat.get("niveau"),
                "wear_probability": resultat.get("proba_usure"),
            }
        },
    )
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
