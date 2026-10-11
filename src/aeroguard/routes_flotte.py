"""Routes HTTP de la flotte : envoyer un vol, consulter ou réinitialiser un moteur."""

import os
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

routeur = APIRouter(prefix="/flotte", tags=["flotte"])


class Vol(BaseModel):
    """Un vol envoyé par l'avion : identifiant du moteur, cycle et mesures."""

    moteur: str = Field(min_length=1, max_length=50)
    cycle: int = Field(ge=1)
    mesures: dict[str, float | None]


@lru_cache(maxsize=1)
def obtenir_service():
    """Charge les modèles une seule fois, au premier appel (chargement paresseux)."""
    dossier = os.environ.get("AEROGUARD_MODELES", "local/models")
    try:
        from aeroguard.service import charger_service

        return charger_service(dossier)
    except (FileNotFoundError, ImportError) as erreur:
        raise HTTPException(status_code=503, detail=f"Modèles indisponibles : {erreur}") from erreur


Service = Annotated[Any, Depends(obtenir_service)]


@routeur.post("/vols")
def recevoir_vol(vol: Vol, service: Service) -> dict:
    """Ajoute un vol à l'historique du moteur et renvoie la recommandation."""
    try:
        return service.traiter_vol(vol.moteur, vol.cycle, vol.mesures)
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur


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


@routeur.delete("/moteurs/{moteur}")
def reinitialiser_moteur(moteur: str, service: Service) -> dict:
    """Oublie l'historique d'un moteur, par exemple après une révision."""
    if not service.reinitialiser(moteur):
        raise HTTPException(status_code=404, detail=f"Moteur inconnu : {moteur}")
    return {"moteur": moteur, "statut": "reinitialise"}
