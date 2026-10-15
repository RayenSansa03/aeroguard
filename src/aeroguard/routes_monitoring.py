"""Routes de monitoring : dérive des données reçues par rapport à l'entraînement.

- /monitoring/drift : route publique, en anglais, visible dans Swagger ;
- /monitoring/derive : ancienne route, gardée pour la compatibilité (workflow de dérive,
  tableau de bord) mais cachée de Swagger.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from aeroguard.derive import FICHIER_REFERENCE, rapport_derive
from aeroguard.routes_flotte import Service

routeur = APIRouter(prefix="/monitoring", tags=["Monitoring"])
journal = logging.getLogger("aeroguard.monitoring")

# Un seul moteur n'est pas une flotte : en dessous, le rapport serait une fausse alarme.
MINIMUM_MOTEURS = 5


@lru_cache(maxsize=1)
def obtenir_reference() -> dict:
    """Référence de dérive (déciles d'entraînement), lue une fois à côté des modèles."""
    chemin = Path(os.environ.get("AEROGUARD_MODELES", "local/models")) / FICHIER_REFERENCE
    if not chemin.exists():
        raise HTTPException(status_code=503, detail=f"Drift reference not found: {chemin}")
    return json.loads(chemin.read_text(encoding="utf-8"))


Reference = Annotated[dict, Depends(obtenir_reference)]


def calculer_rapport(service, reference: dict, limite: int) -> dict:
    """Rapport de dérive (PSI par feature) sur les derniers vols reçus, journalisé."""
    n_moteurs = len(service.etat_flotte())
    if n_moteurs < MINIMUM_MOTEURS:
        rapport = {
            "status": "insufficient_engines",
            "n_engines": n_moteurs,
            "minimum_engines": MINIMUM_MOTEURS,
            "features": [],
        }
    else:
        rapport = rapport_derive(reference, service.vols_recents(limite))
        rapport["n_engines"] = n_moteurs
    journal.info(
        "drift report",
        extra={
            "champs": {
                "status": rapport["status"],
                "n_engines": n_moteurs,
                "n_flights": rapport.get("n_flights", 0),
                "n_major": rapport.get("n_major", 0),
            }
        },
    )
    return rapport


@routeur.get(
    "/drift",
    summary="Data drift report (PSI) on recent flights",
    description="Compares recent flights with the training data, feature by feature, "
    "with the Population Stability Index (PSI < 0.10 stable, 0.10 to 0.25 moderate, "
    ">= 0.25 major). Needs at least 50 recent flights from at least 5 engines; "
    "otherwise the status is insufficient_data or insufficient_engines.",
)
def drift(
    service: Service,
    reference: Reference,
    limit: Annotated[
        int, Query(ge=50, le=50_000, description="Number of most recent flights to compare")
    ] = 5000,
) -> dict:
    return calculer_rapport(service, reference, limit)


@routeur.get("/derive", include_in_schema=False)
def derive(
    service: Service,
    reference: Reference,
    limite: Annotated[int, Query(ge=50, le=50_000)] = 5000,
) -> dict:
    """Ancienne route (compatibilité) : même rapport que /monitoring/drift."""
    return calculer_rapport(service, reference, limite)
