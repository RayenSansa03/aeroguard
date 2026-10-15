"""API AeroGuard : santé du service, mesure des requêtes, flotte et monitoring."""

import logging
import os
import time
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request

from aeroguard.journalisation import configurer_logs
from aeroguard.routes_flotte import obtenir_service, routeur_fleet
from aeroguard.routes_flotte import routeur as routeur_flotte
from aeroguard.routes_monitoring import routeur as routeur_monitoring

configurer_logs(os.getenv("AEROGUARD_LOG_LEVEL", "INFO"))
journal_http = logging.getLogger("aeroguard.http")

# Routes appelées en boucle par l'hébergeur : mesurées mais pas journalisées (bruit).
ROUTES_SILENCIEUSES = {"/health"}


def version_aeroguard():
    """Version livrée.

    Dans l'image Docker, AEROGUARD_VERSION est injectée au build depuis le tag Git : c'est
    la seule source de vérité. En dehors de Docker, on lit la version du package installé.
    """
    injectee = os.getenv("AEROGUARD_VERSION")
    if injectee:
        return injectee
    try:
        return version("aeroguard")
    except PackageNotFoundError:
        return "inconnue"


def mode_inference():
    """Mode d'inférence effectif, avec la même priorité que obtenir_service."""
    if os.getenv("AEROGUARD_TRITON_URL"):
        return "triton"
    return os.getenv("AEROGUARD_INFERENCE", "local")


app = FastAPI(
    title="AeroGuard API",
    description="Anomaly detection for aircraft engines (NASA N-CMAPSS).",
    version=version_aeroguard(),
)

# API publique en anglais (visible dans Swagger), puis ancienne API française (cachée).
app.include_router(routeur_fleet)
app.include_router(routeur_flotte)
app.include_router(routeur_monitoring)


@app.middleware("http")
async def mesurer_requetes(request: Request, call_next):
    """Mesure chaque requête : en-tête X-Response-Time-ms et une ligne de log JSON."""
    debut = time.perf_counter()
    reponse = await call_next(request)
    duree_ms = (time.perf_counter() - debut) * 1000
    reponse.headers["X-Response-Time-ms"] = f"{duree_ms:.1f}"
    if request.url.path not in ROUTES_SILENCIEUSES:
        journal_http.info(
            "request",
            extra={
                "champs": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": reponse.status_code,
                    "duration_ms": round(duree_ms, 2),
                }
            },
        )
    return reponse


@app.get("/", include_in_schema=False)
def accueil():
    """Page d'accueil : indique où trouver la documentation interactive."""
    return {"message": "AeroGuard API", "documentation": "/docs"}


@app.get(
    "/health",
    tags=["System"],
    summary="Service health, version and inference mode",
    description="Returns the service status, the deployed version, the inference mode "
    "(local, onnx or triton) and whether the models are already loaded. "
    "Never loads the models itself.",
)
def sante():
    """Le service répond-il ? Ne charge jamais les modèles (appelé en boucle par l'hébergeur)."""
    return {
        "statut": "ok",
        "version": version_aeroguard(),
        "inference": mode_inference(),
        "modeles_charges": obtenir_service.cache_info().currsize > 0,
        "mlflow": os.getenv("MLFLOW_TRACKING_URI", "non configuré"),
    }
