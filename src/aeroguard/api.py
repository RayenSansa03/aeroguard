"""API AeroGuard : santé du service et routes de la flotte."""

import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from aeroguard.routes_flotte import routeur as routeur_flotte


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


app = FastAPI(
    title="AeroGuard API",
    description="Détection d'anomalies pour moteurs d'avion (NASA N-CMAPSS).",
    version=version_aeroguard(),
)

app.include_router(routeur_flotte)


@app.get("/")
def accueil():
    """Page d'accueil : indique où trouver la documentation interactive."""
    return {"message": "AeroGuard API", "documentation": "/docs"}


@app.get("/health")
def sante():
    """Le service répond-il ? Utilisé par Docker (HEALTHCHECK) et par le monitoring."""
    return {
        "statut": "ok",
        "version": version_aeroguard(),
        "mlflow": os.getenv("MLFLOW_TRACKING_URI", "non configuré"),
    }
