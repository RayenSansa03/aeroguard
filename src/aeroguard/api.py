"""API AeroGuard : premier squelette. Les endpoints de prédiction arrivent au module 7."""

import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI


def version_aeroguard():
    """Version du package installé (lue dans pyproject.toml au moment de l'installation)."""
    try:
        return version("aeroguard")
    except PackageNotFoundError:
        return "inconnue"


app = FastAPI(
    title="AeroGuard API",
    description="Détection d'anomalies pour moteurs d'avion (NASA N-CMAPSS).",
    version=version_aeroguard(),
)


@app.get("/")
def accueil():
    """Page d'accueil : indique où trouver la documentation interactive."""
    return {"message": "AeroGuard API", "documentation": "/docs"}


@app.get("/health")
def sante():
    """Le service répond-il ? Utilisé par Docker (HEALTHCHECK) et plus tard par le monitoring."""
    return {
        "statut": "ok",
        "version": version_aeroguard(),
        "mlflow": os.getenv("MLFLOW_TRACKING_URI", "non configuré"),
    }
