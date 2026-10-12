"""Téléchargement des modèles depuis un dépôt Hugging Face au démarrage de l'API.

Le code et les modèles ont des cycles de vie séparés : l'image Docker ne contient aucun
modèle, elle reçoit l'adresse du dépôt et sa révision par l'environnement. Le token
d'accès est lu automatiquement dans la variable HF_TOKEN par huggingface_hub.
"""

from __future__ import annotations

from pathlib import Path

from aeroguard.derive import FICHIER_REFERENCE
from aeroguard.service import FICHIER_ONNX, FICHIERS_MODELES

FICHIERS_DEPLOIEMENT = [
    FICHIERS_MODELES["alerte"],
    FICHIER_ONNX,
    FICHIERS_MODELES["stats"],
    FICHIERS_MODELES["config"],
    FICHIER_REFERENCE,
]


def telecharger_modeles(depot: str, dossier: str | Path, revision: str | None = None) -> Path:
    """Copie dans `dossier` les fichiers du mode ONNX depuis le dépôt HF `depot`."""
    from huggingface_hub import snapshot_download

    dossier = Path(dossier)
    try:
        snapshot_download(
            repo_id=depot,
            revision=revision,
            local_dir=dossier,
            allow_patterns=FICHIERS_DEPLOIEMENT,
        )
    except OSError as erreur:
        raise FileNotFoundError(
            f"Téléchargement des modèles impossible depuis {depot} : {erreur}"
        ) from erreur
    return dossier
