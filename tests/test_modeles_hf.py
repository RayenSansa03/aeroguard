"""Téléchargement des modèles depuis Hugging Face, sans accès réseau."""

import huggingface_hub
import pytest

from aeroguard import modeles_hf, routes_flotte, service
from aeroguard.modeles_hf import FICHIERS_DEPLOIEMENT, telecharger_modeles


def test_telecharge_seulement_les_fichiers_du_mode_onnx(monkeypatch, tmp_path):
    appels = {}

    def faux_telechargement(**kwargs):
        appels.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", faux_telechargement)
    assert telecharger_modeles("rayen/aeroguard-models", tmp_path, revision="v5") == tmp_path
    assert appels["repo_id"] == "rayen/aeroguard-models"
    assert appels["revision"] == "v5"
    assert sorted(appels["allow_patterns"]) == sorted(FICHIERS_DEPLOIEMENT)
    assert "autoencodeur_centre_v5.keras" not in appels["allow_patterns"]


def test_echec_reseau_devient_fichier_introuvable(monkeypatch, tmp_path):
    def reseau_coupe(**kwargs):
        raise OSError("réseau coupé")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", reseau_coupe)
    with pytest.raises(FileNotFoundError, match="impossible"):
        telecharger_modeles("rayen/aeroguard-models", tmp_path)


def test_api_telecharge_avant_de_charger(monkeypatch):
    depots = []
    monkeypatch.delenv("AEROGUARD_TRITON_URL", raising=False)
    monkeypatch.setenv("AEROGUARD_INFERENCE", "onnx")
    monkeypatch.setenv("AEROGUARD_HF_DEPOT", "rayen/aeroguard-models")

    def faux_telechargement(depot, dossier, revision=None):
        depots.append(depot)

    monkeypatch.setattr(modeles_hf, "telecharger_modeles", faux_telechargement)
    monkeypatch.setattr(service, "charger_service_onnx", lambda dossier: "service-onnx")
    routes_flotte.obtenir_service.cache_clear()
    try:
        assert routes_flotte.obtenir_service() == "service-onnx"
    finally:
        routes_flotte.obtenir_service.cache_clear()
    assert depots == ["rayen/aeroguard-models"]
