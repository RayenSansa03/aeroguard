"""Tests du service en mode Triton (faux client Triton, sans Docker)."""

import json

import numpy as np
import pytest

from aeroguard.service import charger_service_triton

COLONNES = ["T48_moy", "T50_moy"]


class FauxTriton:
    """Faux Triton : proba fixe, reconstruction nulle (l'erreur vaut donc xn²)."""

    def __init__(self, prets=("xgb_alerte", "autoencodeur")):
        self.prets = prets
        self.recus = []

    def pret(self, modele):
        return modele in self.prets

    def probas_usure(self, X):
        self.recus.append(np.array(X))
        return np.full(len(X), 0.2, dtype=np.float32)

    def reconstruction(self, Xn):
        return np.zeros_like(Xn)


@pytest.fixture
def dossier(tmp_path):
    np.savez(
        tmp_path / "autoencodeur_centre_v5_stats.npz",
        colonnes=np.array(COLONNES),
        moyennes=np.array([0.0, 0.0]),
        ecarts=np.array([2.0, 1.0]),
        medianes_imputation=np.array([5.0, 5.0]),
    )
    config = {"seuil": 0.618, "k": 5, "n_premiers": 2}
    (tmp_path / "autoencodeur_centre_v5_alerte.json").write_text(json.dumps(config))
    return tmp_path


def test_mode_triton_calcule_comme_le_mode_local(dossier):
    triton = FauxTriton()
    service = charger_service_triton(dossier, "http://triton", client=triton)
    service.traiter_vol("M1", 1, {"T48_moy": 1.0, "T50_moy": 1.0})
    service.traiter_vol("M1", 2, {"T48_moy": 1.0, "T50_moy": 1.0})
    reco = service.traiter_vol("M1", 3, {"T48_moy": 5.0, "T50_moy": 3.0})
    # Centrage : (5 - 1, 3 - 1) = (4, 2) ; standardisation : (4 / 2, 2 / 1) = (2, 2) ; erreur = 4.
    assert reco["erreur_reconstruction"] == pytest.approx(4.0)
    assert reco["proba_usure"] == pytest.approx(0.2)


def test_les_nan_partent_vers_triton_sans_imputation(dossier):
    triton = FauxTriton()
    service = charger_service_triton(dossier, "http://triton", client=triton)
    service.traiter_vol("M1", 1, {"T48_moy": 1.0})
    assert np.isnan(triton.recus[0][0, 1])


def test_triton_pas_pret_leve_une_erreur_de_connexion(dossier):
    with pytest.raises(ConnectionError, match="autoencodeur"):
        charger_service_triton(dossier, "http://triton", client=FauxTriton(prets=("xgb_alerte",)))


def test_fichiers_manquants(tmp_path):
    with pytest.raises(FileNotFoundError, match="stats"):
        charger_service_triton(tmp_path, "http://triton", client=FauxTriton())


def test_client_triton_cree_par_defaut(dossier):
    with pytest.raises(ConnectionError):
        charger_service_triton(dossier, "http://127.0.0.1:9")
