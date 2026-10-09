import numpy as np
import pytest

from aeroguard.autoencodeurs import creer_autoencodeur_conv1d, erreur_reconstruction_fenetres


class ModeleDeuxCanaux:
    """Faux modèle : reconstruit seulement les 2 premiers canaux, avec un décalage de 1."""

    def predict(self, X, batch_size=None, verbose=0):
        return np.asarray(X)[:, :, :2] + 1


def test_erreur_fenetres_sur_les_canaux_reconstruits():
    X = np.zeros((3, 8, 5))
    assert np.allclose(erreur_reconstruction_fenetres(ModeleDeuxCanaux(), X), 1.0)
    par_canal = erreur_reconstruction_fenetres(ModeleDeuxCanaux(), X, par_canal=True)
    assert par_canal.shape == (3, 2)


def test_architecture_et_parametres():
    pytest.importorskip("tensorflow")
    from aeroguard.reseaux import nombre_parametres_entrainables

    modele = creer_autoencodeur_conv1d(256, 18, canaux_sortie=14)
    attendu = (7 * 18 * 32 + 32) + (7 * 32 * 16 + 16) + (7 * 16 * 8 + 8)  # encodeur
    attendu += (7 * 8 * 8 + 8) + (7 * 8 * 16 + 16) + (7 * 16 * 32 + 32)  # décodeur
    attendu += 7 * 32 * 14 + 14  # reconstruction des 14 capteurs
    assert nombre_parametres_entrainables(modele) == attendu == 16_702
    assert modele.get_layer("compression_3").output.shape[1:] == (32, 8)  # le goulot


def test_forme_de_sortie():
    pytest.importorskip("tensorflow")
    modele = creer_autoencodeur_conv1d(16, 3, canaux_sortie=2, filtres=(4,))
    X = np.random.default_rng(0).normal(size=(5, 16, 3)).astype("float32")
    assert modele.predict(X, verbose=0).shape == (5, 16, 2)


@pytest.mark.parametrize(
    "longueur, canaux, sortie, filtres",
    [(100, 18, 14, (32, 16, 8)), (256, 18, 20, (32,)), (256, 18, 0, (32,)), (256, 18, 14, ())],
)
def test_parametres_invalides(longueur, canaux, sortie, filtres):
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError):
        creer_autoencodeur_conv1d(longueur, canaux, canaux_sortie=sortie, filtres=filtres)


def test_apprend_a_reconstruire_des_sinusoides():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(1)
    temps = np.linspace(0, 2 * np.pi, 16)
    phases = rng.uniform(0, 2 * np.pi, size=(200, 1))
    X = np.stack([np.sin(temps + phases), np.cos(temps + phases)], axis=-1).astype("float32")
    modele = creer_autoencodeur_conv1d(16, 2, filtres=(8,), taille_noyau=3)
    avant = erreur_reconstruction_fenetres(modele, X).mean()
    modele.fit(X, X, epochs=80, batch_size=32, verbose=0)
    assert erreur_reconstruction_fenetres(modele, X).mean() < avant / 2
