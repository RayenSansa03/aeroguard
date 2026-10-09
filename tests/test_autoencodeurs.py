import numpy as np
import pytest

from aeroguard.autoencodeurs import (
    creer_autoencodeur_dense,
    erreur_reconstruction,
    standardiser,
    statistiques_features,
)


class ModeleDecale:
    """Faux modèle : sa « reconstruction » vaut l'entrée + 1 (erreur connue d'avance)."""

    def predict(self, X, verbose=0):
        return np.asarray(X) + 1


def test_statistiques_et_standardisation():
    X = np.random.default_rng(0).normal(loc=20, scale=5, size=(400, 3))
    moyennes, ecarts = statistiques_features(X)
    Xn = standardiser(X, moyennes, ecarts)
    assert Xn.dtype == np.float32
    assert np.allclose(Xn.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(Xn.std(axis=0), 1, atol=1e-5)


def test_feature_constante_sans_division_par_zero():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    moyennes, ecarts = statistiques_features(X)
    assert ecarts[0] == 1.0
    assert np.allclose(standardiser(X, moyennes, ecarts)[:, 0], 0)


def test_statistiques_refusent_les_nan():
    with pytest.raises(ValueError):
        statistiques_features([[1.0, np.nan], [2.0, 3.0]])


def test_erreur_reconstruction_par_vol_et_par_feature():
    X = np.zeros((4, 3))
    assert np.allclose(erreur_reconstruction(ModeleDecale(), X), 1.0)
    par_feature = erreur_reconstruction(ModeleDecale(), X, par_feature=True)
    assert par_feature.shape == (4, 3)
    assert np.allclose(par_feature, 1.0)


def test_architecture_et_parametres():
    pytest.importorskip("tensorflow")
    from aeroguard.reseaux import nombre_parametres_entrainables

    modele = creer_autoencodeur_dense(126)
    attendu = (126 * 64 + 64) + (64 * 16 + 16) + (16 * 8 + 8)  # encodeur + goulot
    attendu += (8 * 16 + 16) + (16 * 64 + 64) + (64 * 126 + 126)  # décodeur + sortie
    assert nombre_parametres_entrainables(modele) == attendu == 18_726
    assert [c.name for c in modele.layers] == [
        "encodeur_1",
        "encodeur_2",
        "goulot",
        "decodeur_1",
        "decodeur_2",
        "reconstruction",
    ]


def test_sortie_de_meme_forme_que_l_entree():
    pytest.importorskip("tensorflow")
    modele = creer_autoencodeur_dense(10, couches_encodeur=(6,), taille_goulot=3)
    X = np.random.default_rng(1).normal(size=(5, 10)).astype("float32")
    assert modele.predict(X, verbose=0).shape == (5, 10)


@pytest.mark.parametrize("features, goulot", [(0, 1), (10, 10), (10, 0)])
def test_parametres_invalides(features, goulot):
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError):
        creer_autoencodeur_dense(features, taille_goulot=goulot)


def test_apprend_a_reconstruire():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(2)
    cache = rng.normal(size=(300, 2))
    X = (cache @ rng.normal(size=(2, 6))).astype("float32")  # 6 features, 2 vraies causes
    modele = creer_autoencodeur_dense(6, couches_encodeur=(8,), taille_goulot=2)
    avant = erreur_reconstruction(modele, X).mean()
    modele.fit(X, X, epochs=100, batch_size=32, verbose=0)
    apres = erreur_reconstruction(modele, X).mean()
    assert apres < avant / 2
