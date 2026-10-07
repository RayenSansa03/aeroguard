import numpy as np
import pytest

pytest.importorskip("tensorflow")

from aeroguard.reseaux import (  # noqa: E402
    creer_normaliseur,
    creer_reseau_dense,
    nombre_parametres_entrainables,
)


def test_sortie_est_une_probabilite():
    modele = creer_reseau_dense(5)
    X = np.random.default_rng(0).normal(size=(10, 5)).astype("float32")
    probas = modele.predict(X, verbose=0)
    assert probas.shape == (10, 1)
    assert np.all((probas >= 0) & (probas <= 1))


def test_nombre_de_parametres_pour_126_features():
    modele = creer_reseau_dense(126)
    attendu = (126 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
    assert nombre_parametres_entrainables(modele) == attendu == 10_241


def test_architecture_personnalisee():
    modele = creer_reseau_dense(4, couches_cachees=(8,))
    assert nombre_parametres_entrainables(modele) == (4 * 8 + 8) + (8 + 1)
    assert [couche.name for couche in modele.layers] == ["cachee_1", "proba_usure"]


def test_normaliseur_centre_et_reduit():
    X = np.random.default_rng(1).normal(loc=50, scale=10, size=(500, 3)).astype("float32")
    sortie = np.asarray(creer_normaliseur(X)(X))
    assert np.allclose(sortie.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(sortie.std(axis=0), 1, atol=1e-2)


def test_normaliseur_refuse_les_nan():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        creer_normaliseur(X)


def test_nombre_features_invalide():
    with pytest.raises(ValueError):
        creer_reseau_dense(0)


def test_meme_graine_memes_predictions():
    X = np.ones((2, 4), dtype="float32")
    probas_1 = creer_reseau_dense(4, graine=7).predict(X, verbose=0)
    probas_2 = creer_reseau_dense(4, graine=7).predict(X, verbose=0)
    assert np.allclose(probas_1, probas_2)


def test_le_reseau_apprend():
    generateur = np.random.default_rng(0)
    X = generateur.normal(size=(400, 2)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype("float32")
    modele = creer_reseau_dense(2, couches_cachees=(8,), taux_apprentissage=0.05)
    historique = modele.fit(X, y, epochs=20, batch_size=32, verbose=0)
    assert historique.history["loss"][-1] < historique.history["loss"][0]


def test_reseau_avec_normaliseur():
    X = np.random.default_rng(2).normal(loc=100, scale=20, size=(50, 3)).astype("float32")
    modele = creer_reseau_dense(3, normaliseur=creer_normaliseur(X))
    assert modele.layers[0].name == "normalisation"
    # Le normaliseur n'ajoute aucun paramètre entraînable
    assert nombre_parametres_entrainables(modele) == (3 * 64 + 64) + (64 * 32 + 32) + (32 + 1)
    assert modele.predict(X, verbose=0).shape == (50, 1)
