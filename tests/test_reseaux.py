import numpy as np
import pytest

pytest.importorskip("tensorflow")

from aeroguard.reseaux import (
    creer_callbacks,
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


def test_dropout_ajoute_des_couches_sans_parametres():
    sans = creer_reseau_dense(4, couches_cachees=(8,))
    avec = creer_reseau_dense(4, couches_cachees=(8,), taux_dropout=0.5)
    assert [couche.name for couche in avec.layers] == ["cachee_1", "dropout_1", "proba_usure"]
    assert nombre_parametres_entrainables(avec) == nombre_parametres_entrainables(sans)


@pytest.mark.parametrize("taux", [-0.1, 1.0])
def test_dropout_invalide(taux):
    with pytest.raises(ValueError, match="taux_dropout"):
        creer_reseau_dense(4, taux_dropout=taux)


def test_dropout_actif_seulement_pendant_entrainement():
    modele = creer_reseau_dense(6, taux_dropout=0.5)
    X = np.ones((20, 6), dtype="float32")
    prediction_1 = np.asarray(modele(X, training=False))
    prediction_2 = np.asarray(modele(X, training=False))
    entrainement = np.asarray(modele(X, training=True))
    assert np.allclose(prediction_1, prediction_2)
    assert not np.allclose(prediction_1, entrainement)


def test_callbacks_sans_sauvegarde():
    from tensorflow import keras

    callbacks = creer_callbacks(patience=7)
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], keras.callbacks.EarlyStopping)
    assert callbacks[0].patience == 7
    assert callbacks[0].restore_best_weights
    assert isinstance(callbacks[1], keras.callbacks.ReduceLROnPlateau)


def test_arret_anticipe_et_sauvegarde(tmp_path):
    generateur = np.random.default_rng(0)
    X = generateur.normal(size=(200, 10)).astype("float32")
    y = generateur.integers(0, 2, size=200).astype("float32")  # étiquettes au hasard :
    X_val = generateur.normal(size=(100, 10)).astype("float32")  # rien de général à apprendre,
    y_val = generateur.integers(0, 2, size=100).astype("float32")  # donc surapprentissage rapide
    chemin = tmp_path / "meilleur.keras"

    modele = creer_reseau_dense(10, taux_apprentissage=0.01)
    historique = modele.fit(
        X,
        y,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=creer_callbacks(chemin, patience=3),
    )
    assert len(historique.history["loss"]) < 100
    assert chemin.exists()
