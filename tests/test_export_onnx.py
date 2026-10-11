"""Tests de la conversion Keras vers ONNX : même sortie que Keras, à l'arrondi près."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aeroguard.export_onnx import keras_dense_vers_onnx  # noqa: E402

X = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)


def sortie_onnx(modele_onnx, x):
    session = ort.InferenceSession(
        modele_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, {"entree": x})[0]


def verifier_parite(modele):
    attendu = modele.predict(X, verbose=0)
    obtenu = sortie_onnx(keras_dense_vers_onnx(modele), X)
    np.testing.assert_allclose(obtenu, attendu, atol=1e-5)


def test_dense_relu_dropout_lineaire():
    keras.utils.set_random_seed(0)
    modele = keras.Sequential(
        [
            keras.Input((6,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(6),
        ]
    )
    verifier_parite(modele)


def test_batchnorm_leakyrelu_et_activations():
    keras.utils.set_random_seed(1)
    normalisation = keras.layers.BatchNormalization()
    modele = keras.Sequential(
        [
            keras.Input((6,)),
            keras.layers.Dense(5, activation="tanh"),
            normalisation,
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dense(6, activation="sigmoid"),
            keras.layers.Activation("elu"),
        ]
    )
    rng = np.random.default_rng(2)
    normalisation.set_weights(
        [rng.uniform(0.5, 2, 5), rng.normal(size=5), rng.normal(size=5), rng.uniform(0.5, 2, 5)]
    )
    verifier_parite(modele)


def test_sous_modeles_encodeur_decodeur():
    keras.utils.set_random_seed(3)
    encodeur = keras.Sequential([keras.Input((6,)), keras.layers.Dense(2, activation="selu")])
    decodeur = keras.Sequential([keras.Input((2,)), keras.layers.Dense(6)])
    entree = keras.Input((6,))
    modele = keras.Model(entree, decodeur(encodeur(entree)))
    verifier_parite(modele)


def test_couche_non_supportee_refusee():
    modele = keras.Sequential(
        [keras.Input((6,)), keras.layers.LayerNormalization(), keras.layers.Dense(6)]
    )
    with pytest.raises(NotImplementedError, match="LayerNormalization"):
        keras_dense_vers_onnx(modele)


def test_activation_non_supportee_refusee():
    modele = keras.Sequential([keras.Input((6,)), keras.layers.Dense(6, activation="softplus")])
    with pytest.raises(NotImplementedError, match="softplus"):
        keras_dense_vers_onnx(modele)
