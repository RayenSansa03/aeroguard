"""Réseaux de neurones Keras pour AeroGuard (module 4)."""

import numpy as np

COUCHES_DENSES = (64, 32)


def creer_normaliseur(X):
    """Couche Keras qui centre-réduit chaque feature avec les statistiques de X (le train)."""
    from tensorflow import keras

    valeurs = np.asarray(X, dtype="float32")
    if np.isnan(valeurs).any():
        raise ValueError("X contient des NaN : imputer les valeurs manquantes avant de normaliser.")
    normaliseur = keras.layers.Normalization(name="normalisation")
    normaliseur.adapt(valeurs)
    return normaliseur


def creer_reseau_dense(
    nombre_features,
    couches_cachees=COUCHES_DENSES,
    taux_apprentissage=1e-3,
    normaliseur=None,
    graine=42,
):
    """Réseau dense d'alerte : features → couches ReLU → probabilité d'usure (sigmoïde)."""
    from tensorflow import keras

    if nombre_features < 1:
        raise ValueError("nombre_features doit être supérieur ou égal à 1.")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(nombre_features,), name="features_vol")]
    if normaliseur is not None:
        couches.append(normaliseur)
    for numero, neurones in enumerate(couches_cachees, start=1):
        couches.append(keras.layers.Dense(neurones, activation="relu", name=f"cachee_{numero}"))
    couches.append(keras.layers.Dense(1, activation="sigmoid", name="proba_usure"))

    modele = keras.Sequential(couches, name="aeroguard_dense")
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taux_apprentissage),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return modele


def nombre_parametres_entrainables(modele):
    """Nombre de poids et de biais que l'entraînement modifie."""
    return int(sum(np.prod(poids.shape) for poids in modele.trainable_weights))
