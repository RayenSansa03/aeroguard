"""Autoencodeurs : apprendre le « normal » pour repérer tout ce qui s'en écarte (module 5)."""

import numpy as np

COUCHES_ENCODEUR = (64, 16)


def statistiques_features(X):
    """Moyenne et écart-type de chaque feature (un écart nul devient 1 pour éviter /0)."""
    valeurs = np.asarray(X, dtype="float64")
    if np.isnan(valeurs).any():
        raise ValueError("X contient des NaN : imputer les valeurs manquantes avant.")
    moyennes = valeurs.mean(axis=0)
    ecarts = valeurs.std(axis=0)
    ecarts[ecarts == 0] = 1.0
    return moyennes, ecarts


def standardiser(X, moyennes, ecarts):
    """Centre-réduit chaque feature avec les statistiques de référence (vols sains du train)."""
    return ((np.asarray(X, dtype="float64") - moyennes) / ecarts).astype("float32")


def creer_autoencodeur_dense(
    nombre_features,
    couches_encodeur=COUCHES_ENCODEUR,
    taille_goulot=8,
    taux_apprentissage=1e-3,
    graine=42,
):
    """Autoencodeur : features → encodeur → goulot → décodeur (miroir) → features reconstruites."""
    from tensorflow import keras

    if nombre_features < 1:
        raise ValueError("nombre_features doit être supérieur ou égal à 1.")
    if not 1 <= taille_goulot < nombre_features:
        raise ValueError("Le goulot doit être plus petit que le nombre de features.")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(nombre_features,), name="vol_normalise")]
    for numero, neurones in enumerate(couches_encodeur, start=1):
        couches.append(keras.layers.Dense(neurones, activation="relu", name=f"encodeur_{numero}"))
    couches.append(keras.layers.Dense(taille_goulot, name="goulot"))
    for numero, neurones in enumerate(reversed(couches_encodeur), start=1):
        couches.append(keras.layers.Dense(neurones, activation="relu", name=f"decodeur_{numero}"))
    couches.append(keras.layers.Dense(nombre_features, name="reconstruction"))

    modele = keras.Sequential(couches, name="aeroguard_autoencodeur")
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=taux_apprentissage), loss="mse")
    return modele


def erreur_reconstruction(modele, X, par_feature=False):
    """Erreur quadratique entre chaque vol et sa reconstruction.

    par_feature=False → une erreur moyenne par vol (le score d'anomalie) ;
    par_feature=True → une erreur par vol ET par feature (quel capteur surprend le modèle).
    """
    X = np.asarray(X, dtype="float32")
    reconstruction = np.asarray(modele.predict(X, verbose=0))
    erreurs = (X - reconstruction) ** 2
    return erreurs if par_feature else erreurs.mean(axis=1)
