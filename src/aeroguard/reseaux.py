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
    taux_dropout=0.0,
    graine=42,
):
    """Réseau dense d'alerte : features → couches ReLU (+ dropout) → probabilité d'usure."""
    from tensorflow import keras

    if nombre_features < 1:
        raise ValueError("nombre_features doit être supérieur ou égal à 1.")
    if not 0 <= taux_dropout < 1:
        raise ValueError("taux_dropout doit être compris entre 0 (inclus) et 1 (exclu).")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(nombre_features,), name="features_vol")]
    if normaliseur is not None:
        couches.append(normaliseur)
    for numero, neurones in enumerate(couches_cachees, start=1):
        couches.append(keras.layers.Dense(neurones, activation="relu", name=f"cachee_{numero}"))
        if taux_dropout > 0:
            couches.append(keras.layers.Dropout(taux_dropout, name=f"dropout_{numero}"))
    couches.append(keras.layers.Dense(1, activation="sigmoid", name="proba_usure"))

    modele = keras.Sequential(couches, name="aeroguard_dense")
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taux_apprentissage),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return modele


def creer_callbacks(
    chemin_sauvegarde=None,
    patience=10,
    patience_taux=5,
    facteur_taux=0.5,
    taux_minimum=1e-5,
):
    """Callbacks contre le surapprentissage : arrêt anticipé, baisse du taux, sauvegarde."""
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=facteur_taux, patience=patience_taux, min_lr=taux_minimum
        ),
    ]
    if chemin_sauvegarde is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(chemin_sauvegarde), monitor="val_loss", save_best_only=True
            )
        )
    return callbacks


def creer_cnn_1d(
    longueur,
    nombre_canaux,
    filtres=(32, 64, 64),
    taille_noyau=7,
    taux_dropout=0.3,
    taux_apprentissage=1e-3,
    graine=42,
):
    """1D-CNN d'alerte : fenêtre (temps, canaux) → convolutions → probabilité d'usure."""
    from tensorflow import keras

    if longueur < 1 or nombre_canaux < 1:
        raise ValueError("longueur et nombre_canaux doivent être supérieurs ou égaux à 1.")
    if not filtres:
        raise ValueError("filtres doit contenir au moins une couche.")
    if taille_noyau < 1:
        raise ValueError("taille_noyau doit être supérieure ou égale à 1.")
    if not 0 <= taux_dropout < 1:
        raise ValueError("taux_dropout doit être compris entre 0 (inclus) et 1 (exclu).")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(longueur, nombre_canaux), name="fenetre_vol")]
    for numero, nombre_filtres in enumerate(filtres, start=1):
        couches.append(
            keras.layers.Conv1D(
                nombre_filtres,
                taille_noyau,
                padding="same",
                activation="relu",
                name=f"conv_{numero}",
            )
        )
        if numero < len(filtres):
            couches.append(keras.layers.MaxPooling1D(2, name=f"pool_{numero}"))
    couches.append(keras.layers.GlobalAveragePooling1D(name="moyenne_globale"))
    if taux_dropout > 0:
        couches.append(keras.layers.Dropout(taux_dropout, name="dropout"))
    couches.append(keras.layers.Dense(1, activation="sigmoid", name="proba_usure"))

    modele = keras.Sequential(couches, name="aeroguard_cnn_1d")
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taux_apprentissage),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return modele


def nombre_parametres_entrainables(modele):
    """Nombre de poids et de biais que l'entraînement modifie."""
    return int(sum(np.prod(poids.shape) for poids in modele.trainable_weights))
