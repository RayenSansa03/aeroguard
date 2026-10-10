"""GAN tabulaire : générer de faux vols pour renforcer les familles rares (module 6)."""

import numpy as np

DIM_BRUIT = 32


def creer_generateur(nombre_features, dim_bruit=DIM_BRUIT, couches_cachees=(64, 128), graine=42):
    """Générateur (le faussaire) : bruit aléatoire → faux vol en features standardisées."""
    from tensorflow import keras

    if nombre_features < 1 or dim_bruit < 1:
        raise ValueError("nombre_features et dim_bruit doivent être supérieurs ou égaux à 1.")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(dim_bruit,), name="bruit")]
    for numero, neurones in enumerate(couches_cachees, start=1):
        couches.append(keras.layers.Dense(neurones, name=f"cachee_{numero}"))
        couches.append(keras.layers.LeakyReLU(negative_slope=0.2, name=f"activation_{numero}"))
    # Sortie linéaire : les features standardisées peuvent être négatives ou dépasser 1
    couches.append(keras.layers.Dense(nombre_features, name="faux_vol"))
    return keras.Sequential(couches, name="generateur")


def creer_discriminateur(nombre_features, couches_cachees=(128, 64), taux_dropout=0.3, graine=42):
    """Discriminateur (l'expert) : vol → logit « vrai » (sans sigmoïde, perte from_logits)."""
    from tensorflow import keras

    if nombre_features < 1:
        raise ValueError("nombre_features doit être supérieur ou égal à 1.")
    if not 0 <= taux_dropout < 1:
        raise ValueError("taux_dropout doit être compris entre 0 (inclus) et 1 (exclu).")
    keras.utils.set_random_seed(graine)

    couches = [keras.Input(shape=(nombre_features,), name="vol")]
    for numero, neurones in enumerate(couches_cachees, start=1):
        couches.append(keras.layers.Dense(neurones, name=f"cachee_{numero}"))
        couches.append(keras.layers.LeakyReLU(negative_slope=0.2, name=f"activation_{numero}"))
        if taux_dropout > 0:
            couches.append(keras.layers.Dropout(taux_dropout, name=f"dropout_{numero}"))
    couches.append(keras.layers.Dense(1, name="logit_vrai"))
    return keras.Sequential(couches, name="discriminateur")


def creer_etape_entrainement(
    generateur, discriminateur, dim_bruit=DIM_BRUIT, taux_apprentissage=2e-4
):
    """Renvoie une étape d'entraînement : 1 pas pour l'expert, puis 1 pas pour le faussaire."""
    import tensorflow as tf
    from tensorflow import keras

    perte_bce = keras.losses.BinaryCrossentropy(from_logits=True)
    optimiseur_d = keras.optimizers.Adam(learning_rate=taux_apprentissage, beta_1=0.5)
    optimiseur_g = keras.optimizers.Adam(learning_rate=taux_apprentissage, beta_1=0.5)

    @tf.function
    def etape(vrais):
        n = tf.shape(vrais)[0]

        # 1) L'expert apprend : vrais vols → 1, faux vols → 0
        bruit = tf.random.normal((n, dim_bruit))
        with tf.GradientTape() as bande_d:
            faux = generateur(bruit, training=True)
            logits_vrais = discriminateur(vrais, training=True)
            logits_faux = discriminateur(faux, training=True)
            perte_d = perte_bce(tf.ones_like(logits_vrais), logits_vrais) + perte_bce(
                tf.zeros_like(logits_faux), logits_faux
            )
        gradients_d = bande_d.gradient(perte_d, discriminateur.trainable_variables)
        optimiseur_d.apply_gradients(
            zip(gradients_d, discriminateur.trainable_variables, strict=True)
        )

        # 2) Le faussaire apprend : il veut que l'expert réponde 1 à ses faux vols
        bruit = tf.random.normal((n, dim_bruit))
        with tf.GradientTape() as bande_g:
            faux = generateur(bruit, training=True)
            logits_faux = discriminateur(faux, training=True)
            perte_g = perte_bce(tf.ones_like(logits_faux), logits_faux)
        gradients_g = bande_g.gradient(perte_g, generateur.trainable_variables)
        optimiseur_g.apply_gradients(zip(gradients_g, generateur.trainable_variables, strict=True))

        proba_vrais = tf.reduce_mean(tf.sigmoid(logits_vrais))
        proba_faux = tf.reduce_mean(tf.sigmoid(logits_faux))
        return perte_d, perte_g, proba_vrais, proba_faux

    return etape


def entrainer_gan(
    generateur,
    discriminateur,
    X,
    epochs=1000,
    taille_lot=64,
    dim_bruit=DIM_BRUIT,
    taux_apprentissage=2e-4,
    graine=42,
    afficher_tous_les=100,
):
    """Entraîne le GAN sur X (vrais vols standardisés). Renvoie l'historique par epoch."""
    import tensorflow as tf

    X = np.asarray(X, dtype="float32")
    if X.ndim != 2:
        raise ValueError("X doit être un tableau à 2 dimensions (vols × features).")
    if np.isnan(X).any():
        raise ValueError("X contient des NaN : imputer les valeurs manquantes avant le GAN.")
    if not 1 <= taille_lot <= len(X):
        raise ValueError("taille_lot doit être entre 1 et le nombre de vols de X.")
    if epochs < 1:
        raise ValueError("epochs doit être supérieur ou égal à 1.")

    tf.random.set_seed(graine)
    etape = creer_etape_entrainement(generateur, discriminateur, dim_bruit, taux_apprentissage)
    jeu = (
        tf.data.Dataset.from_tensor_slices(X)
        .shuffle(len(X), seed=graine, reshuffle_each_iteration=True)
        .batch(taille_lot, drop_remainder=True)
    )

    historique = {"perte_d": [], "perte_g": [], "proba_vrais": [], "proba_faux": []}
    for epoch in range(1, epochs + 1):
        valeurs = np.array([[float(v) for v in etape(lot)] for lot in jeu])
        for cle, moyenne in zip(historique, valeurs.mean(axis=0), strict=True):
            historique[cle].append(float(moyenne))
        if afficher_tous_les and epoch % afficher_tous_les == 0:
            print(
                f"epoch {epoch:5d} | perte D {historique['perte_d'][-1]:.3f} | "
                f"perte G {historique['perte_g'][-1]:.3f} | "
                f"D(vrais) {historique['proba_vrais'][-1]:.2f} | "
                f"D(faux) {historique['proba_faux'][-1]:.2f}"
            )
    return historique


def generer(generateur, n, dim_bruit=DIM_BRUIT, graine=0):
    """Génère n faux vols (features standardisées). Même graine → mêmes vols."""
    if n < 1:
        raise ValueError("n doit être supérieur ou égal à 1.")
    bruit = np.random.default_rng(graine).standard_normal((n, dim_bruit)).astype("float32")
    return np.asarray(generateur.predict(bruit, verbose=0))
