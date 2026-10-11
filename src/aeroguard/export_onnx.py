"""Conversion d'un réseau Keras dense (autoencodeur) en ONNX, sans tf2onnx."""

import numpy as np

ACTIVATIONS_ONNX = {
    "relu": "Relu",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "elu": "Elu",
    "selu": "Selu",
    "leaky_relu": "LeakyRelu",
}
COUCHES_IGNOREES = {
    "InputLayer",
    "Dropout",
    "GaussianNoise",
    "GaussianDropout",
    "AlphaDropout",
    "ActivityRegularization",
}


def couches_a_plat(modele):
    """Parcourt les couches dans l'ordre, en entrant dans les sous-modèles (encodeur, décodeur)."""
    for couche in modele.layers:
        if hasattr(couche, "layers"):
            yield from couches_a_plat(couche)
        else:
            yield couche


def keras_dense_vers_onnx(modele, nom_entree="entree", nom_sortie="reconstruction", opset=17):
    """Traduit un réseau dense Keras en graphe ONNX équivalent (inférence uniquement)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    noeuds, poids = [], []
    courant = nom_entree
    compteur = 0

    def nouveau_nom(prefixe):
        nonlocal compteur
        compteur += 1
        return f"{prefixe}_{compteur}"

    def ajouter_noeud(operation, entrees, **attributs):
        nonlocal courant
        sortie = nouveau_nom(operation.lower())
        noeuds.append(helper.make_node(operation, [courant, *entrees], [sortie], **attributs))
        courant = sortie

    def ajouter_constante(tableau, prefixe):
        nom = nouveau_nom(prefixe)
        poids.append(numpy_helper.from_array(np.asarray(tableau, dtype=np.float32), nom))
        return nom

    def ajouter_activation(nom_activation, pente=0.2):
        if nom_activation == "linear":
            return
        if nom_activation not in ACTIVATIONS_ONNX:
            raise NotImplementedError(f"Activation non supportée : {nom_activation}")
        attributs = {"alpha": float(pente)} if nom_activation == "leaky_relu" else {}
        ajouter_noeud(ACTIVATIONS_ONNX[nom_activation], [], **attributs)

    for couche in couches_a_plat(modele):
        type_couche = type(couche).__name__
        if type_couche in COUCHES_IGNOREES:
            continue
        if type_couche == "Dense":
            noyau, *biais = couche.get_weights()
            ajouter_noeud("MatMul", [ajouter_constante(noyau, "poids")])
            if biais:
                ajouter_noeud("Add", [ajouter_constante(biais[0], "biais")])
            ajouter_activation(couche.activation.__name__)
        elif type_couche == "Activation":
            ajouter_activation(couche.activation.__name__)
        elif type_couche == "LeakyReLU":
            ajouter_activation("leaky_relu", getattr(couche, "negative_slope", 0.3))
        elif type_couche == "BatchNormalization":
            valeurs = list(couche.get_weights())
            gamma = valeurs.pop(0) if couche.scale else 1.0
            beta = valeurs.pop(0) if couche.center else 0.0
            moyenne, variance = valeurs
            echelle = gamma / np.sqrt(variance + couche.epsilon)
            ajouter_noeud("Mul", [ajouter_constante(echelle, "echelle")])
            ajouter_noeud("Add", [ajouter_constante(beta - moyenne * echelle, "decalage")])
        else:
            raise NotImplementedError(f"Couche non supportée : {type_couche}")

    noeuds.append(helper.make_node("Identity", [courant], [nom_sortie]))
    taille_entree = int(modele.inputs[0].shape[-1])
    taille_sortie = int(modele.outputs[0].shape[-1])
    graphe = helper.make_graph(
        noeuds,
        "autoencodeur",
        [helper.make_tensor_value_info(nom_entree, TensorProto.FLOAT, ["batch", taille_entree])],
        [helper.make_tensor_value_info(nom_sortie, TensorProto.FLOAT, ["batch", taille_sortie])],
        initializer=poids,
    )
    modele_onnx = helper.make_model(
        graphe, opset_imports=[helper.make_opsetid("", opset)], producer_name="aeroguard"
    )
    modele_onnx.ir_version = 8
    onnx.checker.check_model(modele_onnx)
    return modele_onnx
