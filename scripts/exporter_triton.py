"""Construit le dépôt de modèles Triton à partir des modèles champions de local/models."""

import argparse
from pathlib import Path

import numpy as np

from aeroguard.export_onnx import keras_dense_vers_onnx
from aeroguard.triton import ecrire_modele_fil, ecrire_modele_onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeles", default="local/models")
    parser.add_argument("--depot", default="local/triton")
    args = parser.parse_args()
    modeles, depot = Path(args.modeles), Path(args.depot)

    import onnxruntime as ort
    from tensorflow import keras

    stats = np.load(modeles / "autoencodeur_centre_v5_stats.npz", allow_pickle=True)
    nombre_features = len(stats["colonnes"])

    ecrire_modele_fil(depot, "xgb_alerte", modeles / "xgb_binaire_v3.json", nombre_features)

    autoencodeur = keras.models.load_model(str(modeles / "autoencodeur_centre_v5.keras"))
    modele_onnx = keras_dense_vers_onnx(autoencodeur)
    ecrire_modele_onnx(depot, "autoencodeur", modele_onnx, nombre_features)

    # Parité Keras / ONNX sur des vols standardisés fictifs.
    x = np.random.default_rng(0).normal(size=(512, nombre_features)).astype(np.float32)
    attendu = autoencodeur.predict(x, verbose=0)
    session = ort.InferenceSession(
        modele_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    obtenu = session.run(None, {"entree": x})[0]
    print(f"{nombre_features} features")
    print(f"Écart max Keras / ONNX : {np.abs(obtenu - attendu).max():.2e}")
    for fichier in sorted(depot.rglob("*")):
        if fichier.is_file():
            print(f"  {fichier.relative_to(depot)}  ({fichier.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
