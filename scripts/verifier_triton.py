"""Compare Triton aux modèles locaux sur tout le jeu de test, puis mesure la latence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.client_triton import ClientTriton
from aeroguard.decision import SEUIL_XGB
from aeroguard.donnees_ml import preparer_xy


def par_lots(fonction, X, taille=1024):
    """Découpe en lots de 1024 vols (le max_batch_size déclaré dans config.pbtxt)."""
    return np.concatenate([fonction(X[i : i + taille]) for i in range(0, len(X), taille)])


def chronometrer(fonction, repetitions):
    """Durée moyenne d'un appel, en millisecondes."""
    fonction()
    debut = time.perf_counter()
    for _ in range(repetitions):
        fonction()
    return (time.perf_counter() - debut) / repetitions * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeles", default="local/models")
    parser.add_argument("--vols", default="local/data/vols_ncmapss.parquet")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    import xgboost
    from tensorflow import keras

    from aeroguard.autoencodeurs import standardiser
    from aeroguard.hybride import centrer_par_moteur

    client = ClientTriton(args.url)
    for nom in ("xgb_alerte", "autoencodeur"):
        if not client.pret(nom):
            raise SystemExit(f"Triton n'est pas prêt pour {nom} : lance d'abord le conteneur.")

    modeles = Path(args.modeles)
    stats = np.load(modeles / "autoencodeur_centre_v5_stats.npz", allow_pickle=True)
    config = json.loads(
        (modeles / "autoencodeur_centre_v5_alerte.json").read_text(encoding="utf-8")
    )
    colonnes = [str(c) for c in stats["colonnes"]]

    vols = pd.read_parquet(args.vols)
    X_test, _ = preparer_xy(vols, "test")
    X_test = X_test[colonnes]
    infos = vols.loc[X_test.index, ["moteur", "cycle"]]
    X_brut = X_test.to_numpy(dtype="float64")
    X_impute = np.where(np.isnan(X_brut), stats["medianes_imputation"], X_brut)
    X_centre = centrer_par_moteur(
        X_impute,
        infos["moteur"].to_numpy(),
        infos["cycle"].to_numpy(),
        n_premiers=config["n_premiers"],
    )
    Xn = standardiser(X_centre, stats["moyennes"], stats["ecarts"]).astype(np.float32)

    xgb = xgboost.XGBClassifier()
    xgb.load_model(str(modeles / "xgb_binaire_v3.json"))
    xgb.set_params(device="cpu")
    autoencodeur = keras.models.load_model(str(modeles / "autoencodeur_centre_v5.keras"))

    probas_local = xgb.predict_proba(X_test)[:, 1]
    probas_triton = par_lots(client.probas_usure, X_brut)
    reconstruction_locale = autoencodeur.predict(Xn, verbose=0)
    reconstruction_triton = par_lots(client.reconstruction, Xn)

    ecart_proba = np.abs(probas_local - probas_triton).max()
    bascules = int(((probas_local >= SEUIL_XGB) != (probas_triton >= SEUIL_XGB)).sum())
    ecart_ae = np.abs(reconstruction_locale - reconstruction_triton).max()
    print(f"{len(X_test)} vols de test")
    print(f"XGBoost      : écart max proba {ecart_proba:.2e}")
    print(f"               vols qui changent de côté du seuil {SEUIL_XGB} : {bascules}")
    print(f"Autoencodeur : écart max reconstruction {ecart_ae:.2e}")

    un_vol, un_vol_n, lot, lot_n = X_brut[:1], Xn[:1], X_brut[:1024], Xn[:1024]
    une_ligne = X_test.iloc[:1]
    mesures = {
        "XGB local, 1 vol": lambda: xgb.predict_proba(une_ligne),
        "XGB Triton, 1 vol": lambda: client.probas_usure(un_vol),
        "XGB Triton, 1024 vols": lambda: client.probas_usure(lot),
        "AE local, 1 vol": lambda: autoencodeur.predict_on_batch(un_vol_n),
        "AE Triton, 1 vol": lambda: client.reconstruction(un_vol_n),
        "AE Triton, 1024 vols": lambda: client.reconstruction(lot_n),
    }
    print("Latence moyenne :")
    for nom, fonction in mesures.items():
        repetitions = 10 if "1024" in nom else 50
        print(f"  {nom:<24} {chronometrer(fonction, repetitions):8.2f} ms")


if __name__ == "__main__":
    main()
