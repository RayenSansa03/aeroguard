"""Pipeline complet : un fichier N-CMAPSS → une ligne par vol, prête pour les modèles."""

import numpy as np
import pandas as pd

from aeroguard.data import COLONNES_T, charger_ncmapss, mode_de_panne, table_vols
from aeroguard.features import ajouter_phase, features_par_vol
from aeroguard.normalisation import CAPTEURS, ajuster_modele_sain, calculer_residus


def preparer(chemin, nom, moteurs_val=(), n_vols_sains=15):
    """Charge un fichier, normalise, résume chaque vol et ajoute les étiquettes."""
    dev = charger_ncmapss(chemin, "dev")
    test = charger_ncmapss(chemin, "test")

    # 1) Modèle du moteur sain : moteurs TRAIN seulement (corrige la fuite de la leçon 1.8)
    train = dev[~dev["unit"].isin(moteurs_val)]
    debut_panne_min = train.loc[train["hs"] == 0].groupby("unit")["cycle"].min().min()
    n_sains = int(min(n_vols_sains, debut_panne_min - 1))
    modele = ajuster_modele_sain(train, n_vols_sains=n_sains)

    morceaux = []
    for partie, df in [("dev", dev), ("test", test)]:
        # 2) Phases, résidus, features
        df = ajouter_phase(df)
        res = calculer_residus(df, modele)
        res["phase"] = df["phase"].values
        vols = features_par_vol(res, CAPTEURS)

        # 3) Étiquettes (hs, RUL, durée), classe de vol, mode de panne du moteur
        fc = df.groupby(["unit", "cycle"])["Fc"].first().reset_index()
        modes = (
            df.groupby("unit")[COLONNES_T]
            .last()
            .apply(mode_de_panne, axis=1)
            .rename("mode")
            .reset_index()
        )
        vols = vols.merge(table_vols(df), on=["unit", "cycle"])
        vols = vols.merge(fc, on=["unit", "cycle"]).merge(modes, on="unit")

        # 4) Groupe : train / val / test
        if partie == "test":
            vols["groupe"] = "test"
        else:
            vols["groupe"] = np.where(vols["unit"].isin(moteurs_val), "val", "train")
        morceaux.append(vols)

    vols = pd.concat(morceaux, ignore_index=True)
    vols.insert(0, "dataset", nom)
    vols.insert(1, "moteur", nom + "_" + vols["unit"].astype(str))
    return vols
