"""Découpe les fichiers N-CMAPSS bruts en fenêtres pour les réseaux (leçon 4.5).

Utilisation (PowerShell, depuis la racine du projet) :

    python scripts/preparer_fenetres.py --dossier-h5 C:/Users/rayen/data_set

Le résultat (fenetres_v4.npz et fenetres_v4_infos.parquet) est écrit dans
data/processed/, avec le même découpage train / val / test que la v1.
"""

import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aeroguard.sequences import (
    CAPTEURS,
    CONDITIONS_VOL,
    extraire_fenetres,
    normaliser_fenetres,
    statistiques_canaux,
)

CANAUX = list(CAPTEURS) + list(CONDITIONS_VOL)


def noms_variables(fichier, cle):
    return [nom.decode() if isinstance(nom, bytes) else str(nom) for nom in fichier[cle][()]]


def lire_partie(fichier, partie):
    """Signal (capteurs + conditions), unité et cycle de chaque seconde d'une partie."""
    noms_capteurs = noms_variables(fichier, "X_s_var")
    noms_conditions = noms_variables(fichier, "W_var")
    noms_auxiliaires = noms_variables(fichier, "A_var")

    capteurs = fichier[f"X_s_{partie}"][()].astype("float32")
    capteurs = capteurs[:, [noms_capteurs.index(nom) for nom in CAPTEURS]]
    conditions = fichier[f"W_{partie}"][()].astype("float32")
    conditions = conditions[:, [noms_conditions.index(nom) for nom in CONDITIONS_VOL]]
    auxiliaires = fichier[f"A_{partie}"][()]

    signal = np.hstack([capteurs, conditions])
    unites = auxiliaires[:, noms_auxiliaires.index("unit")].astype(int)
    cycles = auxiliaires[:, noms_auxiliaires.index("cycle")].astype(int)
    return signal, unites, cycles


def fenetres_du_fichier(fichier, prefixe, etiquettes, longueur, nombre):
    """Fenêtres et informations de tous les vols étiquetés d'un fichier."""
    lots, lignes = [], []
    for partie in ("dev", "test"):
        signal, unites, cycles = lire_partie(fichier, partie)
        lignes_par_vol = (
            pd.DataFrame({"unit": unites, "cycle": cycles})
            .groupby(["unit", "cycle"], sort=False)
            .indices
        )
        for (unite, cycle), positions in lignes_par_vol.items():
            moteur = f"{prefixe}_{unite}"
            etiquette = etiquettes.get((moteur, int(cycle)))
            if etiquette is None:
                continue
            fenetres = extraire_fenetres(signal[positions], longueur, nombre)
            if len(fenetres) == 0:
                continue
            lots.append(fenetres)
            for position in range(len(fenetres)):
                lignes.append(
                    {
                        "moteur": moteur,
                        "cycle": int(cycle),
                        "position": position,
                        "usure": 1 - int(etiquette["hs"]),
                        "RUL": etiquette["RUL"],
                        "groupe": etiquette["groupe"],
                    }
                )
        del signal, unites, cycles, lignes_par_vol
        gc.collect()
    return lots, lignes


def main():
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--dossier-h5", required=True, type=Path)
    analyseur.add_argument("--vols", type=Path, default=Path("data/processed/vols_ncmapss.parquet"))
    analyseur.add_argument("--sortie", type=Path, default=Path("data/processed"))
    analyseur.add_argument("--longueur", type=int, default=256)
    analyseur.add_argument("--fenetres-par-vol", type=int, default=4)
    arguments = analyseur.parse_args()

    import h5py

    vols = pd.read_parquet(arguments.vols)
    etiquettes = (
        vols.assign(cycle=vols["cycle"].astype(int))
        .set_index(["moteur", "cycle"])[["hs", "RUL", "groupe"]]
        .to_dict("index")
    )
    prefixes_v1 = {moteur.split("_")[0] for moteur in vols["moteur"].unique()}
    print(f"Vols étiquetés : {len(etiquettes)} | fichiers de la v1 : {sorted(prefixes_v1)}")

    tous_les_lots, toutes_les_lignes = [], []
    for chemin in sorted(arguments.dossier_h5.glob("*.h5")):
        trouve = re.search(r"(DS\d{2}[a-z]?)", chemin.name)
        if trouve is None or trouve.group(1) not in prefixes_v1:
            print(f"{chemin.name:30s} → ignoré (absent de la v1)")
            continue
        debut = time.perf_counter()
        with h5py.File(chemin, "r") as fichier:
            lots, lignes = fenetres_du_fichier(
                fichier,
                trouve.group(1),
                etiquettes,
                arguments.longueur,
                arguments.fenetres_par_vol,
            )
        tous_les_lots += lots
        toutes_les_lignes += lignes
        print(f"{chemin.name:30s} → {len(lots):4d} vols ({time.perf_counter() - debut:.0f} s)")

    X_toutes = np.concatenate(tous_les_lots)
    infos = pd.DataFrame(toutes_les_lignes)
    del tous_les_lots
    gc.collect()

    masques = {g: (infos["groupe"] == g).to_numpy() for g in ("train", "val", "test")}
    moyennes, ecarts = statistiques_canaux(X_toutes[masques["train"]])
    donnees = {"moyennes": moyennes, "ecarts": ecarts, "canaux": np.array(CANAUX)}
    for groupe, masque in masques.items():
        donnees[f"X_{groupe}"] = normaliser_fenetres(X_toutes[masque], moyennes, ecarts)
        donnees[f"y_{groupe}"] = infos.loc[masque, "usure"].to_numpy().astype("float32")

    arguments.sortie.mkdir(parents=True, exist_ok=True)
    chemin_fenetres = arguments.sortie / "fenetres_v4.npz"
    np.savez(chemin_fenetres, **donnees)
    infos.to_parquet(arguments.sortie / "fenetres_v4_infos.parquet")

    print(f"\nVols retenus : {infos.groupby(['moteur', 'cycle']).ngroups} (attendu : 7473)")
    print(f"Forme de X : {X_toutes.shape} | {X_toutes.nbytes / 1e9:.2f} Go")
    print(
        infos.groupby("groupe")
        .agg(
            fenetres=("moteur", "size"),
            moteurs=("moteur", "nunique"),
            part_usure=("usure", "mean"),
        )
        .round(3)
    )
    print(f"✅ Sauvegardé : {chemin_fenetres} ({chemin_fenetres.stat().st_size / 1e9:.2f} Go)")


if __name__ == "__main__":
    main()
