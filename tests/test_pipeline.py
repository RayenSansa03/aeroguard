import h5py
import numpy as np
import pandas as pd
from aeroguard.data import COLONNES_T, charger_ncmapss, mode_de_panne
from aeroguard.decoupage import verifier_sans_fuite
from aeroguard.normalisation import CAPTEURS, W_COLS
from aeroguard.pipeline import preparer

N_VOLS, PANNE_APRES = 20, 12


def faux_fichier(chemin):
    """Mini fichier au format N-CMAPSS : dev = moteurs 1, 2, 3 ; test = moteur 4."""
    rng = np.random.default_rng(0)
    alt = [0, 5, 10, 10, 10, 10, 10, 10, 5, 0]                 # 10 secondes par vol
    with h5py.File(chemin, "w") as f:
        f["A_var"] = np.array(["unit", "cycle", "Fc", "hs"], dtype="S")
        f["W_var"] = np.array(W_COLS, dtype="S")
        f["X_s_var"] = np.array(CAPTEURS, dtype="S")
        f["T_var"] = np.array(COLONNES_T, dtype="S")
        for partie, moteurs in [("dev", [1, 2, 3]), ("test", [4])]:
            lignes = []
            for u in moteurs:
                for c in range(1, N_VOLS + 1):
                    usure = max(c - PANNE_APRES, 0)
                    for a in alt:
                        w = [a, rng.uniform(0.2, 0.8), rng.uniform(40, 90), rng.uniform(450, 520)]
                        x = [sum(w) + 0.5 * usure] * len(CAPTEURS)          # capteurs = règle simple + usure
                        t = [0.0] * len(COLONNES_T)
                        t[COLONNES_T.index("HPT_eff_mod")] = -0.001 * usure
                        lignes.append([u, c, 1, int(c <= PANNE_APRES)] + w + x + t + [N_VOLS - c])
            lignes = np.array(lignes, dtype=float)
            f[f"A_{partie}"] = lignes[:, 0:4]
            f[f"W_{partie}"] = lignes[:, 4:8]
            f[f"X_s_{partie}"] = lignes[:, 8:22]
            f[f"T_{partie}"] = lignes[:, 22:32]
            f[f"Y_{partie}"] = lignes[:, 32:33]
    return chemin


def ligne_T(**modifs):
    valeurs = {col: 0.0 for col in COLONNES_T}
    valeurs.update(modifs)
    return pd.Series(valeurs)


def test_charger_ncmapss(tmp_path):
    df = charger_ncmapss(faux_fichier(tmp_path / "faux.h5"), "dev")
    assert df.shape == (3 * N_VOLS * 10, 33)
    assert "HPT_eff_mod" in df.columns and "RUL" in df.columns
    assert df["unit"].dtype == "int32"


def test_mode_de_panne_une_piece():
    assert mode_de_panne(ligne_T(HPT_eff_mod=-0.02)) == "HPT"


def test_mode_de_panne_deux_pieces():
    assert mode_de_panne(ligne_T(HPT_eff_mod=-0.02, LPT_flow_mod=-0.01)) == "HPT+LPT"


def test_mode_de_panne_aucune():
    assert mode_de_panne(ligne_T()) == "aucun"


def test_preparer_une_ligne_par_vol(tmp_path):
    vols = preparer(faux_fichier(tmp_path / "faux.h5"), "FAUX", moteurs_val=[3])
    assert len(vols) == 4 * N_VOLS
    assert set(vols["groupe"]) == {"train", "val", "test"}
    assert vols["moteur"].iloc[0] == "FAUX_1"
    assert set(vols["mode"]) == {"HPT"}


def test_preparer_sans_fuite(tmp_path):
    vols = preparer(faux_fichier(tmp_path / "faux.h5"), "FAUX", moteurs_val=[3])
    groupes = [vols[vols["groupe"] == g] for g in ["train", "val", "test"]]
    assert verifier_sans_fuite(*groupes)


def test_preparer_voit_l_usure(tmp_path):
    vols = preparer(faux_fichier(tmp_path / "faux.h5"), "FAUX", moteurs_val=[3])
    debut = vols.loc[vols["cycle"] <= 10, "T48_mean_croisiere"].mean()
    fin = vols.loc[vols["cycle"] >= 18, "T48_mean_croisiere"].mean()
    assert fin - debut > 2                         # le résidu monte avec l'usure