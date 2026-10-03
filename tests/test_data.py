import pandas as pd
from aeroguard.data import COLONNES_T, composant_touche, table_vols


def fausse_ligne(**modifs):
    """Ligne 'moteur neuf' (tout à 0), puis on modifie ce qu'on veut."""
    valeurs = {col: 0.0 for col in COLONNES_T}
    valeurs.update(modifs)
    return pd.Series(valeurs)


def test_moteur_neuf_aucun():
    assert composant_touche(fausse_ligne()) == "aucun"


def test_hpt_degradee():
    assert composant_touche(fausse_ligne(HPT_eff_mod=-0.02)) == "HPT"


def test_flow_compte_aussi():
    ligne = fausse_ligne(LPC_flow_mod=-0.03, HPT_eff_mod=-0.01)
    assert composant_touche(ligne) == "LPC"


def test_sous_le_seuil_aucun():
    assert composant_touche(fausse_ligne(fan_eff_mod=-0.0005)) == "aucun"


def test_table_vols_une_ligne_par_vol():
    df = pd.DataFrame({
        "unit":  [1, 1, 1, 1, 2],
        "cycle": [1, 1, 2, 2, 1],
        "hs":    [1, 1, 0, 0, 1],
        "RUL":   [1, 1, 0, 0, 5],
    })
    vols = table_vols(df)
    assert len(vols) == 3
    assert list(vols["duree_s"]) == [2, 2, 1]
    assert list(vols["RUL"]) == [1, 0, 5]