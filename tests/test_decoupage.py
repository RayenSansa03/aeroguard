import pandas as pd
import pytest

from aeroguard.decoupage import decouper_par_moteur, verifier_sans_fuite


def faux_vols():
    """6 moteurs × 3 vols."""
    return pd.DataFrame(
        {"unit": [u for u in range(1, 7) for _ in range(3)], "cycle": [1, 2, 3] * 6}
    )


def test_aucun_moteur_dans_deux_groupes():
    train, val, test = decouper_par_moteur(faux_vols(), moteurs_val=[2, 5], moteurs_test=[6])
    assert set(train["unit"]) == {1, 3, 4}
    assert set(val["unit"]) == {2, 5}
    assert set(test["unit"]) == {6}


def test_aucune_ligne_perdue():
    vols = faux_vols()
    train, val, test = decouper_par_moteur(vols, moteurs_val=[2, 5], moteurs_test=[6])
    assert len(train) + len(val) + len(test) == len(vols)


def test_meme_moteur_val_et_test_interdit():
    with pytest.raises(ValueError):
        decouper_par_moteur(faux_vols(), moteurs_val=[2, 5], moteurs_test=[5])


def test_verifier_sans_fuite_detecte_un_moteur_partage():
    vols = faux_vols()
    a = vols[vols["unit"].isin([1, 2])]
    b = vols[vols["unit"].isin([2, 3])]  # le moteur 2 est dans les deux !
    with pytest.raises(ValueError):
        verifier_sans_fuite(a, b)
