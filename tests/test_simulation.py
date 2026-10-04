"""Tests automatiques du simulateur de flotte."""

from aeroguard.simulation import simuler_flotte


def test_colonnes_au_format_ncmapss():
    flotte = simuler_flotte(n_moteurs=3)
    assert list(flotte.columns) == ["unit", "cycle", "T48", "P30", "hs", "rul"]


def test_nombre_de_moteurs():
    flotte = simuler_flotte(n_moteurs=4)
    assert flotte["unit"].nunique() == 4


def test_chaque_moteur_finit_en_panne():
    flotte = simuler_flotte(n_moteurs=5)
    dernier_rul = flotte.groupby("unit")["rul"].min()
    assert (dernier_rul == 0).all()


def test_etat_de_sante_vaut_0_ou_1():
    flotte = simuler_flotte()
    assert set(flotte["hs"].unique()) == {0, 1}


def test_moteurs_uses_plus_chauds():
    flotte = simuler_flotte()
    moyennes = flotte.groupby("hs")["T48"].mean()
    assert moyennes[0] > moyennes[1]


def test_meme_graine_meme_flotte():
    flotte_a = simuler_flotte(seed=7)
    flotte_b = simuler_flotte(seed=7)
    assert flotte_a.equals(flotte_b)
