import numpy as np
import pandas as pd
import pytest

from aeroguard.sequences import (
    CAPTEURS,
    CONDITIONS_VOL,
    agreger_par_vol,
    creer_dataset,
    decouper_fenetres,
    extraire_fenetres,
    normaliser_fenetres,
    statistiques_canaux,
)


def signal_test(temps=10, canaux=2):
    return np.arange(temps * canaux, dtype="float32").reshape(temps, canaux)


def test_listes_de_canaux():
    assert len(CAPTEURS) == 14
    assert len(CONDITIONS_VOL) == 4


def test_decouper_fenetres_forme_et_contenu():
    signal = signal_test()
    fenetres = decouper_fenetres(signal, longueur=4, pas=2)
    assert fenetres.shape == (4, 4, 2)
    assert np.array_equal(fenetres[0], signal[0:4])
    assert np.array_equal(fenetres[1], signal[2:6])
    assert np.array_equal(fenetres[-1], signal[6:10])


def test_decouper_fenetres_signal_trop_court():
    assert decouper_fenetres(signal_test(temps=3), longueur=4, pas=1).shape == (0, 4, 2)


@pytest.mark.parametrize("longueur, pas", [(0, 1), (4, 0)])
def test_decouper_fenetres_parametres_invalides(longueur, pas):
    with pytest.raises(ValueError):
        decouper_fenetres(signal_test(), longueur, pas)


def test_signal_a_une_dimension_refuse():
    with pytest.raises(ValueError, match="2 dimensions"):
        decouper_fenetres(np.arange(10), longueur=4, pas=1)
    with pytest.raises(ValueError, match="2 dimensions"):
        extraire_fenetres(np.arange(10), longueur=4, nombre=2)


def test_extraire_fenetres_regulieres():
    signal = signal_test(temps=100, canaux=3)
    fenetres = extraire_fenetres(signal, longueur=10, nombre=4)
    assert fenetres.shape == (4, 10, 3)
    assert np.array_equal(fenetres[0], signal[0:10])
    assert np.array_equal(fenetres[1], signal[30:40])
    assert np.array_equal(fenetres[-1], signal[90:100])


def test_extraire_fenetres_vol_trop_court_ou_nombre_invalide():
    assert extraire_fenetres(signal_test(temps=5), longueur=10, nombre=4).shape == (0, 10, 2)
    with pytest.raises(ValueError):
        extraire_fenetres(signal_test(), longueur=4, nombre=0)


def test_normalisation_par_canal():
    generateur = np.random.default_rng(0)
    fenetres = generateur.normal(loc=500, scale=20, size=(50, 30, 3)).astype("float32")
    fenetres[:, :, 2] = 7.0  # canal constant
    moyennes, ecarts = statistiques_canaux(fenetres)
    assert ecarts[2] == 1.0  # pas de division par zéro
    normalisees = normaliser_fenetres(fenetres, moyennes, ecarts)
    assert normalisees.dtype == np.float32
    assert np.allclose(normalisees[:, :, :2].mean(axis=(0, 1)), 0, atol=1e-3)
    assert np.allclose(normalisees[:, :, :2].std(axis=(0, 1)), 1, atol=1e-3)
    assert np.allclose(normalisees[:, :, 2], 0)


def test_creer_dataset_lots():
    pytest.importorskip("tensorflow")
    X = np.zeros((10, 4, 2), dtype="float32")
    y = np.arange(10, dtype="float32")
    lots = list(creer_dataset(X, y, taille_lot=4, melanger=False))
    assert len(lots) == 3
    assert tuple(lots[0][0].shape) == (4, 4, 2)
    assert np.array_equal(lots[0][1].numpy(), [0, 1, 2, 3])
    lots_melanges = list(creer_dataset(X, y, taille_lot=10, melanger=True))
    assert sorted(lots_melanges[0][1].numpy()) == list(range(10))


def test_agreger_par_vol_moyenne_et_ordre():
    infos = pd.DataFrame(
        {
            "moteur": ["B", "B", "A", "A", "A", "A"],
            "cycle": [1, 1, 2, 2, 1, 1],
            "usure": [0, 0, 1, 1, 0, 0],
            "RUL": [5, 5, 1, 1, 2, 2],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    par_vol = agreger_par_vol(infos, [0.1, 0.3, 0.8, 1.0, 0.2, 0.4])
    assert list(par_vol["moteur"]) == ["A", "A", "B"]
    assert list(par_vol["cycle"]) == [1, 2, 1]
    assert np.allclose(par_vol["proba"], [0.3, 0.9, 0.2])
    assert list(par_vol["usure"]) == [0, 1, 0]


def test_agreger_par_vol_tailles_differentes():
    import pandas as pd

    infos = pd.DataFrame({"moteur": ["A"], "cycle": [1], "usure": [0], "RUL": [3]})
    with pytest.raises(ValueError, match="même nombre"):
        agreger_par_vol(infos, [0.1, 0.2])
