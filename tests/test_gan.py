import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from aeroguard.gan import (  # noqa: E402
    creer_discriminateur,
    creer_generateur,
    entrainer_gan,
    generer,
)


@pytest.fixture
def mode_eager():
    """Exécute @tf.function en Python pur : la couverture voit alors l'intérieur de l'étape."""
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(False)


def petit_gan(nombre_features=4, dim_bruit=8):
    generateur = creer_generateur(nombre_features, dim_bruit=dim_bruit, couches_cachees=(16,))
    discriminateur = creer_discriminateur(nombre_features, couches_cachees=(16,))
    return generateur, discriminateur


def test_formes_generateur_et_discriminateur():
    generateur = creer_generateur(nombre_features=5, dim_bruit=8)
    discriminateur = creer_discriminateur(nombre_features=5)
    faux = generateur(np.zeros((3, 8), dtype="float32"))
    assert tuple(faux.shape) == (3, 5)
    assert tuple(discriminateur(faux).shape) == (3, 1)


def test_parametres_invalides():
    with pytest.raises(ValueError):
        creer_generateur(nombre_features=0)
    with pytest.raises(ValueError):
        creer_discriminateur(nombre_features=0)
    with pytest.raises(ValueError):
        creer_discriminateur(nombre_features=4, taux_dropout=1.0)


def test_entrainement_court_donne_un_historique_valide(mode_eager, capsys):
    X = np.random.default_rng(0).normal(3.0, 0.5, size=(128, 4)).astype("float32")
    generateur, discriminateur = petit_gan()
    historique = entrainer_gan(
        generateur, discriminateur, X, epochs=2, taille_lot=32, dim_bruit=8, afficher_tous_les=1
    )
    assert set(historique) == {"perte_d", "perte_g", "proba_vrais", "proba_faux"}
    for valeurs in historique.values():
        assert len(valeurs) == 2
        assert np.isfinite(valeurs).all()
    assert all(0 <= p <= 1 for p in historique["proba_vrais"] + historique["proba_faux"])
    assert "perte D" in capsys.readouterr().out


def test_entrainement_compile_fonctionne_aussi():
    X = np.random.default_rng(1).normal(size=(64, 4)).astype("float32")
    generateur, discriminateur = petit_gan()
    historique = entrainer_gan(
        generateur, discriminateur, X, epochs=1, taille_lot=32, dim_bruit=8, afficher_tous_les=0
    )
    assert len(historique["perte_g"]) == 1


def test_entrainement_refuse_les_donnees_invalides():
    generateur, discriminateur = petit_gan(nombre_features=2, dim_bruit=4)
    with pytest.raises(ValueError):
        entrainer_gan(generateur, discriminateur, np.ones(10), taille_lot=5)
    with pytest.raises(ValueError):
        entrainer_gan(generateur, discriminateur, np.ones((10, 2)), taille_lot=64)
    with pytest.raises(ValueError):
        entrainer_gan(generateur, discriminateur, np.full((10, 2), np.nan), taille_lot=5)
    with pytest.raises(ValueError):
        entrainer_gan(generateur, discriminateur, np.ones((10, 2)), epochs=0, taille_lot=5)


def test_generer_forme_et_reproductibilite():
    generateur = creer_generateur(nombre_features=3, dim_bruit=4)
    a = generer(generateur, 6, dim_bruit=4, graine=1)
    b = generer(generateur, 6, dim_bruit=4, graine=1)
    c = generer(generateur, 6, dim_bruit=4, graine=2)
    assert a.shape == (6, 3)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    with pytest.raises(ValueError):
        generer(generateur, 0)
