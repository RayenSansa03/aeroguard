import numpy as np
import pandas as pd
import pytest

from aeroguard.registre import (
    ALIAS_CHAMPION,
    definir_alias,
    doit_promouvoir,
    enregistrer_version,
    lire_alias,
    uri_alias,
)


def comparaison_exemple():
    """Comme la sortie de comparer_a_reference(tirages, "champion")."""
    return pd.DataFrame(
        {
            "ecart_moyen": [0.0, 0.03, -0.01, -0.10],
            "difference_prouvee": [False, True, False, True],
        },
        index=pd.Index(["champion", "meilleur", "egal", "moins_bon"], name="modele"),
    )


@pytest.mark.parametrize(
    "challenger, attendu",
    [("meilleur", True), ("egal", False), ("moins_bon", False), ("champion", False)],
)
def test_doit_promouvoir(challenger, attendu):
    assert doit_promouvoir(comparaison_exemple(), challenger) is attendu


def test_doit_promouvoir_challenger_absent():
    with pytest.raises(ValueError):
        doit_promouvoir(comparaison_exemple(), "inconnu")


def test_uri_alias():
    assert uri_alias("champion") == "models:/aeroguard-alerte@champion"
    assert uri_alias("challenger", nom="test") == "models:/test@challenger"


@pytest.fixture
def registre_temporaire(tmp_path):
    """Un MLflow vide (base SQLite) le temps d'un test, avec un petit modèle enregistré."""
    mlflow = pytest.importorskip("mlflow")
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("test-registre")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    modele = LogisticRegression().fit(X, [0, 0, 1, 1])
    with mlflow.start_run():
        infos = mlflow.sklearn.log_model(modele, name="model", serialization_format="cloudpickle")
    return mlflow, infos.model_uri, modele, X


def test_versions_alias_et_chargement(registre_temporaire):
    mlflow, uri_modele, modele, X = registre_temporaire

    v1 = enregistrer_version(
        uri_modele, nom="test", description="premier", etiquettes={"seuil": 0.81, "k": 3}
    )
    v2 = enregistrer_version(uri_modele, nom="test")
    assert (v1, v2) == (1, 2)  # chaque inscription crée une NOUVELLE version

    assert lire_alias(ALIAS_CHAMPION, nom="test") is None  # pas encore d'alias

    definir_alias(ALIAS_CHAMPION, v1, nom="test")
    infos = lire_alias(ALIAS_CHAMPION, nom="test")
    assert infos["version"] == 1
    assert infos["etiquettes"]["seuil"] == "0.81"  # MLflow stocke du texte
    assert infos["etiquettes"]["k"] == "3"
    assert infos["description"] == "premier"

    definir_alias(ALIAS_CHAMPION, v2, nom="test")  # l'alias se déplace
    assert lire_alias(ALIAS_CHAMPION, nom="test")["version"] == 2

    recharge = mlflow.sklearn.load_model(uri_alias(ALIAS_CHAMPION, nom="test"))
    assert np.array_equal(recharge.predict(X), modele.predict(X))
