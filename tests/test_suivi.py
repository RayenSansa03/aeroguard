import pytest

from aeroguard.suivi import enregistrer_run


def test_enregistrer_run_puis_relire(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    base = (tmp_path / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{base}")

    run_id = enregistrer_run(
        "test_xgb",
        reglages={"max_depth": 4, "seuil": 0.81},
        scores={"f1_macro": 0.885, "fp": 27},
        experience="test",
        etiquettes={"lecon": "D3"},
    )

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "test_xgb"
    assert run.data.params["max_depth"] == "4"  # MLflow stocke les réglages en texte
    assert run.data.metrics["f1_macro"] == pytest.approx(0.885)
    assert run.data.tags["lecon"] == "D3"
