"""Suivre les expériences avec MLflow : un run = des réglages + des scores + des étiquettes."""


def enregistrer_run(nom, reglages, scores, experience="aeroguard", etiquettes=None):
    """Crée un run MLflow avec ses paramètres, ses métriques et ses étiquettes ; renvoie son id."""
    import mlflow

    mlflow.set_experiment(experience)
    with mlflow.start_run(run_name=nom) as run:
        mlflow.log_params(reglages)
        mlflow.log_metrics({cle: float(valeur) for cle, valeur in scores.items()})
        if etiquettes:
            mlflow.set_tags(etiquettes)
    return run.info.run_id
