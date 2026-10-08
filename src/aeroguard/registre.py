"""Registre de modèles MLflow : versions numérotées, alias « champion » et règle de promotion."""

NOM_MODELE_ALERTE = "aeroguard-alerte"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"


def enregistrer_version(uri_modele, nom=NOM_MODELE_ALERTE, description=None, etiquettes=None):
    """Inscrit un modèle déjà enregistré dans un run ; renvoie le numéro de la nouvelle version."""
    import mlflow
    from mlflow import MlflowClient

    version = mlflow.register_model(uri_modele, nom)
    client = MlflowClient()
    if description:
        client.update_model_version(nom, version.version, description=description)
    for cle, valeur in (etiquettes or {}).items():
        client.set_model_version_tag(nom, version.version, cle, str(valeur))
    return int(version.version)


def definir_alias(alias, version, nom=NOM_MODELE_ALERTE):
    """Pose l'alias sur une version (s'il était sur une autre version, il se déplace)."""
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(nom, alias, str(version))


def lire_alias(alias, nom=NOM_MODELE_ALERTE):
    """Version qui porte l'alias : numéro, run, étiquettes. None si l'alias n'existe pas."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        version = MlflowClient().get_model_version_by_alias(nom, alias)
    except MlflowException:
        return None
    return {
        "version": int(version.version),
        "run_id": version.run_id,
        "etiquettes": dict(version.tags),
        "description": version.description,
    }


def uri_alias(alias, nom=NOM_MODELE_ALERTE):
    """Adresse MLflow d'un modèle par son alias, par exemple models:/aeroguard-alerte@champion."""
    return f"models:/{nom}@{alias}"


def doit_promouvoir(comparaison, challenger):
    """True seulement si le challenger bat le champion de façon PROUVÉE (bootstrap par moteur).

    `comparaison` est le tableau renvoyé par `comparer_a_reference(tirages, champion)`.
    """
    if challenger not in comparaison.index:
        raise ValueError(f"Le challenger « {challenger} » est absent de la comparaison.")
    ligne = comparaison.loc[challenger]
    return bool(ligne["difference_prouvee"]) and float(ligne["ecart_moyen"]) > 0
