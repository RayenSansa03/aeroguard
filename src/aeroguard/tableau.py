"""Outils du tableau de bord : lecture de l'API flotte et mise en forme des données."""

import httpx
import pandas as pd

NIVEAUX = ("URGENT", "A_SURVEILLER", "OK")
COULEURS = {"OK": "#55A868", "A_SURVEILLER": "#DD8452", "URGENT": "#C44E52"}
LIBELLES = {"OK": "OK", "A_SURVEILLER": "Watch", "URGENT": "Urgent"}
MESSAGES = {
    "OK": "No confirmed alert: normal operation.",
    "A_SURVEILLER": "Early warning: keep this engine under close watch.",
    "URGENT": "Confirmed degradation: schedule maintenance as a priority.",
}


class ClientFlotte:
    """Lit l'API AeroGuard (routes /flotte) pour le tableau de bord."""

    def __init__(self, url="http://127.0.0.1:8000", client=None, timeout=10.0):
        self.client = client or httpx.Client(base_url=url, timeout=timeout)

    def disponible(self):
        """True si l'API répond sur /health."""
        try:
            return self.client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def flotte(self):
        """Synthèse de la flotte (une ligne par moteur, les plus urgents en premier)."""
        reponse = self.client.get("/flotte/moteurs")
        reponse.raise_for_status()
        return reponse.json()

    def historique(self, moteur):
        """Toutes les recommandations d'un moteur, ou une liste vide s'il est inconnu."""
        reponse = self.client.get(f"/flotte/moteurs/{moteur}/historique")
        if reponse.status_code == 404:
            return []
        reponse.raise_for_status()
        return reponse.json()


def compter_niveaux(flotte):
    """Nombre de moteurs par niveau, avec les trois niveaux toujours présents."""
    comptes = dict.fromkeys(NIVEAUX, 0)
    for ligne in flotte:
        comptes[ligne["niveau"]] = comptes.get(ligne["niveau"], 0) + 1
    return comptes


def tableau_flotte(flotte):
    """Tableau de la flotte prêt à afficher (colonnes en anglais pour le tableau de bord)."""
    colonnes = ["moteur", "niveau", "cycle", "vols_recus"]
    tableau = pd.DataFrame(flotte, columns=colonnes)
    return pd.DataFrame(
        {
            "Engine": tableau["moteur"],
            "Status": tableau["niveau"].map(LIBELLES),
            "Last cycle": tableau["cycle"],
            "Flights received": tableau["vols_recus"],
            "niveau": tableau["niveau"],
        }
    )


def tableau_historique(historique):
    """Une ligne par vol : cycle, probabilité d'usure, erreur de reconstruction, niveau."""
    colonnes = ["cycle", "proba_usure", "erreur_reconstruction", "niveau"]
    tableau = pd.DataFrame(historique, columns=colonnes).sort_values("cycle", ignore_index=True)
    tableau["status"] = tableau["niveau"].map(LIBELLES)
    return tableau
