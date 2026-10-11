"""Simulateur de flotte : des moteurs volent en parallèle, l'API alerte, l'atelier intervient."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aeroguard.rejeu import en_mesures


@dataclass
class MoteurSimule:
    """Un moteur de test rejoué vol par vol, avec sa durée de vie restante (RUL) réelle."""

    nom: str
    cycles: np.ndarray
    X: pd.DataFrame
    rul: np.ndarray
    position: int = 0
    rul_alerte: int | None = None
    vols_avant_atelier: int | None = None
    issue: str | None = None
    rul_sortie: int | None = None


class Simulateur:
    """Fait voler les moteurs tour par tour et applique la règle de l'atelier.

    Règle : au premier niveau URGENT, le moteur fait encore `delai_atelier` vols (le temps
    de planifier l'intervention), puis part en maintenance. S'il arrive au bout de sa vie
    avant, c'est une panne en service.
    """

    def __init__(self, moteurs, envoyer, reinitialiser=None, delai_atelier=5, en_vol_max=8):
        self.attente = list(moteurs)
        self.actifs = []
        self.termines = []
        self.envoyer = envoyer
        self.reinitialiser = reinitialiser or (lambda nom: None)
        self.delai_atelier = delai_atelier
        self.en_vol_max = en_vol_max

    def termine(self):
        return not self.attente and not self.actifs

    def _sortir(self, moteur, issue, evenements):
        moteur.issue = issue
        moteur.rul_sortie = int(moteur.rul[moteur.position - 1])
        self.actifs.remove(moteur)
        self.termines.append(moteur)
        self.reinitialiser(moteur.nom)
        evenements.append({"moteur": moteur.nom, "evenement": issue, "rul": moteur.rul_sortie})

    def tour(self):
        """Chaque moteur en vol fait un vol. Renvoie les événements du tour."""
        while self.attente and len(self.actifs) < self.en_vol_max:
            moteur = self.attente.pop(0)
            self.reinitialiser(moteur.nom)
            self.actifs.append(moteur)
        evenements = []
        for moteur in list(self.actifs):
            i = moteur.position
            reco = self.envoyer(moteur.nom, int(moteur.cycles[i]), en_mesures(moteur.X.iloc[i]))
            moteur.position += 1
            if reco["niveau"] == "URGENT" and moteur.rul_alerte is None:
                moteur.rul_alerte = int(moteur.rul[i])
                moteur.vols_avant_atelier = self.delai_atelier
                evenements.append(
                    {"moteur": moteur.nom, "evenement": "alerte", "rul": moteur.rul_alerte}
                )
            elif moteur.vols_avant_atelier is not None:
                moteur.vols_avant_atelier -= 1
            # Le dernier vol (RUL = 0) est celui de la panne : l'atelier arrive trop tard.
            if moteur.position == len(moteur.cycles):
                self._sortir(moteur, "panne", evenements)
            elif moteur.vols_avant_atelier == 0:
                self._sortir(moteur, "maintenance", evenements)
        return evenements

    def bilan(self):
        """Résultat opérationnel : pannes évitées, pannes en service, vie gâchée."""
        maintenances = [m for m in self.termines if m.issue == "maintenance"]
        alertes = [m.rul_alerte for m in self.termines if m.rul_alerte is not None]
        return {
            "moteurs": len(self.termines),
            "pannes_evitees": len(maintenances),
            "pannes_en_service": sum(m.issue == "panne" for m in self.termines),
            "avance_mediane": float(np.median(alertes)) if alertes else None,
            "vols_restants_moyens": (
                float(np.mean([m.rul_sortie for m in maintenances])) if maintenances else None
            ),
        }
