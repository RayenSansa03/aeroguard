"""Stockage de l'historique des moteurs : interface commune et implémentation en mémoire."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class HistoriqueMoteur:
    """Ce que le service garde pour un moteur."""

    cycles: list = field(default_factory=list)
    vols: list = field(default_factory=list)
    probas: list = field(default_factory=list)
    erreurs: list = field(default_factory=list)
    recommandations: list = field(default_factory=list)
    derniere: dict | None = None


class DepotHistorique(Protocol):
    """Contrat de stockage dont le service a besoin, quel que soit le support."""

    def verrou(self, moteur: str) -> AbstractContextManager:
        """Verrou qui garantit que les vols d'un même moteur sont traités un par un."""

    def lire(self, moteur: str) -> HistoriqueMoteur | None:
        """Historique du moteur, ou None s'il est inconnu."""

    def ecrire(self, moteur: str, historique: HistoriqueMoteur) -> None:
        """Remplace l'historique du moteur par la nouvelle version complète."""

    def supprimer(self, moteur: str) -> bool:
        """Oublie le moteur. False s'il était inconnu."""

    def dernieres(self) -> dict[str, dict]:
        """Dernière recommandation de chaque moteur connu."""


class DepotMemoire:
    """Historiques gardés dans la mémoire du processus.

    Simple et rapide, mais limité à un seul worker et perdu au redémarrage.
    """

    def __init__(self):
        self._historiques: dict[str, HistoriqueMoteur] = {}
        self._verrous: dict[str, threading.Lock] = {}
        self._registre = threading.Lock()

    def verrou(self, moteur: str) -> threading.Lock:
        with self._registre:
            return self._verrous.setdefault(moteur, threading.Lock())

    def lire(self, moteur: str) -> HistoriqueMoteur | None:
        with self._registre:
            return self._historiques.get(moteur)

    def ecrire(self, moteur: str, historique: HistoriqueMoteur) -> None:
        with self._registre:
            self._historiques[moteur] = historique

    def supprimer(self, moteur: str) -> bool:
        with self._registre:
            return self._historiques.pop(moteur, None) is not None

    def dernieres(self) -> dict[str, dict]:
        with self._registre:
            return {
                moteur: h.derniere
                for moteur, h in self._historiques.items()
                if h.derniere is not None
            }
