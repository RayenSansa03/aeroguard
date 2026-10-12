"""Logs structurés : une ligne JSON par événement, lisible par les outils de monitoring."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class FormatJSON(logging.Formatter):
    """Transforme chaque enregistrement en une ligne JSON.

    Les champs métier passent par `extra={"champs": {...}}` et sont ajoutés au même niveau
    que les champs standard. Les clés sont en anglais : c'est une interface pour les outils.
    """

    def format(self, record: logging.LogRecord) -> str:
        entree = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entree.update(getattr(record, "champs", {}))
        if record.exc_info:
            entree["exception"] = self.formatException(record.exc_info)
        return json.dumps(entree, ensure_ascii=False, default=str)


def configurer_logs(niveau: str = "INFO") -> logging.Logger:
    """Installe le format JSON sur tous les loggers « aeroguard.* » (sortie standard)."""
    gestionnaire = logging.StreamHandler(sys.stdout)
    gestionnaire.setFormatter(FormatJSON())
    logger = logging.getLogger("aeroguard")
    logger.handlers[:] = [gestionnaire]
    logger.setLevel(niveau)
    return logger
