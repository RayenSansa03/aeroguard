"""Santé du système vue du dashboard et de la CI : appels de monitoring et mise en forme.

Le texte affiché est en anglais : c'est l'interface publique (dashboard, rapports CI).
"""

from __future__ import annotations

import time

import httpx

STATUTS = {
    "stable": ("Stable", "🟢", "Recent flights look like the training data."),
    "moderate": ("Moderate drift", "🟠", "Some features moved: worth a look."),
    "major": (
        "Major drift",
        "🔴",
        "Investigate the drifting features: drift calls for an investigation, "
        "not an automatic retraining.",
    ),
    "insufficient_data": ("Not enough flights", "⚪", "The report needs more recent flights."),
    "insufficient_engines": (
        "Not enough engines",
        "⚪",
        "One engine is not a fleet: waiting for flights from more engines.",
    ),
}


def lecture_statut(statut: str) -> tuple[str, str, str]:
    """Titre, icône et explication d'un statut de dérive."""
    return STATUTS.get(statut, (statut, "⚪", ""))


class ClientMonitoring:
    """Appels de monitoring vers l'API (santé et dérive)."""

    def __init__(self, url: str, client: httpx.Client | None = None, timeout: float = 30.0):
        self.client = client or httpx.Client(base_url=url, timeout=timeout)

    def sante(self) -> dict:
        """Réponse de /health, avec le temps aller-retour et le temps serveur."""
        debut = time.perf_counter()
        reponse = self.client.get("/health")
        reponse.raise_for_status()
        corps = reponse.json()
        corps["round_trip_ms"] = round((time.perf_counter() - debut) * 1000, 1)
        corps["server_ms"] = float(reponse.headers.get("X-Response-Time-ms", "nan"))
        return corps

    def derive(self, limite: int = 5000) -> dict:
        """Rapport de dérive sur les derniers vols reçus."""
        reponse = self.client.get("/monitoring/derive", params={"limite": limite})
        reponse.raise_for_status()
        return reponse.json()


def lignes_derive(rapport: dict, n: int = 10) -> list[dict]:
    """Les n features qui dérivent le plus, prêtes à afficher."""
    return [
        {"Feature": ligne["feature"], "PSI": ligne["psi"], "Status": ligne["status"]}
        for ligne in rapport.get("features", [])[:n]
    ]


def rapport_markdown(rapport: dict, url: str) -> str:
    """Résumé Markdown d'un rapport de dérive (résumé de job CI, corps d'issue)."""
    titre, icone, explication = lecture_statut(rapport["status"])
    lignes = [f"## {icone} AeroGuard drift report: {titre}", "", f"API: {url}", "", explication, ""]
    if "n_flights" in rapport:
        lignes.append(f"- Flights analysed: {rapport['n_flights']}")
    if "n_engines" in rapport:
        lignes.append(f"- Engines: {rapport['n_engines']}")
    if rapport.get("features"):
        lignes += ["", "| Feature | PSI | Status |", "|---|---|---|"]
        lignes += [
            f"| {ligne['Feature']} | {ligne['PSI']:.3f} | {ligne['Status']} |"
            for ligne in lignes_derive(rapport)
        ]
    return "\n".join(lignes) + "\n"
