"""Rapport de dérive hebdomadaire : interroge l'API en ligne et écrit un résumé Markdown."""

import argparse
import os
import time
from pathlib import Path

import httpx

from aeroguard.sante_systeme import ClientMonitoring, rapport_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sortie", default="rapport_derive.md")
    parser.add_argument("--essais", type=int, default=6)
    args = parser.parse_args()

    # Timeout long et plusieurs essais : un service gratuit endormi met ~1 minute à se réveiller.
    client = ClientMonitoring(args.url, timeout=120)
    for essai in range(1, args.essais + 1):
        try:
            client.sante()
            rapport = client.derive()
            break
        except httpx.HTTPError as erreur:
            print(f"Attempt {essai}/{args.essais} failed: {erreur}")
            time.sleep(20)
    else:
        raise SystemExit(f"API unreachable after {args.essais} attempts: {args.url}")

    texte = rapport_markdown(rapport, args.url)
    Path(args.sortie).write_text(texte, encoding="utf-8")
    print(texte)
    sortie_github = os.environ.get("GITHUB_OUTPUT")
    if sortie_github:
        with open(sortie_github, "a", encoding="utf-8") as fichier:
            fichier.write(f"status={rapport['status']}\n")


if __name__ == "__main__":
    main()
