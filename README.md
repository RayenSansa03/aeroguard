# AeroGuard

Système de détection d'anomalies pour une flotte de moteurs d'avion, construit sur les données NASA N-CMAPSS et accéléré par GPU (NVIDIA RAPIDS, XGBoost, TensorFlow).

## Objectifs
- Détecter le début d'une panne le plus tôt possible
- Identifier le composant touché (fan, compresseur, turbine)
- Repérer les pannes jamais vues à l'entraînement

## Structure du projet
```
data/            données brutes et traitées (non versionnées)
notebooks/       exploration et analyses
src/aeroguard/   code réutilisable (détection, simulation)
tests/           tests automatiques (pytest)
results/         graphiques et résultats
docs/            documentation et journal de bord
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
pip install -e .
```

## Utilisation
```bash
python -m aeroguard.simulation    # générer une flotte simulée
pytest -v                         # lancer les tests
```

## Premiers résultats : baseline par score z (flotte simulée)
Le « normal » est appris sur les vols sains de 5 moteurs ; la détection est testée sur 5 autres moteurs, jamais vus.

| Seuil z | Retard moyen de détection | Avance d'alerte moyenne | Fausses alarmes |
|---|---|---|---|
| 2,5 | 11,4 vols | 28,8 vols | 3 |
| **3** | **13,4 vols** | **26,8 vols** | **1** |
| 4 | 18,8 vols | 21,4 vols | 0 |

Cette baseline sert de référence : les modèles suivants (XGBoost, autoencodeur) devront détecter l'usure plus tôt.

## Statut
🚧 Module 0 terminé : environnement, détection par score z, simulateur, tests.
Prochaine étape : données NASA N-CMAPSS.