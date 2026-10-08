# Baselines — module 2

## Données

- Train : 3 716 vols, 70,5 % usés, 29,5 % sains
- Validation : 819 vols, 575 usés, 244 sains
- La classe rare est « sain » (les moteurs volent jusqu'à la panne)

## Le piège du déséquilibre

- Modèle bête : exactitude 70,2 %, sains reconnus 0 %, F1 macro 0,412
- Conclusion : l'exactitude seule est trompeuse → métrique principale = F1 macro

## Pondération des classes (régression logistique, validation)

Poids balanced : sain 1,695 ; usé 0,709

| Modèle              | Sains reconnus | Rappel | FP  | FN  | F1 macro | PR-AUC |
| ------------------- | -------------- | ------ | --- | --- | -------- | ------ |
| logistique          | 84,0 %         | 0,932  | 39  | 39  | 0,886    | 0,987  |
| logistique_balanced | 94,7 %         | 0,873  | 13  | 73  | 0,882    | 0,987  |

## Décision

- On garde la logistique SANS pondération : la pondération retire 26 fausses
  alarmes mais ajoute 34 pannes ratées, et le F1 macro ne s'améliore pas.
  Pour AeroGuard, rater une panne coûte plus cher qu'une fausse alarme.
- La PR-AUC est identique : pondérer change surtout le seuil, pas l'ordre des vols.

## Modèles du module 2 (validation, 819 vols)

| Modèle               | F1 macro | PR-AUC | Rappel | FP  | FN  |
| -------------------- | -------- | ------ | ------ | --- | --- |
| logistique           | 0,886    | 0,987  | 0,932  | 39  | 39  |
| logistique_balanced  | 0,882    | 0,987  | 0,873  | 13  | 73  |
| random_forest        | 0,844    | 0,985  | 0,951  | 73  | 28  |
| copieur_knn          | 0,763    | 0,903  | 0,843  | 75  | 90  |
| isolation_forest_p95 | 0,612    | 0,915  | 0,485  | 20  | 296 |
| bete                 | 0,412    | 0,702  | 1,000  | 244 | 0   |

## Métriques métier (11 moteurs de validation)

| Modèle               | k   | Moteurs détectés | Avance moyenne | Fausses alarmes |
| -------------------- | --- | ---------------- | -------------- | --------------- |
| random_forest        | 1   | 11 / 11          | 51,0           | 73              |
| logistique           | 1   | 11 / 11          | 49,5           | 39              |
| random_forest        | 3   | 11 / 11          | 48,1           | 31              |
| logistique           | 3   | 11 / 11          | 46,1           | 9               |
| isolation_forest_p90 | 1   | 11 / 11          | 42,2           | 29              |
| isolation_forest_p90 | 3   | 11 / 11          | 24,8           | 3               |

## Conclusion du module 2

- Modèle de référence à battre au module 3 : **logistique, confirmation k = 3**
  (F1 macro 0,886 ; 11/11 moteurs ; 46,1 vols d'avance ; 9 fausses alarmes).
- Le non supervisé (Isolation Forest) prévient tous les moteurs mais tard (24,8 vols avec k = 3).
- Points à creuser : moteurs prévenus tard (DS03_4, DS08c_5) et pannes hors turbines.

## Limite connue : famille de panne et fichier NASA

- Chaque famille de panne provient d'un seul fichier N-CMAPSS (DS01 → HPT, DS04 → fan, etc.).
- Depuis les vols SAINS, une logistique devine le fichier à 56,5 % (hasard 18 %) : les features portent une signature du fichier.
- Le diagnostic de famille peut donc s'appuyer en partie sur le fichier plutôt que sur la panne ; ses scores sont probablement optimistes pour une flotte nouvelle.
- Vérification prévue : SHAP (leçon 3.5).
- SHAP (leçon 3.5) : le diagnostic s'appuie surtout sur les capteurs physiques (P24, T30, Nc, T50),
  sans ressemblance globale avec le détecteur de fichier (Spearman 0,13), mais il utilise P2
  (pression d'entrée, n° 1 du détecteur de fichier). Piste : réentraîner sans les features de P2.

## Module 3 — XGBoost sur GPU (v3) : évaluation finale sur le test (39 moteurs)

Configuration figée sur la validation, évaluée une seule fois sur le test.

### Alerte (k = 3)

| Modèle                   | Moteurs prévenus | Avance moyenne | Fausses alarmes    | F1 macro | PR-AUC |
| ------------------------ | ---------------- | -------------- | ------------------ | -------- | ------ |
| Logistique (seuil 0,5)   | 39 / 39          | 44,7 vols      | 54 (12 moteurs)    | 0,840    | 0,978  |
| **XGBoost (seuil 0,81)** | **39 / 39**      | 40,1 vols      | **28 (4 moteurs)** | 0,852    | 0,978  |

Modèle d'alerte retenu : **XGBoost, seuil 0,81, k = 3** (fausses alarmes divisées par 2 pour 4,6 vols d'avance en moins).

### Diagnostic de la famille de panne

| Modèle           | F1 macro (par vol) | Moteurs bien diagnostiqués |
| ---------------- | ------------------ | -------------------------- |
| Logistique       | 0,846              | 35 / 39                    |
| XGBoost (Optuna) | 0,826              | 37 / 39                    |

### Familles non vues à l'entraînement

- L'alerte détecte encore tous les moteurs pour 6 familles sur 7 ; exception : le fan (rappel 0,78 → 0,22, 2 moteurs sur 4 non prévenus), seule famille signalée par des pressions et non des températures.
- Le diagnostic prédit alors une famille voisine avec ~90 % de confiance (fan → mixte, 92 %) : il ne sait pas dire « inconnu » → motivation de l'autoencodeur (module 5).

### Vitesse GPU (371 600 vols, 300 arbres)

- Entraînement : 12,4 s (GPU T4) contre 52,8 s (CPU), soit ×4,3.
- Prédiction : 1,74 s (GPU) contre 1,47 s (CPU) : la copie des données CPU → GPU coûte plus que le calcul.

### Remarques

- 30 vols du test (moteur DS02_14) ont des valeurs manquantes (phase de montée) : imputation par la médiane du train pour la logistique.
- Rappel de la limite connue : famille de panne = fichier NASA → scores de diagnostic probablement optimistes pour une flotte nouvelle.

## Module 4 — Deep learning (v4) : évaluation finale sur le test (39 moteurs)

Seuils figés sur la validation (relus dans MLflow), k = 3, bootstrap par moteur (1 000 tirages, graine 42).

| Modèle           | Seuil | F1 macro | IC 95 %         | Chute val → test | Moteurs prévenus | Avance | Fausses alarmes |
| ---------------- | ----- | -------- | --------------- | ---------------- | ---------------- | ------ | --------------- |
| **XGBoost**      | 0,81  | 0,852    | [0,827 ; 0,874] | −0,033           | 39 / 39          | 40,1   | 28 (4 moteurs)  |
| Logistique       | 0,50  | 0,840    | [0,814 ; 0,862] | −0,046           | 39 / 39          | 44,7   | 54 (12 moteurs) |
| Réseau dense (D) | 0,49  | 0,823    | [0,797 ; 0,845] | −0,049           | 39 / 39          | 41,3   | 51 (11 moteurs) |
| CNN 1D           | 0,62  | 0,743    | [0,682 ; 0,800] | +0,003           | 37 / 39          | 32,0   | 35 (6 moteurs)  |

- Écart à XGBoost : logistique −0,013 [−0,038 ; +0,011] (non prouvé, P(meilleur) = 15 %) ; dense −0,029 [−0,042 ; −0,015] ; CNN −0,108 [−0,164 ; −0,055].
- Par famille, le CNN égale XGBoost sur HPT+LPT (0,867 / 0,857) et mixte (0,857 / 0,851) mais s'effondre sur le fan (0,288 / 0,777).
- Coût : prédiction de 2 938 vols en 0,007 s (logistique), 0,023 s (XGBoost), 0,076 s (dense), 3,1 s (CNN).
- Décision : **XGBoost, seuil 0,81, k = 3 reste le modèle d'alerte.**
