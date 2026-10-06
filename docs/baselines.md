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
