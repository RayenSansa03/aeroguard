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
