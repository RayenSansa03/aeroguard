# Journal de bord AeroGuard

- **Fait :** installation de VS Code, Python et Git ; création du repo ; structure du projet.
- **Appris :** les 4 zones de Git (dossier, préparation, historique, GitHub), le rôle du .gitignore.
- **Bloquant :** rien.

#####

- **Fait :** fonctions de détection (score z, état moteur) et simulateur de flotte.
- **Appris :** pyproject.toml, pip install -e, np.clip, pd.concat, le bloc if **name** == "**main**".
- **Bloquant :** rien.

#####

- **Fait :** tests pytest, branches et pull requests, tag v0, Docker (testé dans Codespaces), CI GitHub Actions, README final.
- **Appris :** TDD, cycle branche → PR → merge, Dockerfile (image / conteneur), intégration continue.
- **Bloquant :** rien.

#####

#####

- **Fait :** chargement du dataset NASA DS01 (HDF5) dans un DataFrame avec Colab + Drive, exploration (6 moteurs dev, 4 test), sauvegarde en parquet.
- **Appris :** h5py, np.hstack, groupby / agg, float32 pour économiser la mémoire ; T48 dépend de la phase du vol (maximale en montée).
- **Bloquant :** erreur h5py sur les noms de colonnes (bytes) → résolue avec .decode().

#####

- **Fait :** vérification des étiquettes (RUL, hs), fonction composant_touche et table_vols dans data.py + 5 tests.
- **Appris :** cycle + RUL constant, hs ne remonte jamais ; durée de vie 75 à 100 vols ; panne qui commence tôt (vols 19 à 38) ; seule l'efficacité HPT se dégrade ; ne jamais utiliser les colonnes T en entrée (fuite de données).
- **Bloquant :** FileNotFoundError avec savefig (dossier manquant) → résolu avec os.makedirs(..., exist_ok=True).

#####

- **Fait :** normalisation par les conditions de vol (modèle du moteur sain + résidus) dans normalisation.py + 3 tests (21 passed).
- **Appris :** résidu = mesuré − attendu ; régression polynomiale avec scikit-learn ; entraîner seulement sur les 15 premiers vols ; bruit divisé par 35 à 57 ; capteurs les plus sensibles : T50, T48, T24, Nc, Wf ; panne visible seulement vers les vols 40-50.
- **Bloquant :** rien.

#####

- **Fait :** détection par seuil sur les résidus (score z + confirmation sur k vols), fonctions confirmer_alarmes et premiere_alarme + 5 tests (26 passed), évaluation dev puis test.
- **Appris :** TDD (rouge → vert), installer mon package depuis GitHub dans Colab, compromis seuil / k, ne jamais régler sur test. Baseline : avance 42,7 vols (dev), 38,2 vols (test), 0 fausse alarme, 0 panne ratée.
- **Bloquant :** import mal indenté dans les tests → corrigé (imports toujours en haut, collés à gauche).

#####

- **Fait :** découpage de chaque vol en 3 phases (montée, croisière, descente) et 126 features par vol (mean, std, max des résidus) dans features.py + 3 tests (29 passed). DS01 : 553 vols dev, 341 vols test.
- **Appris :** groupby + agg + unstack, aplatir les noms de colonnes, pd.Categorical, crosstab. La montée (pleine puissance) montre le mieux la panne : T48_mean_montee score 40,8 contre 31,2 pour le résidu T48 global.
- **Bloquant :** Colab ne trouvait pas aeroguard.features → la PR n'était pas encore mergée dans main.

#####

- **Fait :** découpage train / validation / test par moteur dans decoupage.py + 4 tests (33 passed). DS01 : train moteurs 2, 3, 4, 6 (364 vols), validation 1, 5 (189 vols), test 7 à 10 (341 vols). Découpage sauvegardé dans decoupage_ds01.json.
- **Appris :** validation = régler (bac blanc), test = note finale (une seule fois). Un moteur entier dans un seul groupe. Expérience du copieur (1 plus proche voisin) : pas d'effet de fuite visible sur DS01 (10,6 contre 10,1 vols), car tous les moteurs ont la même panne. Erreur de 4,3 vols en fin de vie : les features contiennent beaucoup d'information sur le RUL.
- **Bloquant :** Colab ne trouvait pas aeroguard.decoupage → merger la PR avant d'utiliser le package dans Colab.

#####

- **Fait :** pipeline complet `preparer()` appliqué à 9 fichiers N-CMAPSS (DS01 à DS08c) : 7 473 vols, 99 moteurs (49 train, 11 val, 39 test), sans fuite. DS01 identique aux leçons précédentes (553 / 341 vols). Tag v1.
- **Appris :** traiter des fichiers trop gros un par un (extraire → traiter → supprimer), points de sauvegarde, noms de moteurs uniques. 11 modes détaillés regroupés en 7 familles. Signatures : compresseurs → T30 et Nc montent ; turbines → T48/T50 montent, Nc baisse ; fan → P15/P21/P24 baissent.
- **Bloquant :** DS08d est tronqué dans le zip officiel NASA (CRC correct mais données incomplètes) → exclu et documenté.

#####

- **Fait :** leçon D1 — ruff (lint + format), pre-commit (dont blocage des fichiers > 5 Mo), couverture pytest-cov + Codecov, CI en 3 jobs (lint, tests, docker), badges, protection de main (PR + 3 checks obligatoires), .gitattributes pour les notebooks.
- **Appris :** linter vs formateur, hooks pre-commit, couverture (Stmts / Miss / Missing), secrets GitHub, needs entre jobs, git rm --cached, git reset --hard origin/main. Couverture : 78 % → 97 %.
- **Bloquant :** la couverture a révélé que tests/test_detection.py contenait le code de detection.py (copier-coller de la leçon 1.6) : 12 tests perdus sans alerte, restaurés. Commits annulés par les hooks → toujours vérifier la ligne « X files changed ».

#####

- **Fait :** préparation de X (126 features) et y (1 = usé) dans donnees_ml.py, avec un test anti-fuite. Premier modèle : modèle bête (DummyClassifier) et copieur (KNN, 5 voisins).
- **Appris :** fit / predict, classification vs régression, classe positive, modèle de référence. Train : 3 716 vols, 70,5 % usés. Modèle bête : 70,2 % (répond toujours « usé »). Copieur : ≈ 80 % (sains 69 %, usés 84 %). Erreurs : fausses alarmes sur moteurs neufs et hésitation au début de la panne (dégradation encore invisible).
- **Bloquant :** Colab ne trouvait pas aeroguard.donnees_ml → merger la PR avant Colab.

#####

- **Fait :** régression logistique (StandardScaler + LogisticRegression), probabilités avec predict_proba, effet du seuil de décision, poids des features. evaluation.py : appliquer_seuil, taux_par_classe, ajouter_au_leaderboard + 5 tests. Premier leaderboard (bête, copieur KNN, logistique).
- **Appris :** sigmoïde (score → probabilité), seuil bas = plus de détections mais plus de fausses alarmes, features corrélées = poids à interpréter avec prudence. Logistique : exactitude […], usés détectés […], sains reconnus […]. Le modèle bête a 100 % d'usés détectés mais 0 % de sains reconnus : un seul chiffre ne suffit jamais.
- **Bloquant :** ruff format annulait le commit → lancer `ruff format src tests` avant de commiter.

#####

**Fait :**

- Fonction `metriques()` dans `evaluation.py` : matrice de confusion, précision, rappel, F1, F1 macro, PR-AUC
- 10 tests dans `test_evaluation.py`, ruff et pytest au vert
- Notebook AeroGuard_23_metriques : matrices, F1 macro des 3 modèles, courbes PR, leaderboard

**Appris :**

- Précision = « quand je sonne, ai-je raison ? » ; rappel = « ai-je trouvé tous les usés ? »
- Le modèle bête a un F1 ≈ 0,82 mais un F1 macro ≈ 0,41 : le F1 macro démasque le tricheur
- La PR-AUC du hasard = part des positifs (≈ 0,70 ici), pas 0,5
- Pour AeroGuard, rater une panne est plus grave qu'une fausse alarme → privilégier le rappel

**Bloquant :**

- Erreurs ruff E402/F811 : imports des tests collés dans src → un fichier src ne s'importe jamais lui-même

#####

**Fait :**

- Fonction `poids_classes()` dans `donnees_ml.py` + 6 tests (dont comparaison avec sklearn)
- Notebook AeroGuard_24_desequilibre : logistique avec et sans class_weight="balanced"
- docs/baselines.md rédigé

**Appris :**

- Chez nous la classe rare est « sain » ; le modèle bête a 70 % d'exactitude et 0 % de sains reconnus
- balanced : FP 39 → 13 mais FN 39 → 73, F1 macro 0,886 → 0,882 → on ne le garde pas
- Pondérer déplace surtout le seuil : la PR-AUC ne bouge pas

**Bloquant :**

- FP/FN absents du leaderboard pour la ligne balanced → ajouter "fp" et "fn" aux clés enregistrées
