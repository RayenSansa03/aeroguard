# Journal de bord AeroGuard

---

- **Fait :** installation de VS Code, Python et Git ; création du repo ; structure du projet.
- **Appris :** les 4 zones de Git (dossier, préparation, historique, GitHub), le rôle du .gitignore.
- **Bloquant :** rien.

---

- **Fait :** fonctions de détection (score z, état moteur) et simulateur de flotte.
- **Appris :** pyproject.toml, pip install -e, np.clip, pd.concat, le bloc `if __name__ == "__main__"`.
- **Bloquant :** rien.

---

- **Fait :** tests pytest, branches et pull requests, tag v0, Docker (testé dans Codespaces), CI GitHub Actions, README final.
- **Appris :** TDD, cycle branche → PR → merge, Dockerfile (image / conteneur), intégration continue.
- **Bloquant :** rien.

---

- **Fait :** chargement du dataset NASA DS01 (HDF5) dans un DataFrame avec Colab + Drive, exploration (6 moteurs dev, 4 test), sauvegarde en parquet.
- **Appris :** h5py, np.hstack, groupby / agg, float32 pour économiser la mémoire ; T48 dépend de la phase du vol (maximale en montée).
- **Bloquant :** erreur h5py sur les noms de colonnes (bytes) → résolue avec .decode().

---

- **Fait :** vérification des étiquettes (RUL, hs), fonction composant_touche et table_vols dans data.py + 5 tests.
- **Appris :** cycle + RUL constant, hs ne remonte jamais ; durée de vie 75 à 100 vols ; panne qui commence tôt (vols 19 à 38) ; seule l'efficacité HPT se dégrade ; ne jamais utiliser les colonnes T en entrée (fuite de données).
- **Bloquant :** FileNotFoundError avec savefig (dossier manquant) → résolu avec os.makedirs(..., exist_ok=True).

---

- **Fait :** normalisation par les conditions de vol (modèle du moteur sain + résidus) dans normalisation.py + 3 tests (21 passed).
- **Appris :** résidu = mesuré − attendu ; régression polynomiale avec scikit-learn ; entraîner seulement sur les 15 premiers vols ; bruit divisé par 35 à 57 ; capteurs les plus sensibles : T50, T48, T24, Nc, Wf ; panne visible seulement vers les vols 40-50.
- **Bloquant :** rien.

---

- **Fait :** détection par seuil sur les résidus (score z + confirmation sur k vols), fonctions confirmer_alarmes et premiere_alarme + 5 tests (26 passed), évaluation dev puis test.
- **Appris :** TDD (rouge → vert), installer mon package depuis GitHub dans Colab, compromis seuil / k, ne jamais régler sur test. Baseline : avance 42,7 vols (dev), 38,2 vols (test), 0 fausse alarme, 0 panne ratée.
- **Bloquant :** import mal indenté dans les tests → corrigé (imports toujours en haut, collés à gauche).

---

- **Fait :** découpage de chaque vol en 3 phases (montée, croisière, descente) et 126 features par vol (mean, std, max des résidus) dans features.py + 3 tests (29 passed). DS01 : 553 vols dev, 341 vols test.
- **Appris :** groupby + agg + unstack, aplatir les noms de colonnes, pd.Categorical, crosstab. La montée (pleine puissance) montre le mieux la panne : T48_mean_montee score 40,8 contre 31,2 pour le résidu T48 global.
- **Bloquant :** Colab ne trouvait pas aeroguard.features → la PR n'était pas encore mergée dans main.

---

- **Fait :** découpage train / validation / test par moteur dans decoupage.py + 4 tests (33 passed). DS01 : train moteurs 2, 3, 4, 6 (364 vols), validation 1, 5 (189 vols), test 7 à 10 (341 vols). Découpage sauvegardé dans decoupage_ds01.json.
- **Appris :** validation = régler (bac blanc), test = note finale (une seule fois). Un moteur entier dans un seul groupe. Expérience du copieur (1 plus proche voisin) : pas d'effet de fuite visible sur DS01 (10,6 contre 10,1 vols), car tous les moteurs ont la même panne. Erreur de 4,3 vols en fin de vie : les features contiennent beaucoup d'information sur le RUL.
- **Bloquant :** Colab ne trouvait pas aeroguard.decoupage → merger la PR avant d'utiliser le package dans Colab.

---

- **Fait :** pipeline complet preparer() appliqué à 9 fichiers N-CMAPSS (DS01 à DS08c) : 7 473 vols, 99 moteurs (49 train, 11 val, 39 test), sans fuite. DS01 identique aux étapes précédentes (553 / 341 vols). Tag v1.
- **Appris :** traiter des fichiers trop gros un par un (extraire → traiter → supprimer), points de sauvegarde, noms de moteurs uniques. 11 modes détaillés regroupés en 7 familles. Signatures : compresseurs → T30 et Nc montent ; turbines → T48/T50 montent, Nc baisse ; fan → P15/P21/P24 baissent.
- **Bloquant :** DS08d est tronqué dans le zip officiel NASA (CRC correct mais données incomplètes) → exclu et documenté.

---

- **Fait :** ruff (lint + format), pre-commit (dont blocage des fichiers > 5 Mo), couverture pytest-cov + Codecov, CI en 3 jobs (lint, tests, docker), badges, protection de main (PR + 3 checks obligatoires), .gitattributes pour les notebooks.
- **Appris :** linter vs formateur, hooks pre-commit, couverture (Stmts / Miss / Missing), secrets GitHub, needs entre jobs, git rm --cached, git reset --hard origin/main. Couverture : 78 % → 97 %.
- **Bloquant :** la couverture a révélé que tests/test_detection.py contenait le code de detection.py (copier-coller) : 12 tests perdus sans alerte, restaurés. Commits annulés par les hooks → toujours vérifier la ligne « X files changed ».

---

- **Fait :** préparation de X (126 features) et y (1 = usé) dans donnees_ml.py, avec un test anti-fuite. Premiers modèles : modèle bête (DummyClassifier) et copieur (KNN, 5 voisins).
- **Appris :** fit / predict, classification vs régression, classe positive, modèle de référence. Train : 3 716 vols, 70,5 % usés. Modèle bête : 70,2 % (répond toujours « usé »). Copieur : ≈ 80 % (sains 69 %, usés 84 %). Erreurs : fausses alarmes sur moteurs neufs et hésitation au début de la panne (dégradation encore invisible).
- **Bloquant :** Colab ne trouvait pas aeroguard.donnees_ml → merger la PR avant Colab.

---

- **Fait :** régression logistique (StandardScaler + LogisticRegression), probabilités avec predict_proba, effet du seuil de décision, poids des features. evaluation.py : appliquer_seuil, taux_par_classe, ajouter_au_leaderboard + 5 tests. Premier leaderboard (bête, copieur KNN, logistique).
- **Appris :** sigmoïde (score → probabilité), seuil bas = plus de détections mais plus de fausses alarmes, features corrélées = poids à interpréter avec prudence. Logistique : exactitude 90,5 %, usés détectés 93,2 %, sains reconnus 84,0 %. Le modèle bête a 100 % d'usés détectés mais 0 % de sains reconnus : un seul chiffre ne suffit jamais.
- **Bloquant :** ruff format annulait le commit → lancer ruff format src tests avant de commiter.

---

- **Fait :** fonction metriques() dans evaluation.py (matrice de confusion, précision, rappel, F1, F1 macro, PR-AUC) + 10 tests ; notebook AeroGuard_23_metriques (matrices, F1 macro des 3 modèles, courbes PR, leaderboard).
- **Appris :** précision = « quand je sonne, ai-je raison ? » ; rappel = « ai-je trouvé tous les usés ? ». Le modèle bête a un F1 ≈ 0,82 mais un F1 macro ≈ 0,41 : le F1 macro démasque le tricheur. La PR-AUC du hasard = part des positifs (≈ 0,70 ici), pas 0,5. Pour AeroGuard, rater une panne est plus grave qu'une fausse alarme → privilégier le rappel.
- **Bloquant :** erreurs ruff E402/F811 : imports des tests collés dans src → un fichier src ne s'importe jamais lui-même.

---

- **Fait :** fonction poids_classes() dans donnees_ml.py + 6 tests (dont comparaison avec sklearn) ; notebook AeroGuard_24_desequilibre (logistique avec et sans class_weight="balanced") ; docs/baselines.md rédigé.
- **Appris :** chez nous la classe rare est « sain » ; le modèle bête a 70 % d'exactitude et 0 % de sains reconnus. Balanced : FP 39 → 13 mais FN 39 → 73, F1 macro 0,886 → 0,882 → on ne le garde pas. Pondérer déplace surtout le seuil : la PR-AUC ne bouge pas.
- **Bloquant :** FP/FN absents du leaderboard pour la ligne balanced → ajouter "fp" et "fn" aux clés enregistrées.

---

- **Fait :** module explication.py (decrire_feature, importances_triees) + 5 tests ; notebook AeroGuard_25_random_forest : arbre de profondeur 3, courbe de surapprentissage, Random Forest 200 arbres, top 10 des features, leaderboard.
- **Appris :** un arbre = suite de questions oui/non ; 1re question = T48_mean_montee <= 1,255 (turbines), puis P21 (fan). Surapprentissage : profondeur None → train 1,000 mais val 0,787 ; meilleur arbre seul = profondeur 2 (val 0,851). Random Forest : F1 macro 0,844 < logistique 0,886 ; FN 28 (au lieu de 39) mais FP 73 (au lieu de 39) ; PR-AUC presque égale (0,985) → le problème vient surtout du seuil. Top features : T48 et T50 (turbines), puis T24 et Nc. Pas besoin de StandardScaler pour les arbres.
- **Bloquant :** rien.

---

- **Fait :** module anomalies.py (score_anomalie, seuil_percentile, alertes) + 5 tests avec un faux modèle ; notebook AeroGuard_26_isolation_forest : entraînement sur les 1 096 vols sains du train, histogramme des scores, 3 seuils (90, 95, 99), leaderboard.
- **Appris :** non supervisé = apprendre la normale, pas la panne ; une anomalie s'isole en peu de coupures ; règle d'or : entraîner sur les sains uniquement. Histogrammes sains/usés très chevauchés (début de panne invisible). p90 : FP 29, FN 241, F1 macro 0,663 ; p99 : FP 2, FN 444. Rappel 0,485 (p95) contre 0,951 pour la Random Forest ; PR-AUC 0,915 identique pour tous les seuils. Utile pour les pannes jamais vues → motivation de l'autoencodeur (module 5).
- **Bloquant :** rien. Piège retenu : score_samples renvoie « plus grand = plus normal » → on inverse le signe.

---

- **Fait :** module metier.py (alertes_confirmees, avance_alerte, fausses_alarmes, bilan_flotte, resume_flotte) + 7 tests ; notebook AeroGuard_27_metier : bilan par moteur des 3 meilleurs modèles avec k = 1 et k = 3, graphique d'avance par moteur, leaderboard_metier.csv.
- **Appris :** compter des moteurs, pas des vols : les 11 moteurs de validation sont détectés par tous les modèles, même l'Isolation Forest. Au classement par avance, la Random Forest passe devant (51,0 vols) mais avec 73 fausses alarmes. k = 3 : −3 vols d'avance et −77 % de fausses alarmes pour la logistique, mais −17 vols pour l'Isolation Forest (alertes intermittentes). Choix : logistique k = 3 → 11/11 moteurs, 46,1 vols d'avance, 9 fausses alarmes. Moteurs prévenus tard par tous : DS03_4, DS08c_5.
- **Bloquant :** colonne `unit` utilisée au lieu de `moteur` → moteurs de fichiers différents fusionnés (6 au lieu de 11). Corrigé + assert de contrôle. Leçon : toujours vérifier un chiffre connu avant d'interpréter.

---

- **Fait :** notebook AeroGuard_31_boosting : boosting fait main (arbres de profondeur 3 qui apprennent les erreurs), comparaison η = 0,1 / η = 1,0, test de HistGradientBoostingClassifier.
- **Appris :** forêt = arbres en parallèle qui votent ; boosting = petits arbres en série qui corrigent les erreurs restantes, prédiction = somme. Exemple à la main : erreur 40 → 20 → 10 → 5 avec η = 0,5. Avec 1 arbre, F1 macro = 0,412 (= modèle naïf) ; η = 0,1 plafonne à 0,850 vers 100 arbres puis baisse un peu ; η = 1,0 : erreur train 0,04 mais validation qui chute vers 0,79 → surapprentissage. HistGradientBoosting : F1 macro 0,846, PR-AUC 0,984, meilleur rappel (0,962, 22 pannes ratées) mais 76 fausses alarmes ; ne bat pas la logistique (0,886).
- **Bloquant :** rien.

---

- **Fait :** modeles.py (creer_xgboost, seuil_optimal) + 4 tests, xgboost ajouté aux dépendances ; notebook AeroGuard_32_xgboost_gpu : XGBoost sur GPU Tesla T4 avec early stopping, seuil optimal, métriques métier, modèle sauvegardé (models/xgb_binaire_v3.json).
- **Appris :** device="cuda" + tree_method="hist" ; entraînement en 2,0 s ; early stopping à 411 arbres sur 2 000 (PR-AUC val 0,9867, train 1,000 = par cœur). Seuil optimal 0,81 : F1 macro 0,865 → 0,885, FP 67 → 27, FN 20 → 54. XGBoost bat la Random Forest et égale la logistique (PR-AUC 0,987 tous les deux) : les résidus ont rendu le problème presque linéaire. Métier k = 3 : XGBoost 0,81 → 11/11 moteurs, 42,5 vols d'avance, seulement 2 fausses alarmes. Seuil choisi sur la validation = score optimiste → décision finale sur le test en 3.6.
- **Bloquant :** notebook d'abord lancé en CPU (nvidia-smi introuvable) → passer le type d'exécution en T4 GPU.

---

- **Fait :** preparer_xy_famille (donnees_ml.py) et diagnostic_par_moteur (metier.py) + 3 tests ; notebook AeroGuard_33_diagnostic : XGBoost multi-classe (multi:softprob) sur GPU contre logistique multi-classe, 7 familles, vols usés seulement, pondération des familles, vote par moteur.
- **Appris :** 7 familles, 5 à 12 moteurs par famille au train, 1 seul moteur en validation pour 5 familles → scores par famille fragiles. Surprise : la logistique gagne (F1 macro 0,829 contre 0,721 ; 11/11 moteurs contre 9/11). Confusions physiquement logiques : LPT ↔ HPT+LPT (43 % / 30 %) et HPC ↔ LPC+HPC (26 % / 21 %), jamais fan ↔ turbine ; fan et HPT reconnus à 100 %. Le vote par moteur corrige les erreurs dispersées, pas les confusions systématiques. Hypothèses : XGBoost surapprend avec trop peu de moteurs ; réglages non adaptés (→ 3.4) ; famille = fichier NASA → risque de raccourci (→ vérifier avec SHAP en 3.5).
- **Bloquant :** cellule de sauvegarde dupliquée par erreur → relancée.

---

- **Fait :** proposer_reglages_xgb (modeles.py) + 2 tests avec FixedTrial, optuna ajouté aux dépendances ; notebook AeroGuard_34_optuna : courbe max_depth, validation croisée groupée par moteur (StratifiedGroupKFold, 5 plis), recherche Optuna de 40 essais sur GPU (8,5 min), verdict sur la validation, enquête « raccourci fichier ».
- **Appris :** train à 1,000 dès max_depth 4, meilleure validation à max_depth 3 (0,743). CV : logistique 0,571 ± 0,090, XGBoost 0,521 ± 0,050 (notes CV basses car familles absentes de certains plis). Optuna : 0,530 (+0,009), arbres petits (profondeur 3) et feuilles grandes (min_child_weight 9,8), colsample_bytree = réglage le plus important. Validation : XGBoost réglé 0,726 contre logistique 0,829 (11/11) → aucun réglage ne compense le manque de moteurs. ⚠️ Le fichier NASA se devine à 56,5 % (hasard 18 %) à partir des vols sains : signature du fichier dans les features, et famille = fichier → le diagnostic peut prendre un raccourci. À vérifier avec SHAP (3.5) ; limite du dataset à documenter.
- **Bloquant :** commit annulé par end-of-file-fixer sur requirements.txt ; tests sautés (ss) car optuna absent du venv → pip install optuna.

---

- **Fait :** importance_par_capteur et expliquer_vol (explication.py) + 3 tests ; notebook AeroGuard_35_shap : SHAP calculé par XGBoost sur GPU (GPUTreeShap), explication globale et locale de l'alerte, carte des capteurs par famille (XGBoost et logistique), enquête raccourci (Spearman).
- **Appris :** SHAP sur GPU 0,17 s contre 1,99 s sur CPU (×11,7) ; départ + somme des SHAP = score (écart 1e-5). Alerte : T50 et T48 en montée dominent (signature turbine), même histoire que la Random Forest mais partage équitable. Alerte DS01_1 au vol 37 (RUL 63, 91 %) car T50 montée +0,77, T50 croisière +0,47, T48 montée +0,35. Diagnostic : T48 ≈ 0 car il ne distingue pas les familles de turbines ; la logistique est plus physique (fan = P15/P21/P24) que XGBoost (fan = T24 62 %). Enquête : Spearman 0,13 (pas de ressemblance globale) mais P2, capteur n° 1 du détecteur de fichier, est utilisé par le diagnostic → surtout physique, avec un raccourci identifié via P2.
- **Bloquant :** ImportError importance_par_capteur dans Colab → réinstaller le package puis redémarrer la session.

---

- **Fait :** evaluer_alerte (metier.py) + 1 test ; notebook AeroGuard_36_test_final : configuration figée, évaluation finale unique sur les 39 moteurs de test (alerte et diagnostic), familles de panne non vues à l'entraînement (14 modèles), test de charge CPU/GPU sur 371 600 vols.
- **Appris :** test final : 39/39 moteurs prévenus par les deux modèles ; logistique 44,7 vols d'avance et 54 fausses alarmes (12 moteurs) ; XGBoost 0,81 40,1 vols et 28 fausses alarmes (4 moteurs) → XGBoost retenu. Baisse modérée validation → test (F1 −0,03 à −0,05) = réglages sains. Diagnostic test : logistique 35/39, XGBoost 37/39 (la validation à 1 moteur par famille était trop petite). Famille non vue : l'alerte tient pour 6 familles sur 7 mais devient aveugle au fan (rappel 0,78 → 0,22, 2/4 moteurs) ; le diagnostic se trompe avec ~90 % de confiance (fan → mixte 92 %) → motivation de l'autoencodeur. GPU : entraînement ×4,3 ; prédiction plus lente (copie CPU → GPU).
- **Bloquant :** NaN dans 30 vols du test (moteur DS02_14, features de montée) → SimpleImputer(median) appris sur le train pour la logistique ; XGBoost gère les NaN nativement. Package absent après une nouvelle machine Colab → relancer la cellule d'installation.

---

- **Fait :** neurone artificiel codé en NumPy (somme pondérée, ReLU, sigmoïde), testé sur T48 et Wf en montée ; preuve qu'un neurone sigmoïde = une régression logistique.
- **Appris :** apprendre les poids fait passer le F1 macro de 0,656 (à la main) à 0,755 ; le neurone a appris « chauffer plus que ce que le carburant explique = usure » (poids T48 +8,6, Wf −3,1) ; une seule droite ne sépare pas la zone où sains et usés se mélangent.
- **Bloquant :** rien.

---

- **Fait :** descente de gradient codée en NumPy (sur une droite, puis sur le neurone T48 + Wf d'AeroGuard), rétropropagation vérifiée par différences finies, entraînement par lot complet et par mini-lots de 64.
- **Appris :** taux 0,001 trop lent, 0,1 à 0,5 idéal, 0,9 fait exploser la perte ; 59 mises à jour par époque en lots de 64 contre 1 en lot complet (perte 0,36 contre 0,48 après 50 époques) ; la log loss punit 460 fois plus une erreur sûre d'elle qu'une bonne réponse.
- **Bloquant :** rien.

---

- **Fait :** premier réseau dense Keras (126-64-32-1, 10 241 paramètres) dans `reseaux.py` avec normalisation intégrée, 9 tests ; entraîné sur GPU, comparé à XGBoost et à la logistique dans MLflow ; tests de 5 graines et de 3 architectures.
- **Appris :** le réseau prévient 11/11 moteurs avec 43,4 vols d'avance mais reste derrière en F1 macro (0,878 en moyenne contre 0,887 et 0,894) ; surapprentissage net après l'époque 17 ; le hasard de la graine fait varier le F1 de 0,020, plus que l'écart entre modèles : il faut comparer des moyennes sur plusieurs graines.
- **Bloquant :** rien.

---

- **Fait :** dropout et callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) ajoutés à `reseaux.py` avec 6 tests ; 4 configurations × 3 graines comparées et tracées dans MLflow.
- **Appris :** le dropout baisse la perte de validation de 0,290 à 0,272 et stabilise le seuil (0,36 → 0,51 au lieu de 0,39 → 0,92) ; EarlyStopping divise les époques par deux et garde la meilleure ; le F1 macro plafonne vers 0,88 quelle que soit la protection : la limite vient des features tabulaires, pas du surapprentissage.
- **Bloquant :** ImportError de `creer_callbacks` dans Colab, réglé en redémarrant la session après le merge.

---

- **Fait :** module `sequences.py` (fenêtres glissantes, normalisation par canal, pipeline tf.data) avec 10 tests ; script `scripts/preparer_fenetres.py` qui découpe les 9 fichiers NASA bruts en 29 892 fenêtres de 256 s × 18 canaux (0,55 Go), avec le découpage 49 / 11 / 39 de la v1.
- **Appris :** un pas de 1 donnerait ≈ 62 millions de fenêtres (≈ 1,1 To), d'où 4 fenêtres régulières par vol ; la normalisation se fait avec les statistiques du train seulement ; les conditions de vol sont ajoutées comme canaux pour séparer l'usure de la façon de voler ; sur DS01_1, T50 monte de 0,28 et Nc baisse de 0,20 entre le cycle 1 et le cycle 100.
- **Bloquant :** seul DS01 était sur le Drive ; l'archive NASA (15,8 Go) ne tenait pas dans le Drive (9 Go libres) ; réglé en traitant les 10 fichiers sur le PC avec un script, puis en envoyant seulement le résultat sur le Drive.

## Leçon 4.6 — Réseau convolutif 1D (CNN)

**Ce que j'ai appris :** un CNN 1D fait glisser des petits filtres sur le signal (partage des poids → 47 265 paramètres seulement). Conv1D + MaxPooling + GlobalAveragePooling, puis moyenne des 4 fenêtres par vol pour comparer avec XGBoost.

**Résultats (validation, 819 vols) :** F1 macro 0.740, PR-AUC 0.940, précision 0.947, rappel 0.683, 11/11 moteurs, 32 vols d'avance, 9 fausses alarmes. Meilleure époque 19/29.

**Classement :** logistique 0.894 > XGBoost 0.887 > dense 0.872 > CNN 0.740.

**Pourquoi le CNN perd :** il ne voit que ≈12 % du vol, les signaux bruts mélangent usure et conditions de vol (le filtre n°16 utilise T50 mais aussi alt et T2), et 49 moteurs c'est peu.

**Taille du noyau :** k = 3/7/15 → 0.743/0.740/0.746, aucun effet réel.

**Leçon :** le deep learning ne bat pas automatiquement de bonnes features. Pistes : plus de fenêtres par vol, fenêtres en croisière seulement, entrée en résidus.
