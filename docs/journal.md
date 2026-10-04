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
