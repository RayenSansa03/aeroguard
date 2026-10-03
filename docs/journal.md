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
