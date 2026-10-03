"""Normalisation : résidu = capteur mesuré − capteur attendu pour un moteur sain."""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

W_COLS = ["alt", "Mach", "TRA", "T2"]
CAPTEURS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
            "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
COLONNES_ID = ["unit", "cycle", "hs", "RUL"]


def ajuster_modele_sain(df, colonnes_w=W_COLS, colonnes_capteurs=CAPTEURS,
                        n_vols_sains=15, degre=2, n_max=300_000, seed=42):
    """Apprend 'capteurs attendus' à partir des conditions de vol, sur les premiers vols seulement."""
    sains = df[df["cycle"] <= n_vols_sains]
    if len(sains) > n_max:
        sains = sains.sample(n_max, random_state=seed)
    modele = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degre), LinearRegression())
    modele.fit(sains[colonnes_w], sains[colonnes_capteurs])
    return modele


def calculer_residus(df, modele, colonnes_w=W_COLS, colonnes_capteurs=CAPTEURS):
    """Retourne un DataFrame avec les colonnes d'identité + résidu de chaque capteur."""
    attendu = modele.predict(df[colonnes_w])
    residus = df[colonnes_capteurs].to_numpy() - attendu
    sortie = df[[c for c in COLONNES_ID if c in df.columns]].copy()
    sortie[colonnes_capteurs] = residus
    return sortie