"""Préparer X (features) et y (label) pour les modèles de machine learning."""

STATS = ("mean", "std", "max")
PHASES = ("montee", "croisiere", "descente")


def colonnes_features(vols):
    """Les colonnes '<capteur>_<stat>_<phase>' : uniquement ce qu'un vrai avion mesure."""
    return [
        c
        for c in vols.columns
        if c.count("_") == 2 and c.split("_")[1] in STATS and c.split("_")[2] in PHASES
    ]


def preparer_xy(vols, groupe):
    """X = features du groupe ('train', 'val' ou 'test') ; y = 1 si le vol est usé (hs = 0)."""
    sous = vols[vols["groupe"] == groupe]
    X = sous[colonnes_features(vols)]
    y = (1 - sous["hs"]).astype(int)
    return X, y
