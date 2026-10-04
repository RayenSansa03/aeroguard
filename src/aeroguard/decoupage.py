"""Découpage train / validation / test PAR MOTEUR (pas de fuite)."""


def decouper_par_moteur(vols, moteurs_val, moteurs_test=()):
    """Retourne (train, val, test). Un moteur n'est jamais dans deux groupes."""
    communs = set(moteurs_val) & set(moteurs_test)
    if communs:
        raise ValueError(f"Moteurs à la fois en validation et en test : {sorted(communs)}")
    est_val = vols["unit"].isin(moteurs_val)
    est_test = vols["unit"].isin(moteurs_test)
    return vols[~est_val & ~est_test], vols[est_val], vols[est_test]


def verifier_sans_fuite(*groupes):
    """Lève une erreur si un même moteur apparaît dans deux groupes."""
    vus = set()
    for groupe in groupes:
        moteurs = set(groupe["unit"])
        partages = vus & moteurs
        if partages:
            raise ValueError(f"Fuite ! Moteurs présents dans plusieurs groupes : {sorted(partages)}")
        vus |= moteurs
    return True