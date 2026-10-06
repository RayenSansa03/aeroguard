import pandas as pd

from aeroguard.metier import evaluer_alerte


def test_evaluer_alerte_deux_moteurs():
    infos = pd.DataFrame(
        {
            "moteur": ["A"] * 4 + ["B"] * 4,
            "cycle": [1, 2, 3, 4] * 2,
            "RUL": [3, 2, 1, 0] * 2,
        }
    )
    y = [0, 0, 1, 1] * 2
    # A : alerte sur ses 2 vols usés ; B : une fausse alarme, puis rien
    probas = [0.1, 0.2, 0.9, 0.95, 0.6, 0.1, 0.2, 0.3]

    bilan = evaluer_alerte(y, probas, seuil=0.5, infos=infos, k=1)

    assert (bilan["vp"], bilan["fp"], bilan["fn"]) == (2, 1, 2)
    assert bilan["moteurs"] == 2
    assert bilan["moteurs_detectes"] == 1  # B n'est jamais prévenu
    assert bilan["avance_moyenne"] == 1.0  # A : 1re alerte au RUL 1
    assert bilan["fausses_alarmes_total"] == 1
