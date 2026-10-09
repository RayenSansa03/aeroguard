import pandas as pd

from aeroguard.metier import evaluer_alerte, evaluer_alerte_score


def flotte():
    infos = pd.DataFrame(
        {
            "moteur": ["A"] * 4 + ["B"] * 4,
            "cycle": [1, 2, 3, 4] * 2,
            "RUL": [3, 2, 1, 0] * 2,
        }
    )
    y = [0, 0, 1, 1] * 2
    return infos, y


def test_scores_superieurs_a_1_acceptes():
    infos, y = flotte()
    # Erreurs de reconstruction : A monte à 40 et 90 ; B a un pic isolé (fausse alarme)
    scores = [0.2, 0.3, 40.0, 90.0, 12.0, 0.2, 0.3, 0.4]

    bilan = evaluer_alerte_score(y, scores, seuil=5.0, infos=infos, k=1)

    assert (bilan["vp"], bilan["fp"], bilan["fn"]) == (2, 1, 2)
    assert bilan["moteurs_detectes"] == 1
    assert bilan["avance_moyenne"] == 1.0
    assert bilan["fausses_alarmes_total"] == 1
    assert 0 <= bilan["pr_auc"] <= 1


def test_confirmation_k2_supprime_le_pic_isole():
    infos, y = flotte()
    scores = [0.2, 0.3, 40.0, 90.0, 12.0, 0.2, 0.3, 0.4]
    bilan = evaluer_alerte_score(y, scores, seuil=5.0, infos=infos, k=2)
    assert bilan["fausses_alarmes_total"] == 0  # le pic de B n'est pas confirmé
    assert bilan["avance_moyenne"] == 0.0  # A : alerte confirmée au 2e vol au-dessus (RUL 0)


def test_meme_resultat_que_evaluer_alerte_pour_des_probas():
    infos, y = flotte()
    probas = [0.1, 0.2, 0.9, 0.95, 0.6, 0.1, 0.2, 0.3]
    attendu = evaluer_alerte(y, probas, seuil=0.5, infos=infos, k=1)
    assert evaluer_alerte_score(y, probas, seuil=0.5, infos=infos, k=1) == attendu
