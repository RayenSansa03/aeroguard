import time
from aeroguard.outils import chrono


def test_chrono_mesure_un_temps_positif():
    assert chrono(lambda: sum(range(1000))) >= 0


def test_chrono_garde_le_meilleur_temps():
    attentes = iter([0.05, 0.01, 0.03])          # 3 essais de durées différentes
    t = chrono(lambda: time.sleep(next(attentes)), repetitions=3)
    assert 0.01 <= t < 0.03                       # le plus court : ~0,01 s