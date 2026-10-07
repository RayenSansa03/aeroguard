import numpy as np

CAPTEURS = (
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
)
CONDITIONS_VOL = ("alt", "Mach", "TRA", "T2")


def _verifier_signal(signal):
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError("signal doit avoir 2 dimensions : (temps, canaux).")
    return signal


def decouper_fenetres(signal, longueur, pas):
    """Fenêtres glissantes : (temps, canaux) → (nombre, longueur, canaux)."""
    signal = _verifier_signal(signal)
    if longueur < 1 or pas < 1:
        raise ValueError("longueur et pas doivent être supérieurs ou égaux à 1.")
    if len(signal) < longueur:
        return np.empty((0, longueur, signal.shape[1]), dtype=signal.dtype)
    vues = np.lib.stride_tricks.sliding_window_view(signal, longueur, axis=0)
    return np.ascontiguousarray(vues[::pas].transpose(0, 2, 1))


def extraire_fenetres(signal, longueur, nombre):
    """`nombre` fenêtres régulièrement espacées, du début à la fin du vol."""
    signal = _verifier_signal(signal)
    if longueur < 1 or nombre < 1:
        raise ValueError("longueur et nombre doivent être supérieurs ou égaux à 1.")
    if len(signal) < longueur:
        return np.empty((0, longueur, signal.shape[1]), dtype=signal.dtype)
    debuts = np.linspace(0, len(signal) - longueur, nombre).round().astype(int)
    return np.stack([signal[debut : debut + longueur] for debut in debuts])


def statistiques_canaux(fenetres):
    """Moyenne et écart-type de chaque canal, sur toutes les fenêtres et tous les instants."""
    fenetres = np.asarray(fenetres)
    moyennes = fenetres.mean(axis=(0, 1), dtype="float64")
    ecarts = fenetres.std(axis=(0, 1), dtype="float64")
    ecarts = np.where(ecarts == 0, 1.0, ecarts)
    return moyennes, ecarts


def normaliser_fenetres(fenetres, moyennes, ecarts):
    """Centre-réduit chaque canal avec les statistiques du train."""
    return ((np.asarray(fenetres) - moyennes) / ecarts).astype("float32")


def creer_dataset(X, y, taille_lot=256, melanger=True, graine=42):
    """Pipeline tf.data : (mélange) → lots → préchargement."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if melanger:
        dataset = dataset.shuffle(min(len(X), 10_000), seed=graine)
    return dataset.batch(taille_lot).prefetch(tf.data.AUTOTUNE)
