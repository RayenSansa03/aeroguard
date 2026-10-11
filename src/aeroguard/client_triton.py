"""Client HTTP minimal pour Triton (protocole KServe v2, tenseurs envoyés en binaire)."""

import json

import httpx
import numpy as np


class ClientTriton:
    """Interroge les modèles AeroGuard servis par Triton."""

    def __init__(self, url="http://127.0.0.1:8000", client=None, timeout=30.0):
        self.client = client or httpx.Client(base_url=url, timeout=timeout)

    def pret(self, modele=None):
        """True si le serveur (ou un modèle précis) est prêt à répondre."""
        chemin = f"/v2/models/{modele}/ready" if modele else "/v2/health/ready"
        try:
            return self.client.get(chemin).status_code == 200
        except httpx.HTTPError:
            return False

    def inferer(self, modele, nom_entree, nom_sortie, X):
        """Envoie un lot de vols en binaire (les NaN passent tels quels) et lit la sortie."""
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float32)
        entete = json.dumps(
            {
                "inputs": [
                    {
                        "name": nom_entree,
                        "shape": list(X.shape),
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": X.nbytes},
                    }
                ],
                "outputs": [{"name": nom_sortie, "parameters": {"binary_data": False}}],
            }
        ).encode("utf-8")
        reponse = self.client.post(
            f"/v2/models/{modele}/infer",
            content=entete + X.tobytes(),
            headers={
                "Inference-Header-Content-Length": str(len(entete)),
                "Content-Type": "application/octet-stream",
            },
        )
        reponse.raise_for_status()
        sortie = reponse.json()["outputs"][0]
        return np.asarray(sortie["data"], dtype=np.float32).reshape(sortie["shape"])

    def probas_usure(self, X):
        """Probabilité d'usure de chaque vol (XGBoost v3 servi par FIL)."""
        return self.inferer("xgb_alerte", "input__0", "output__0", X)[:, 1]

    def reconstruction(self, Xn):
        """Reconstruction de vols centrés et standardisés (autoencodeur v5 servi par ONNX)."""
        return self.inferer("autoencodeur", "entree", "reconstruction", Xn)
