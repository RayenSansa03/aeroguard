
# ---------- Stage 1 : build ----------
FROM python:3.13-slim AS builder

WORKDIR /build

# Environnement virtuel isolé, copié tel quel dans l'image finale
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installation des dépendances (couche mise en cache tant que le fichier ne change pas)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Installation du package applicatif
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# Suppression de pip : inutile à l'exécution, réduit la surface d'attaque
RUN pip uninstall -y pip


# ---------- Stage 2 : runtime ----------
FROM python:3.13-slim AS runtime

# Configuration de l'environnement Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Utilisateur non privilégié et suppression de pip de l'image de base
RUN useradd --create-home --uid 1000 aeroguard \
    && /usr/local/bin/python -m pip uninstall -y pip

# Récupération de l'environnement construit au stage précédent
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER aeroguard

EXPOSE 8000

# Vérification périodique de l'état du service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Démarrage du serveur
CMD ["uvicorn", "aeroguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
