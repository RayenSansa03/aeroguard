# 1. Partir d'une image officielle : un mini-Linux avec Python 3.13 déjà installé
FROM python:3.13-slim

# 2. Se placer dans le dossier /app à l'intérieur de la boîte
WORKDIR /app

# 3. Copier la liste des librairies et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code du projet et l'installer
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
RUN pip install --no-cache-dir -e .

# 5. Créer le dossier où le simulateur enregistre la flotte
RUN mkdir -p data/processed

# 6. La commande lancée par défaut quand on démarre le conteneur
CMD ["python", "-m", "aeroguard.simulation"]
