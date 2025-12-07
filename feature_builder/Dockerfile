FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script CLI
COPY build_features.py .

# Hacer el script ejecutable
RUN chmod +x build_features.py

# Comando por defecto (se sobreescribe con docker compose run)
ENTRYPOINT ["python", "build_features.py"]