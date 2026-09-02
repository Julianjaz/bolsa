FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema requeridas (opcional para compilar pandas/numpy si no hay wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer el puerto
EXPOSE 8000

# Variable de entorno de ejemplo, el usuario debe proveer GEMINI_API_KEY en runtime
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
