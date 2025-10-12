# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Evita que Python genere archivos .pyc y buffers de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependencias del sistema necesarias para compilaciones
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crea un usuario no root por seguridad
RUN useradd -m appuser

# Define el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . .

# Instala las dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Asigna el usuario no root
USER appuser

# Comando por defecto (ajústalo a tu script principal)
CMD ["python", "run.py"]
