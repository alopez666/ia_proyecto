# ---- Etapa 1: Builder ----
# Usa una imagen completa que incluya las herramientas de compilación
FROM python:3.11 as builder

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
 && rm -rf /var/lib/apt/lists/*

# Crea un directorio para las wheels y copia requirements.txt
WORKDIR /wheels
COPY requirements.txt .

# Descarga y compila todas las dependencias en formato "wheel"
# Esto compilará sentencepiece y guardará el resultado en /wheels
RUN pip wheel --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -w .

# ---- Etapa 2: Final ----
# Empieza de nuevo desde la imagen slim limpia
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copia solo las wheels (paquetes pre-compilados) desde la etapa builder
COPY --from=builder /wheels /wheels

# Instala las dependencias usando las wheels locales.
# --no-index le dice a pip que no busque en internet, solo en la carpeta local.
# --find-links le indica dónde están los paquetes.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /wheels/requirements.txt

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto
CMD ["./start.sh"]