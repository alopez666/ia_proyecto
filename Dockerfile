# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# ------------------- CAMBIO CLAVE -------------------
# 1. Instalar la versión de PyTorch solo para CPU (mucho más pequeña)
#    Esto evita descargar las librerías de GPU (CUDA) que pesan gigabytes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt
# ----------------------------------------------------

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
[cite_start]RUN chmod +x start.sh [cite: 3]

# Comando por defecto para ejecutar start.sh
[cite_start]CMD ["./start.sh"] [cite: 3]
