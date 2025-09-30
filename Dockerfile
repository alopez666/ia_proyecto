# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# 1. Instalar la versión de PyTorch solo para CPU (mucho más pequeña)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto para ejecutar start.sh
CMD ["./start.sh"]