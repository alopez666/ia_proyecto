FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# ------------------- INICIO DE LA CORRECCIÓN -------------------

# Instala PyTorch usando su propio índice y el resto de paquetes desde PyPI en un solo paso.
# Se usa --extra-index-url para AÑADIR el repositorio de PyTorch sin eliminar el de PyPI.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# -------------------- FIN DE LA CORRECCIÓN ---------------------

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto
CMD ["./start.sh"]