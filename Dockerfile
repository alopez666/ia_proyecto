# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar todas las dependencias en una sola capa (layer)
RUN pip install --no-cache-dir \
    protobuf>=4.23.4 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copiar la aplicación y start.sh
COPY app.py start.sh ./

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto
CMD ["./start.sh"]