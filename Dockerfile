# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar protobuf primero para evitar errores con transformers
RUN pip install --no-cache-dir protobuf>=4.23.4

# Instalar torch (CPU) y el resto de dependencias
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto
CMD ["./start.sh"]