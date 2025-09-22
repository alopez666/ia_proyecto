# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación y start.sh
COPY app.py .
COPY start.sh .

# Dar permisos de ejecución a start.sh
RUN chmod +x start.sh

# Comando por defecto para ejecutar start.sh
CMD ["./start.sh"]




# # Usar imagen oficial de Python 3.11
# FROM python:3.11-slim

# # Establecer directorio de trabajo
# WORKDIR /app

# # Copiar archivos necesarios
# COPY requirements.txt .
# COPY app.py .

# # Instalar dependencias
# RUN pip install --no-cache-dir -r requirements.txt

# # Exponer el puerto que usará FastAPI
# EXPOSE 8000

# # Comando para levantar la app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]