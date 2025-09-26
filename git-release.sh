#!/bin/bash
# git-release.sh
# Script para automatizar release de la app FastAPI y Docker

# -----------------------------
# Validar versión
# -----------------------------
VERSION=$1
if [ -z "$VERSION" ]; then
  echo "Por favor especifica la versión: ./git-release.sh 1.0.0"
  exit 1
fi

# -----------------------------
# Git: agregar, commit, push
# -----------------------------
echo "Agregando cambios..."
git add .

echo "Haciendo commit..."
git commit -m "Release versión $VERSION"

echo "Haciendo push a la rama principal..."
git push origin main

echo "Creando tag de versión $VERSION..."
git tag -a "v$VERSION" -m "Versión $VERSION"

echo "Haciendo push de tags..."
git push origin "v$VERSION"

# -----------------------------
# Docker: construir y etiquetar imagen
# -----------------------------
IMAGE_NAME="alanlj/mi_proyecto_ia_api_fastapi"

echo "Construyendo imagen Docker..."
docker build -t "$IMAGE_NAME:$VERSION" .

# Etiquetar como latest también (opcional)
docker tag "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"

# -----------------------------
# Push de imagen (opcional)
# -----------------------------
# echo "Haciendo push de la imagen a Docker Hub..."
# docker push "$IMAGE_NAME:$VERSION"
# docker push "$IMAGE_NAME:latest"

# -----------------------------
# Mensaje final
# -----------------------------
echo "Release v$VERSION completado."
echo "Imagen Docker lista: $IMAGE_NAME:$VERSION"