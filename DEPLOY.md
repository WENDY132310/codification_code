# ⚡ DataLab · Guía de Despliegue en Streamlit Cloud

## Estructura del proyecto

```
datalab/
├── app.py              ← Aplicación principal (todo en un archivo)
├── requirements.txt    ← Dependencias Python
└── README.md           ← (Opcional) descripción del proyecto
```

---

## Opción A — Streamlit Community Cloud (gratuito, recomendado)

### Paso 1 · Subir a GitHub

```bash
# En la carpeta del proyecto
git init
git add app.py requirements.txt
git commit -m "feat: DataLab compresión de datos v1.0"

# Crear repo en github.com y conectar
git remote add origin https://github.com/TU_USUARIO/datalab-compresion.git
git push -u origin main
```

### Paso 2 · Desplegar en Streamlit Cloud

1. Ve a **https://share.streamlit.io**
2. Inicia sesión con tu cuenta de GitHub
3. Clic en **"New app"**
4. Selecciona:
   - **Repository**: `TU_USUARIO/datalab-compresion`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Clic en **"Deploy!"**
6. En ~2 minutos tendrás tu URL pública:
   `https://tu-usuario-datalab-compresion-app-XXXX.streamlit.app`

---

## Opción B — Local (desarrollo)

```bash
# 1. Crear entorno virtual
python -m venv venv

# En Mac/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la app
streamlit run app.py

# La app abre automáticamente en http://localhost:8501
```

---

## Opción C — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

```bash
docker build -t datalab .
docker run -p 8501:8501 datalab
```

---

## Configuración opcional (.streamlit/config.toml)

Crea la carpeta `.streamlit/` y el archivo `config.toml`:

```toml
[theme]
base = "dark"
backgroundColor = "#030712"
secondaryBackgroundColor = "#0d1424"
textColor = "#cdd9e5"
primaryColor = "#06b6d4"

[server]
maxUploadSize = 200
enableCORS = false
headless = true

[browser]
gatherUsageStats = false
```

---

## Notas importantes

- **Tamaño máximo de archivo**: 200 MB por defecto en Streamlit Cloud (ajustable)
- **PIL/Pillow**: necesario para análisis de imágenes (ya en requirements.txt)
- **Audio**: el análisis completo de PCM requiere archivos `.wav`; para `.mp3` se analiza el flujo comprimido
- **Video**: el análisis es a nivel byte del contenedor; para análisis frame-a-frame instala `opencv-python` y `moviepy`

## Dependencias opcionales para extensión futura

```bash
# Análisis avanzado de audio
pip install librosa soundfile

# Análisis de video frame-a-frame
pip install opencv-python moviepy

# Scipy para DCT optimizada
pip install scipy
```
