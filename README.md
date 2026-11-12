# BIODIVERSITY-GUARD-Predict

Sistema de Machine Learning para la gestión forestal y alerta temprana de amenazas en la Amazonía (curso 1ACC0057 Machine Learning, UPC, grupo 3).

Integra cuatro fuentes abiertas: **DETER**, **BDQueimadas** y **SISAM** (INPE) y **NASA POWER**. Entrega:

- **Frente 1 (regresión):** focos de calor por municipio a 7 y 30 días.
- **Frente 2 (clasificación):** clase de riesgo de fuego (Bajo, Moderado, Alto, Crítico).
- **Alertas** por municipio, **KPI** de resultado y de control del modelo, un **servicio** con base SQLite, una **API REST** y un **dashboard** Streamlit.

Metodología CRISP-DM. Los datos de 2026 son simulados con fuentes públicas (el SISAM de 2026 es un pronóstico sin calibrar).

## Qué hay en cada carpeta

| Carpeta | Contenido |
|---|---|
| `frontend/` | Dashboard Streamlit (`app.py`), selector de servicio (`conexion.py`) y servicio de ejemplo (`servicio_stub.py`). |
| `backend/` | Servicio real (`servicio.py`), API FastAPI (`api.py`), base SQLite (`base_datos.py`, `crear_base.py`), KPI (`kpis.py`) y pruebas (`probar_*.py`). |
| `ia_models/` | Modelos entrenados (`.joblib`), metadatos con versiones y umbrales (`modelos_metadata.json`) y el pipeline de inferencia (`pipeline.py`). |
| `codigo/` en el repositorio, `codigo/notebooks/` en el zip | Notebooks de preprocesamiento, EDA, Data Preparation y Modelado. |
| `datos/datasets_eda/` | Panel semanal crudo por municipio (lo usan `crear_base.py` y los KPI de contexto). |
| `datos/datasets_preparation/` | Dataset preparado y JSON de parámetros (`kpi_parametros.json`, `parametros_preparacion.json`). |
| `datos/modulo_ia_simulado/` | Un CSV por semana de 2026: simula lo que el módulo de IA envía cada semana. |
| `docs/` | Informe, manual de usuario, fichas de KPI (`fichas_kpi.json`), figuras, tablas y capturas. |
| `resultados/` | Métricas y predicciones de 2025 de los modelos (CSV). |
| `requirements.txt`, `.streamlit/config.toml` | Requisitos con versiones exactas (Python 3.11) y tema del dashboard. |
| `empaquetar.py` | Genera el zip de entrega. |
| `DESPLIEGUE.md` | Cómo publicar el dashboard en Streamlit Community Cloud. |

## Cómo correr todo

Desde la raíz del repositorio, con Python 3.11:

```bash
pip install -r requirements.txt
python backend/crear_base.py
streamlit run frontend/app.py
uvicorn backend.api:app
```

1. `python backend/crear_base.py` crea `backend/biodiversity_guard.db` con el historial hasta 2025-12-29 y las 42 semanas de 2026 (unos 30 s). Es opcional para el dashboard y la API: **si la base no existe, `backend/servicio.py` la crea sola** con el historial y las semanas de 2026 hasta **2026-08-31**, y deja las siguientes para el botón «Simular siguiente semana» y para subir archivos.
2. El dashboard usa el servicio de ejemplo por defecto. Para usar el servicio real:
   - PowerShell: `$env:BG_SERVICIO="real"; streamlit run frontend/app.py`
   - Bash: `BG_SERVICIO=real streamlit run frontend/app.py`
3. La API queda en <http://localhost:8000> y su documentación automática en <http://localhost:8000/docs>. Endpoints: `GET /salud`, `POST /ingesta`, `POST /predecir?semana=`, `GET /alertas`, `GET /predicciones`, `GET /kpis`, `GET /kpis/control`.
4. Variable de entorno `BG_DB`: ruta de la base (por defecto `backend/biodiversity_guard.db`). Útil en servidores con disco efímero, por ejemplo `BG_DB=/tmp/biodiversity_guard.db`.

Pruebas (usan una base temporal y no tocan la tuya):

```bash
python backend/probar_servicio.py
python backend/probar_api.py
python backend/probar_flujo.py
```

`probar_api.py` necesita `httpx` (`pip install httpx`).

## Correr desde el zip de entrega

El zip separa `codigo/`, `datos/` y `docs/` como pide el enunciado, pero el código busca `datos/`, `docs/` y `resultados/` junto a `frontend/`. Para correrlo desde el zip, copie esas carpetas dentro de `codigo/` y trabaje desde ahí:

```bash
cp -r datos docs resultados requirements.txt .streamlit codigo/
cd codigo
pip install -r requirements.txt
BG_SERVICIO=real streamlit run frontend/app.py
```

(En PowerShell use `Copy-Item -Recurse` y `$env:BG_SERVICIO="real"`.) Los notebooks están en `codigo/notebooks/`.

## Publicar el dashboard

Ver `DESPLIEGUE.md` (Streamlit Community Cloud conectado a GitHub).

## Empaquetar la entrega

```bash
python empaquetar.py                    # TB2_1ACC0057_16623_GRUPO_3.zip
python empaquetar.py OTRO_NOMBRE.zip
```
