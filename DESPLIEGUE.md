# Despliegue del dashboard en Streamlit Community Cloud

El dashboard (`frontend/app.py`) se publica conectando un repositorio de GitHub a Streamlit Community Cloud. Corre con el **servicio real** (`backend/servicio.py`).

## 1. Qué debe haber en el repositorio

Para no subir los datasets pesados, publique un repositorio (o una rama) que tenga solo estos archivos. El comando `python empaquetar.py --despliegue CARPETA` copia exactamente esta lista a `CARPETA`.

| Archivo | Para qué | Tamaño |
|---|---|---|
| `requirements.txt`, `.streamlit/config.toml` | Dependencias exactas y tema claro | menos de 0,01 MB |
| `frontend/app.py`, `conexion.py`, `servicio_stub.py` | Dashboard y selector de servicio | 0,05 MB |
| `backend/__init__.py`, `base_datos.py`, `crear_base.py`, `kpis.py`, `servicio.py`, `api.py` (las pruebas `probar_*.py` y `demo_flujo.py` son opcionales) | Servicio, base y KPI | 0,1 MB |
| `ia_models/__init__.py`, `pipeline.py`, `utilidades.py`, `modelos_metadata.json` | Pipeline de inferencia y metadatos | 0,05 MB |
| `ia_models/modelo_frente1_7d.joblib`, `modelo_frente1_30d.joblib`, `modelo_frente2.joblib` | Modelos entrenados | 0,81 + 0,65 + 27,74 MB |
| `datos/datasets_preparation/kpi_parametros.json`, `parametros_preparacion.json` | Parámetros de KPI y de preparación | 0,02 MB |
| `datos/datasets_eda/dataset_panel_semanal_COMPLETO.csv` | Panel crudo histórico: la base lo usa para crearse y `kpis.py` para los umbrales DETER | 40,8 MB |
| `datos/modulo_ia_simulado/semana_*.csv` (42) y `README.md` | Semanas de 2026 para «Simular siguiente semana» | 6,8 MB |
| `docs/fichas_kpi.json` | Fichas de sustento de los KPI | 0,02 MB |
| `backend/semilla/biodiversity_guard_semilla.db` | Base semilla hasta 2026-08-31 (arranque rápido) | 39,2 MB |

**Total: unos 116 MB. El archivo más grande es el panel crudo (40,8 MB); ninguno se acerca al límite de 100 MB de GitHub (GitHub avisa a partir de 50 MB, y no llegamos).**

Aviso: en la carpeta de trabajo `datos/datasets_preprocesamiento/SISAM/sisam_limpio.csv` pesa 112,6 MB y **GitHub lo rechaza** (límite de 100 MB por archivo). No lo suba (ni `datasets_preprocesamiento/` completo): use el repositorio mínimo de arriba, déjelo fuera con `.gitignore` o use Git LFS. Si ya lo confirmó en un commit local, hay que quitarlo del historial antes del primer `push`.

### Base de datos: semilla versionada

Se sube una base **semilla** ya creada: `backend/semilla/biodiversity_guard_semilla.db` (39 MB, historial hasta 2025-12-29 y semanas de 2026 hasta **2026-08-31**). No está en `.gitignore` (solo se ignora `backend/biodiversity_guard.db`) y **no va en el zip de entrega** (`empaquetar.py` excluye todo `.db`).

Al arrancar, `servicio.py` copia la semilla a `BG_DB` (o, si ahí no se puede escribir, a un directorio temporal) cuando la base no existe: el arranque es inmediato. Si la semilla falta o no llega a 2026-08-31, la crea como antes (unos 30 s, o más en la nube) y si no encuentra los `semana_*.csv` o no llegan a esa semana, falla con un error claro. Para regenerar la semilla (por ejemplo, si cambian los modelos):

```bash
python backend/servicio.py --crear-semilla
```

Las semanas 2026-09-07 a 2026-10-19 quedan para «Simular siguiente semana» y para subir el CSV. En Community Cloud el disco es efímero: al reiniciar la app se pierden las semanas ingestadas y vuelve a copiarse la semilla.

## 2. Publicar paso a paso

1. Suba el repositorio a GitHub (público, o privado si autoriza a Streamlit a leerlo).
2. Entre a <https://share.streamlit.io> con la cuenta de GitHub y pulse **Create app** (o **New app**) y elija **Deploy a public app from GitHub**.
3. Elija el **repositorio** y la **rama** (por ejemplo `main`).
4. En **Main file path** escriba `frontend/app.py`.
5. En **Advanced settings**:
   - **Python version**: `3.11`.
   - **Secrets**: pegue lo siguiente (Streamlit entrega los secretos también como variables de entorno):

     ```toml
     BG_SERVICIO = "real"
     BG_DB = "/tmp/biodiversity_guard.db"
     ```

6. Pulse **Deploy**. Streamlit instala `requirements.txt` (varios minutos, por xgboost, lightgbm y scikit-learn). El primer acceso copia la semilla (segundos). Verá `Servicio: real` en la barra lateral y, en los logs (**Manage app**), `[servicio] Base copiada de la semilla...`.

La URL queda del tipo `https://<nombre>.streamlit.app`. Se puede abrir una página directa con `?pagina=Ingesta`.

### Alternativa sin secretos

Cambiar `frontend/conexion.py` para que use el servicio real por defecto si encuentra el módulo (por ejemplo, `os.environ.get("BG_SERVICIO", "real")`). Ese archivo es del frontend y no lo modificó esta entrega.

## 3. Si el servidor se duerme

Las apps gratuitas se duermen tras unos días sin visitas (hacia 12 horas sin tráfico según la política vigente; confírmelo en la documentación de Streamlit):

1. Abra la URL y pulse **Yes, get this app back up!**. Arranca en uno o dos minutos y vuelve a crear la base (la ingesta hecha antes se pierde).
2. Para que no se duerma durante la demostración, abra la URL unos minutos antes de presentar.
3. Si queda en un error de instalación, en **Manage app** use **Reboot app**, o **Clear cache** y luego **Reboot**.
4. Si la app se queda sin memoria, reinícela; el límite de recursos de la capa gratuita es pequeño (carga el panel de 40 MB y los modelos).
5. Si falla `BG_SERVICIO=real`, la app cae al servicio de ejemplo y muestra un aviso arriba, y ese servicio necesita archivos que no están en el repositorio mínimo (`resultados/avanzado_predicciones_2025.csv`).

## 4. Comprobación previa en local

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux o Mac: source .venv/bin/activate
pip install -r requirements.txt
BG_SERVICIO=real streamlit run frontend/app.py
```

Verificado en un entorno virtual nuevo con Python 3.11: instalación correcta, `pip check` sin conflictos y recorrido de las 7 páginas sin errores.
