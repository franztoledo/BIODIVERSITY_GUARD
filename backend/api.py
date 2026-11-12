"""API REST de BIODIVERSITY-GUARD-Predict (FastAPI): envoltura delgada sobre backend/servicio.py.

Uso: uvicorn backend.api:app --reload      (documentacion automatica en /docs)
Base de datos: variable de entorno BG_DB (por defecto backend/biodiversity_guard.db; se crea sola si no existe).
"""
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402

from backend import servicio as S  # noqa: E402

app = FastAPI(title="BIODIVERSITY-GUARD-Predict API", version="1.0.0",
              description="Ingesta semanal, predicción de focos y riesgo de fuego, alertas y KPI de control.")


def _limpio(v):
    """Convierte tipos de numpy/pandas a JSON valido (NaN pasa a null)."""
    if isinstance(v, dict):
        return {k: _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _df(d: pd.DataFrame) -> list:
    return _limpio(d.astype(object).where(d.notna(), None).to_dict(orient="records"))


def _semana(semana):
    """Valida que la semana tenga predicciones y devuelve el texto AAAA-MM-DD."""
    sems = S.semanas_disponibles()
    s = semana or sems[-1]
    if s not in sems:
        raise HTTPException(404, f"La semana {s} no tiene predicciones. Disponibles: {sems[0]} a {sems[-1]}.")
    return s


@app.get("/salud", summary="Estado del servicio")
def salud():
    try:
        return _limpio(S.salud())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"No hay conexión con la base de datos: {e}")


@app.post("/ingesta", summary="Sube el CSV de una semana, valida, guarda y predice")
async def ingesta(archivo: UploadFile = File(..., description="CSV con una sola semana y las columnas del panel crudo")):
    contenido = await archivo.read()
    nombre = archivo.filename or "semana.csv"
    v = S.validar_semana(contenido, nombre)
    if not v["ok"]:
        raise HTTPException(422, {"mensaje": "El archivo no pasó la validación.", "mensajes": v["mensajes"]})
    r = S.ingestar_y_predecir(contenido, nombre)
    if not r["ok"]:
        raise HTTPException(409, r["mensaje"])
    return _limpio({**r, "mensajes": v["mensajes"]})


@app.post("/predecir", summary="Recalcula la predicción de una semana ya ingestada")
def predecir(semana: str = Query(..., description="Lunes AAAA-MM-DD")):
    r = S.predecir(semana)
    if not r["ok"]:
        raise HTTPException(404, r["mensaje"])
    return _limpio(r)


@app.get("/alertas", summary="Tabla de alertas por prioridad")
def alertas(semana: str | None = Query(None, description="Lunes AAAA-MM-DD; por defecto la última")):
    s = _semana(semana)
    return {"semana": s, "alertas": _df(S.tabla_alertas(s))}


@app.get("/predicciones", summary="Predicciones por municipio (focos 7 y 30 días, clase de riesgo)")
def predicciones(semana: str | None = None, uf: str | None = Query(None, description="Sigla del estado, por ejemplo PA")):
    s = _semana(semana)
    d = S.predicciones(s, uf)
    if uf and d.empty:
        raise HTTPException(404, f"No hay predicciones para el estado {uf}.")
    return {"semana": s, "uf": uf, "predicciones": _df(d)}


@app.get("/kpis", summary="KPI de resultado, contexto y control de una semana")
def kpis(semana: str | None = None):
    s = _semana(semana)
    return {"semana": s, "kpis": _limpio(S.resumen_kpis(s))}


@app.get("/kpis/control", summary="KPI de control del modelo y estado")
def kpis_control():
    return {"estado": _limpio(S.estado_modelo()), "serie": _df(S.kpis_control())}
