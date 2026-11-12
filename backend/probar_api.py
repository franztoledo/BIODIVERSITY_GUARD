"""Prueba de backend/api.py con fastapi.testclient.TestClient y una base temporal (no toca backend/biodiversity_guard.db).

Uso: python backend/probar_api.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

TMP = Path(tempfile.mkdtemp(prefix="bg_api_"))
os.environ["BG_DB"] = str(TMP / "prueba.db")

from fastapi.testclient import TestClient  # noqa: E402

from backend import servicio as S  # noqa: E402
from backend.api import app  # noqa: E402
from backend.base_datos import DIR_SIMULADO  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'OK   ' if cond else 'FALLA'} {nombre}{(' | ' + str(detalle)) if detalle else ''}")
    if not cond:
        fallos.append(nombre)


try:
    cli = TestClient(app)
    print("Los 7 endpoints")
    r = cli.get("/salud")
    j = r.json()
    check("GET /salud", r.status_code == 200 and j["base"] == "ok" and j["modelo"]["version"] and j["ultima_semana"] == S.PRECARGA_HASTA, j)
    u = j["ultima_semana"]
    r = cli.get("/alertas", params={"semana": u})
    j = r.json()
    check("GET /alertas", r.status_code == 200 and j["semana"] == u and len(j["alertas"]) == 543 and j["alertas"][0]["prioridad"] == 1,
          f"{sum(a['es_alerta'] for a in j['alertas'])} alertas")
    r = cli.get("/predicciones", params={"semana": u, "uf": "PA"})
    j = r.json()
    check("GET /predicciones?uf=PA", r.status_code == 200 and j["predicciones"] and {p["uf"] for p in j["predicciones"]} == {"PA"}, f"{len(j['predicciones'])} municipios")
    r = cli.get("/kpis", params={"semana": u})
    j = r.json()
    check("GET /kpis", r.status_code == 200 and len(j["kpis"]) == 10, [k["id"] for k in j["kpis"]][:3])
    r = cli.get("/kpis/control")
    j = r.json()
    check("GET /kpis/control", r.status_code == 200 and j["estado"]["estado"] and len(j["serie"]) > 0, j["estado"]["estado"])
    r = cli.post("/predecir", params={"semana": u})
    check("POST /predecir", r.status_code == 200 and r.json()["ok"], r.json())
    check("GET /docs (documentacion automatica)", cli.get("/docs").status_code == 200)

    print("\nErrores con codigo HTTP")
    check("semana sin datos: 404 en /alertas", cli.get("/alertas", params={"semana": "2019-01-07"}).status_code == 404)
    check("semana no ingestada: 404 en /predecir", cli.post("/predecir", params={"semana": "2027-01-04"}).status_code == 404)
    check("estado inexistente: 404 en /predicciones", cli.get("/predicciones", params={"uf": "XX"}).status_code == 404)

    print("\n/ingesta")
    sig = S._mas7(u)
    b = (Path(DIR_SIMULADO) / f"semana_{sig}.csv").read_bytes()
    malo = pd.read_csv(io.BytesIO(b)).assign(RH2M=150.0).to_csv(index=False).encode()
    r = cli.post("/ingesta", files={"archivo": ("malo.csv", malo, "text/csv")})
    check("archivo malo (humedad 150): 4xx con mensaje", 400 <= r.status_code < 500 and "ERROR" in str(r.json()), f"{r.status_code} {str(r.json())[:120]}")
    r = cli.post("/ingesta", files={"archivo": ("vacio.csv", b"", "text/csv")})
    check("archivo vacio: 4xx", 400 <= r.status_code < 500, r.status_code)
    check("tras los rechazos la ultima semana no cambio", cli.get("/salud").json()["ultima_semana"] == u)
    r = cli.post("/ingesta", files={"archivo": (f"semana_{sig}.csv", b, "text/csv")})
    j = r.json()
    check("archivo bueno: 200 y semana guardada", r.status_code == 200 and j["ok"] and j["semana"] == sig, j.get("mensaje"))
    check("/salud refleja la semana nueva", cli.get("/salud").json()["ultima_semana"] == sig)
    r = cli.post("/ingesta", files={"archivo": (f"semana_{sig}.csv", b, "text/csv")})
    check("mismo archivo otra vez: 4xx (semana repetida)", 400 <= r.status_code < 500, r.status_code)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if fallos:
    print(f"FALLA: {len(fallos)} comprobaciones: {fallos}")
    sys.exit(1)
print("OK: todas las comprobaciones de la API pasaron")
