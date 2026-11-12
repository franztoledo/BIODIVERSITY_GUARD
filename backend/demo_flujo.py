"""Demo del flujo semanal de 2026: ingesta y prediccion semana por semana. Uso: python backend/demo_flujo.py"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from backend.base_datos import DIR_SIMULADO, RUTA_DB, conectar, ingestar_semana, predecir_semana
from backend.crear_base import crear_base


def correr_flujo(conn, carpeta=DIR_SIMULADO, imprimir=True):
    archivos = sorted(Path(carpeta).glob("semana_*.csv"))
    assert archivos, "No hay archivos simulados; corre backend/crear_base.py"
    for ruta in archivos:
        t = time.time()
        ok, mensaje, filas = ingestar_semana(conn, ruta)
        if not ok:
            raise RuntimeError(f"{ruta.name}: ingesta rechazada: {mensaje}")
        semana = pd.Timestamp(ruta.stem.replace("semana_", ""))
        R = predecir_semana(conn, semana)
        kpi = conn.execute("SELECT estado FROM kpis_control WHERE semana_base = ?", ((semana - pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"),)).fetchone()
        if imprimir:
            print(f"{semana.date()} | filas ingestadas {filas} | alertas {int(R['es_alerta'].sum()):3d} | estado (semana base anterior) {kpi[0] if kpi else 'sin KPI'} | {time.time() - t:.1f} s")


if __name__ == "__main__":
    t0 = time.time()
    conn = crear_base()
    correr_flujo(conn)
    n = conn.execute("SELECT COUNT(*) FROM predicciones").fetchone()[0]
    print(f"\nFlujo de 2026 terminado: {n:,} predicciones guardadas | tiempo total {time.time() - t0:,.1f} s | base {RUTA_DB.stat().st_size / 1e6:,.1f} MB")
    conn.close()
