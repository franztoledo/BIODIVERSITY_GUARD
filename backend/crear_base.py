"""Crea la base comun (backend/biodiversity_guard.db), carga municipios, historial y modelos, y genera la simulacion del modulo de IA.

Uso: python backend/crear_base.py [ruta_db]
Las funciones ingestar_semana y predecir_semana viven en base_datos.py y se reexportan aqui.
"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from backend.base_datos import (COLUMNAS_PANEL, DIR_SIMULADO, RUTA_CRUDO, RUTA_DB, cargar_municipios_y_historial, cargar_tabla_modelos, conectar,  # noqa: F401
                                crear_tablas, ingestar_semana, predecir_semana)
from ia_models.pipeline import cargar_metadata

SEMANA_FIN_HISTORIAL = "2025-12-29"
PRIMERA_SEMANA_2026, ULTIMA_SEMANA_2026 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-10-19")

README = """# Simulacion del modulo de IA

Cada archivo `semana_AAAA-MM-DD.csv` simula lo que el modulo de IA envia al backend una vez por semana: las columnas del panel crudo
municipio-semana (una fila por municipio de esa semana), sin targets y sin columnas preparadas.

**Son datos publicos de INPE (BDQueimadas, DETER), NASA POWER y SISAM** del panel `datos/datasets_eda/dataset_panel_semanal_COMPLETO.csv`,
usados para simular la interfaz. No son datos de una integracion en vivo. Las semanas de 2026 de SISAM son un pronostico sin calibrar.

Los archivos los genera `backend/crear_base.py` (semanas de 2026-01-05 a 2026-10-19) y los consume `backend/demo_flujo.py`.
"""


def generar_simulacion(crudo, carpeta=DIR_SIMULADO):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    n = 0
    for semana, g in crudo[(crudo["semana"] >= PRIMERA_SEMANA_2026) & (crudo["semana"] <= ULTIMA_SEMANA_2026)].groupby("semana"):
        g = g[COLUMNAS_PANEL].copy()
        g["semana"] = semana.strftime("%Y-%m-%d")
        g.to_csv(carpeta / f"semana_{semana.strftime('%Y-%m-%d')}.csv", index=False, encoding="utf-8")
        n += 1
    (carpeta / "README.md").write_text(README, encoding="utf-8")
    return n


def crear_base(ruta_db=None, carpeta_simulada=DIR_SIMULADO):
    """Crea la base desde cero (reemplaza el archivo si existe) y devuelve la conexion."""
    ruta = Path(ruta_db or RUTA_DB)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        ruta.unlink()
    conn = conectar(ruta)
    crear_tablas(conn)
    crudo, n_muni, n_hist = cargar_municipios_y_historial(conn, SEMANA_FIN_HISTORIAL)
    cargar_tabla_modelos(conn, cargar_metadata())
    n_sim = generar_simulacion(crudo, carpeta_simulada)
    # prediccion de la ultima semana del historial para que el KPI de control exista desde la primera semana de 2026
    predecir_semana(conn, SEMANA_FIN_HISTORIAL)
    print(f"Base creada: {ruta} | municipios {n_muni} | filas de historial {n_hist:,} | archivos simulados {n_sim}")
    return conn


if __name__ == "__main__":
    t0 = time.time()
    c = crear_base(sys.argv[1] if len(sys.argv) > 1 else None)
    c.close()
    print(f"Tiempo: {time.time() - t0:,.1f} s")
