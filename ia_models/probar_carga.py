"""Prueba de carga en un proceso nuevo, sin el notebook y sin definir nada en __main__.

Importa `ia_models.utilidades`, carga los tres modelos de `ia_models/` y comprueba que reproducen las
predicciones de 2025 guardadas en `resultados/avanzado_predicciones_2025.csv` (diferencia maxima menor a 1e-3).
Uso: python probar_carga.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import joblib
import numpy as np
import pandas as pd

import ia_models.utilidades  # noqa: F401  (hace importable la referencia que guardan los modelos)

TOLERANCIA = 1e-3
meta = json.load(open(RAIZ / "ia_models" / "modelos_metadata.json", encoding="utf-8"))
panel = pd.read_csv(RAIZ / "datos" / "datasets_preparation" / "dataset_panel_PREPARADO.csv", parse_dates=["semana"])
panel["semana"] = panel["semana"].dt.strftime("%Y-%m-%d")
guardadas = pd.read_csv(RAIZ / "resultados" / "avanzado_predicciones_2025.csv")
guardadas = guardadas[guardadas["modelo"] != "persistencia"]

print("Proceso nuevo: __main__ no define funciones del modelo; modulo importado:", ia_models.utilidades.__name__)
mal = 0
for tarea, info in meta["modelos"].items():
    modelo = joblib.load(RAIZ / "ia_models" / info["archivo"])
    g = guardadas[guardadas["frente"] == tarea]
    datos = g.merge(panel, on=["semana", "municipio_norm", "uf"], how="left", validate="one_to_one")
    assert len(datos) == len(g) and datos[info["variables"]].notna().all().all(), f"{tarea}: filas o variables faltantes"
    pred = np.asarray(modelo.predict(datos[info["variables"]]), dtype=float)
    dif = float(np.abs(pred - datos["predicho"].to_numpy(float)).max())
    ok = dif < TOLERANCIA
    mal += not ok
    print(f"{'OK' if ok else 'FALLA'} {tarea}: {type(modelo).__name__}, {len(datos):,} predicciones de 2025, diferencia maxima {dif:.2e} (tolerancia {TOLERANCIA:.0e})")
sys.exit(1 if mal else 0)
