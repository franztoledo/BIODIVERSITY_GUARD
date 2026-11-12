"""Prueba obligatoria del flujo: (1) las predicciones guardadas en la base coinciden con las de aplicar los modelos a PREPARADO,
(2) prueba de estres de la ingesta. Usa una base temporal (no toca backend/biodiversity_guard.db). Imprime OK o FALLA.
Uso: python backend/probar_flujo.py
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from backend.base_datos import DIR_SIMULADO, conectar, ingestar_semana
from backend.crear_base import crear_base
from backend.demo_flujo import correr_flujo
from ia_models.pipeline import CLAVE, CLASES, cargar_modelos

TOL = 1e-6
tmp = Path(tempfile.mkdtemp(prefix="bg_prueba_"))
try:
    t0 = time.time()
    conn = crear_base(tmp / "prueba.db", carpeta_simulada=DIR_SIMULADO)
    correr_flujo(conn, imprimir=False)
    t_flujo = time.time() - t0
    print(f"Flujo completo de 2026 por la base: {t_flujo:,.1f} s (incluye crear la base)")

    # 1. predicciones de la base contra los modelos aplicados a las filas de PREPARADO
    mod = cargar_modelos()
    meta, M = mod["meta"], mod["modelos"]
    prep = pd.read_csv(RAIZ / "datos" / "datasets_preparation" / "dataset_panel_PREPARADO.csv", parse_dates=["semana"])
    prep = prep[prep["particion"] == "flujo_2026"].sort_values(["semana"] + CLAVE).reset_index(drop=True)
    bd = pd.read_sql_query("SELECT * FROM predicciones WHERE semana_base >= '2026-01-05'", conn, parse_dates=["semana_base"]).sort_values(["semana_base"] + CLAVE).reset_index(drop=True)
    assert len(bd) == len(prep), (len(bd), len(prep))
    assert (bd[["semana_base"] + CLAVE].to_numpy() == prep[["semana"] + CLAVE].to_numpy()).all()
    v = {t: meta["modelos"][t]["variables"] for t in M}
    d7, d30 = M["f1_7d"].predict(prep[v["f1_7d"]]), M["f1_30d"].predict(prep[v["f1_30d"]])
    clases = np.array([CLASES[int(c)] for c in M["f2"].predict(prep[v["f2"]])])
    proba = M["f2"].predict_proba(prep[v["f2"]])
    a = meta["alerta"]
    mes = (prep["semana"] + pd.Timedelta(weeks=1)).dt.month
    alerta = (a["factor_k"] * d7 > np.where(mes.between(7, 11), a["umbral_seca"], a["umbral_humeda"])).astype(int)
    dif7, dif30 = float(np.abs(bd["focos_7d"] - d7).max()), float(np.abs(bd["focos_30d"] - d30).max())
    difp = float(np.abs(bd[["prob_bajo", "prob_moderado", "prob_alto", "prob_critico"]].to_numpy() - proba).max())
    ok_clase, ok_alerta = bool((bd["clase_riesgo"].to_numpy() == clases).all()), bool((bd["es_alerta"].to_numpy() == alerta).all())
    ok1 = dif7 < TOL and dif30 < TOL and ok_clase and ok_alerta
    print(f"\n1. Base contra modelos sobre PREPARADO ({len(bd):,} predicciones, 42 semanas): focos 7d {dif7:.2e} | focos 30d {dif30:.2e} | probabilidades {difp:.2e} | "
          f"clases iguales {ok_clase} | alertas iguales {ok_alerta} ({int(alerta.sum()):,} alertas)")
    print(f"   {'OK' if ok1 else 'FALLA'}")

    # 2. prueba de estres: la base debe quedar sin cambios (las observaciones y las predicciones); solo se agrega el registro en `ingestas`
    def foto():
        return (conn.execute("SELECT COUNT(*), SUM(num_incendios), SUM(RH2M) FROM observaciones_semanales").fetchone(),
                conn.execute("SELECT COUNT(*), SUM(focos_7d) FROM predicciones").fetchone(), conn.execute("SELECT MAX(semana) FROM observaciones_semanales").fetchone())

    # estado de partida para los casos: base con el historial y la semana 2026-01-05 ya ingestada
    conn.close()
    conn = crear_base(tmp / "estres.db", carpeta_simulada=DIR_SIMULADO)
    ok, msg, _ = ingestar_semana(conn, DIR_SIMULADO / "semana_2026-01-05.csv")
    assert ok, msg
    carpeta = tmp / "casos"
    carpeta.mkdir()
    base_2 = pd.read_csv(DIR_SIMULADO / "semana_2026-01-12.csv")
    base_1 = pd.read_csv(DIR_SIMULADO / "semana_2026-01-05.csv")
    casos = {}
    casos["a) semana repetida"] = DIR_SIMULADO / "semana_2026-01-05.csv"
    casos["b) semana que salta una semana"] = DIR_SIMULADO / "semana_2026-01-19.csv"
    sin_col = base_2.drop(columns=["T2M"]); sin_col.to_csv(carpeta / "c.csv", index=False); casos["c) columna faltante (T2M)"] = carpeta / "c.csv"
    humedad = base_2.copy(); humedad.loc[0, "RH2M"] = 150; humedad.to_csv(carpeta / "d.csv", index=False); casos["d) humedad 150"] = carpeta / "d.csv"
    todos_rechazados = True
    for nombre, ruta in casos.items():
        antes = foto()
        n_ing = conn.execute("SELECT COUNT(*) FROM ingestas").fetchone()[0]
        ok, mensaje, _ = ingestar_semana(conn, ruta)
        despues = foto()
        registro = conn.execute("SELECT estado, mensaje FROM ingestas ORDER BY id DESC LIMIT 1").fetchone()
        n_ing2 = conn.execute("SELECT COUNT(*) FROM ingestas").fetchone()[0]
        bien = (not ok) and antes == despues and registro[0] == "rechazada" and n_ing2 == n_ing + 1
        todos_rechazados &= bien
        print(f"   {nombre}: {'RECHAZADO' if not ok else 'ACEPTADO (error)'} | base sin cambios: {antes == despues} | registrado en ingestas: {registro[0]} | motivo: {mensaje}")
    # control: la semana correcta si se acepta
    ok_control, msg_control, filas_control = ingestar_semana(conn, DIR_SIMULADO / "semana_2026-01-12.csv")
    print(f"   control) semana correcta 2026-01-12: {'ACEPTADA' if ok_control else 'RECHAZADA'} ({filas_control} filas)")
    ok2 = todos_rechazados and ok_control
    print(f"2. Prueba de estres: {'OK' if ok2 else 'FALLA'}")
    conn.close()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

ok = ok1 and ok2
print("\nOK" if ok else "\nFALLA")
sys.exit(0 if ok else 1)
