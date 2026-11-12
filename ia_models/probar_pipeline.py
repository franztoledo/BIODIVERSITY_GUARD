"""Valida el pipeline: variables y predicciones contra dataset_panel_PREPARADO.csv (referencia de verdad). Imprime OK o FALLA.
Uso: python ia_models/probar_pipeline.py
"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from ia_models.pipeline import CLAVE, CLASES, cargar_modelos, cargar_parametros, columnas_de_modelos, predecir, transformar_semana

TOL = 1e-6
P = cargar_parametros()
modelos = cargar_modelos()
meta = modelos["meta"]
columnas = columnas_de_modelos(meta)

crudo = pd.read_csv(RAIZ / "datos" / "datasets_eda" / "dataset_panel_semanal_COMPLETO.csv", parse_dates=["semana"])
prep = pd.read_csv(RAIZ / "datos" / "datasets_preparation" / "dataset_panel_PREPARADO.csv", parse_dates=["semana"])

sem_2026 = sorted(prep.loc[prep["particion"] == "flujo_2026", "semana"].unique())
sem_2025 = sorted(prep.loc[prep["particion"] == "prueba", "semana"].unique())
sem_2025 = [sem_2025[i] for i in np.linspace(0, len(sem_2025) - 1, 26).round().astype(int)]   # 26 semanas repartidas en 2025
semanas = sem_2025 + sem_2026
print(f"Semanas a validar: {len(sem_2026)} de 2026 y {len(sem_2025)} de 2025 (muestra repartida)")

t0 = time.time()
feats, preds = [], []
for s in semanas:
    # solo el historial crudo anterior o igual a la semana (el pipeline ignora lo posterior; aqui ni siquiera se le pasa)
    hist = crudo[crudo["semana"] <= s]
    X = transformar_semana(hist, s, parametros=P, meta=meta)
    feats.append(X)
    preds.append(predecir(X, modelos))
X = pd.concat(feats, ignore_index=True)
R = pd.concat(preds, ignore_index=True)
print(f"Pipeline: {len(X):,} filas en {time.time() - t0:,.1f} s")

# variante sin redondeo: muestra que la diferencia de la parte 1 sin redondeo es solo la precision de 6 cifras del CSV de referencia
X_exacto = pd.concat([transformar_semana(crudo[crudo["semana"] <= s_], s_, parametros=P, meta=meta, redondear=False) for s_ in semanas[::7]], ignore_index=True)

# 1. variables contra PREPARADO
ref = prep.merge(X[["semana"] + CLAVE].assign(_x=1), on=["semana"] + CLAVE, how="inner")
ref = ref.sort_values(["semana"] + CLAVE).reset_index(drop=True)
Xo = X.sort_values(["semana"] + CLAVE).reset_index(drop=True)
assert len(ref) == len(Xo) and (ref[["semana"] + CLAVE].to_numpy() == Xo[["semana"] + CLAVE].to_numpy()).all()
dif = pd.Series({c: float(np.abs(Xo[c].to_numpy(float) - ref[c].to_numpy(float)).max()) for c in columnas}).sort_values(ascending=False)
print(f"\n1. Variables ({len(columnas)}) contra PREPARADO: diferencia maxima {dif.max():.3e}")
print("   Las 5 mayores:", {k: f"{v:.2e}" for k, v in dif.head(5).items()})
Xe = X_exacto.merge(prep[["semana"] + CLAVE + columnas], on=["semana"] + CLAVE, suffixes=("", "_ref"))
dif_exacta = max(float(np.abs(Xe[c].to_numpy(float) - Xe[c + "_ref"].to_numpy(float)).max()) for c in columnas)
print(f"   Sin redondeo (precision completa, {len(Xe):,} filas de {len(semanas[::7])} semanas): {dif_exacta:.2e}; es el redondeo a 6 cifras significativas del CSV de referencia (error relativo menor a 1e-6)")
ok1 = dif.max() < TOL

# 2. predicciones: pipeline contra modelos aplicados directo a las filas de PREPARADO
M = modelos["modelos"]
v = {t: meta["modelos"][t]["variables"] for t in M}
d7 = np.asarray(M["f1_7d"].predict(ref[v["f1_7d"]]), float)
d30 = np.asarray(M["f1_30d"].predict(ref[v["f1_30d"]]), float)
dc = M["f2"].predict(ref[v["f2"]])
dp = M["f2"].predict_proba(ref[v["f2"]])
Ro = R.sort_values(["semana_base"] + CLAVE).reset_index(drop=True)
k, a = meta["alerta"]["factor_k"], meta["alerta"]
mes = (pd.to_datetime(ref["semana"]) + pd.Timedelta(weeks=1)).dt.month
alerta_ref = (k * d7 > np.where(mes.between(7, 11), a["umbral_seca"], a["umbral_humeda"])).astype(int)
dif7, dif30 = np.abs(Ro["focos_7d"] - d7).max(), np.abs(Ro["focos_30d"] - d30).max()
difp = np.abs(Ro[["prob_bajo", "prob_moderado", "prob_alto", "prob_critico"]].to_numpy() - dp).max()
clases_ok = (Ro["clase_riesgo"].to_numpy() == np.array([CLASES[int(c)] for c in dc])).all()
alertas_ok = (Ro["es_alerta"].to_numpy() == alerta_ref).all()
print(f"\n2. Predicciones contra los modelos sobre PREPARADO: focos 7d {dif7:.3e} | focos 30d {dif30:.3e} | probabilidades {difp:.3e} | clases iguales: {clases_ok} | alertas iguales: {alertas_ok}")
ok2 = dif7 < TOL and dif30 < TOL and clases_ok and alertas_ok
print(f"   Alertas en la validacion: {int(alerta_ref.sum()):,} de {len(alerta_ref):,} filas")

ok = ok1 and ok2
print("\nOK" if ok else "\nFALLA")
sys.exit(0 if ok else 1)
