"""Prueba de backend/kpis.py con los CSV existentes (sin base SQLite). Uso: python backend/probar_kpis.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpis as K  # noqa: E402

R = K.RAIZ
fallas = []


def chequeo(nombre, cond, detalle=""):
    print(("OK    " if cond else "FALLA ") + nombre + (f"  {detalle}" if detalle else ""))
    if not cond:
        fallas.append(nombre)


panel = pd.read_csv(R / "datos/datasets_eda/dataset_panel_semanal_COMPLETO.csv", parse_dates=["semana"])
pred = pd.read_csv(R / "resultados/avanzado_predicciones_2025.csv", parse_dates=["semana"])
p7 = pred[(pred.frente == "f1_7d") & (pred.modelo == "lgbm_final_sin_sisam")].copy()
ult = p7.semana.max()
semanas = {"humeda": pd.Timestamp("2025-03-10"), "seca": pd.Timestamp("2025-09-08"), "ultima": ult}

# 1. siete KPI en tres semanas
print("== 1. KPI de resultado ==")
for etq, s in semanas.items():
    obs = panel[panel.semana == s]
    pr = p7[p7.semana == s].rename(columns={"predicho": "focos_7d"})
    try:
        res = K.kpis_resultado(obs, pr, panel[panel.semana <= s], s)
        chequeo(f"7 KPI semana {etq} {s.date()}", len(res) == 7 and all(k in r for r in res for k in
                ["id", "nombre", "valor", "unidad", "estado", "umbral_ambar", "umbral_rojo", "base_umbral", "lectura"]))
        for r in res:
            print(f"      {r['id']:<22} valor={r['valor']} {r['unidad']:<24} estado={r['estado']}")
    except Exception as e:
        chequeo(f"7 KPI semana {etq}", False, repr(e))

# 2. control sobre 2025
print("== 2. KPI de control 2025 ==")
ctl = K.kpis_control(p7)
g = K.kpis_control_global(p7)
u = K.umbrales_control()
chequeo("48 semanas evaluadas", len(ctl) == 48, f"({len(ctl)})")
chequeo("recall global 0.494", abs(g["recall_alerta"] - 0.494) < 0.0006, f"({g['recall_alerta']:.4f})")
chequeo("falsas alarmas globales 0.0144", abs(g["falsas_alarmas"] - 0.0144) < 0.00006, f"({g['falsas_alarmas']:.5f})")
chequeo("ninguna semana sobre MAE movil maximo", (ctl.mae_movil_4s <= u["mae_max"]).all(), f"(max {ctl.mae_movil_4s.max():.3f} vs {u['mae_max']:.3f})")
print("      estados:", ctl.estado.value_counts().to_dict())
print("      recall movil min/max:", round(ctl.recall_alerta.min(), 3), round(ctl.recall_alerta.max(), 3),
      "| falsas alarmas movil max:", round(ctl.falsas_alarmas.max(), 4), "| MAE global:", round(g["mae"], 3))
print("      estado_modelo:", K.estado_modelo(ctl)["estado"], "-", len(K.estado_modelo(ctl)["semanas_fuera_umbral"]), "semanas fuera")

em = K.estado_modelo(ctl)
chequeo("estado_modelo no dice 'dentro de los umbrales' sin mencionar semanas fuera", em["estado"] != "normal" or not em["semanas_fuera_umbral"] or "salieron" in em["mensaje"], f"({em['mensaje']})")
vm = json.loads(K.RUTA_META.read_text(encoding="utf-8"))["kpi_control"]
chequeo("metadato con ventanas_moviles y claves previas intactas", "ventanas_moviles" in vm and abs(vm["mae_movil_4_semanas_maximo"]["final"] - 5.786892274941955) < 1e-9)
for etq, x in (("2025", p7),):
    c = K.kpis_control(x)
    print(f"      semanas {etq} fuera de umbral: {(c.estado != 'normal').mean() * 100:.1f} % (reentrenar {(c.estado == 'reentrenar recomendado').mean() * 100:.1f} %)")

# tabla de alertas
s = semanas["seca"]
ta = K.tabla_alertas(p7[p7.semana == s].rename(columns={"predicho": "focos_7d"}), panel[panel.semana == s])
ok = list(ta.columns) == ["municipio_norm", "uf", "nivel_focos", "focos_7d", "focos_30d", "clase_riesgo", "prob_critico", "es_alerta", "prioridad"]
ok &= ta.prioridad.tolist() == list(range(1, len(ta) + 1))
ok &= ta.es_alerta.is_monotonic_decreasing
n_al = int(ta.es_alerta.sum())
ok &= ta.loc[ta.es_alerta == 1, "focos_7d"].is_monotonic_decreasing
chequeo("tabla_alertas columnas y prioridad", bool(ok), f"({n_al} alertas de {len(ta)})")

# 3. casos limite
print("== 3. Casos limite ==")
try:
    s = semanas["seca"]
    pr = p7[p7.semana == s].rename(columns={"predicho": "focos_7d"})
    obs0 = panel[panel.semana == s].copy()
    obs0["num_incendios"] = 0
    r = {k["id"]: k for k in K.kpis_resultado(obs0, pr, panel[panel.semana <= s], s)}
    chequeo("semana sin focos", r["nivel_focos"]["valor"] == 0.0 and r["nivel_focos"]["estado"] == "verde", f"({r['nivel_focos']['estado']})")
    obs1 = panel[panel.semana == s].copy()
    obs1["riesgo_fuego_promedio"] = np.nan
    r = {k["id"]: k for k in K.kpis_resultado(obs1, pr, panel[panel.semana <= s], s)}
    chequeo("riesgo nulo -> sin dato", r["riesgo_fuego_inpe"]["estado"] == "sin_dato" and r["riesgo_fuego_inpe"]["valor"] is None)
    hist = panel[(panel.semana <= s) & (panel.semana.dt.year >= 2025)]
    r = {k["id"]: k for k in K.kpis_resultado(panel[panel.semana == s], pr, hist, s)}
    chequeo("sin anio anterior -> sin dato", r["variacion_interanual"]["estado"] == "sin_dato")
    r = K.kpis_resultado(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), s)
    chequeo("todo vacio no falla", len(r) == 7 and sum(k["estado"] == "sin_dato" for k in r) == 6, f"({[k['estado'] for k in r]})")
    chequeo("control vacio", K.estado_modelo(K.kpis_control(pd.DataFrame()))["estado"] == "sin_dato")
    chequeo("tabla_alertas vacia", K.tabla_alertas(pd.DataFrame(columns=["semana_base"]), None).empty)
except Exception as e:
    chequeo("casos limite sin error", False, repr(e))

# 4. fichas
print("== 4. Fichas ==")
try:
    K.generar_fichas()
    d = json.loads(K.RUTA_FICHAS.read_text(encoding="utf-8"))
    campos = ["id", "nombre", "que_mide", "formula", "umbral", "base_del_umbral", "fuente_dato", "limitaciones", "accion_sugerida"]
    chequeo("10 fichas con todos los campos", len(d) == 10 and all(all(c in f and f[c] for c in campos) for f in d))
    texto = K.RUTA_FICHAS.read_text(encoding="utf-8")
    chequeo("fichas con tildes (UTF-8)", all(w in texto for w in ["Variación", "Más", "Anticipación", "deforestación"]) and "movil de" not in texto)
    chequeo("umbral con verde, ambar y rojo", all(set(f["umbral"]) == {"verde", "ambar", "rojo"} for f in d))
    ids = {f["id"] for f in d}
    chequeo("ids de resultado en fichas", {"nivel_focos", "alertas_vs_capacidad", "variacion_interanual", "anticipacion_alerta",
                                           "riesgo_fuego_inpe", "calidad_aire_pm25", "deforestacion_deter"} <= ids)
except Exception as e:
    chequeo("fichas", False, repr(e))

print("\nRESULTADO:", "TODO OK" if not fallas else f"FALLAS: {fallas}")
sys.exit(1 if fallas else 0)
