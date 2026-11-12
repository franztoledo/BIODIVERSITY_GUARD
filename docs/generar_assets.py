"""Genera las figuras (docs/figuras) y tablas (docs/tablas) del informe.

Uso (desde la raiz del repositorio):  python docs/generar_assets.py
Todas las cifras salen de resultados/, ia_models/modelos_metadata.json y
datos/datasets_preparation/*.json. Solo se escribe en docs/figuras y docs/tablas.

Supuestos anotados:
- Los valores de criterio de exito (2.946, 11.044, 0.615, 0.645 en validacion;
  1.165 y 4.523 en 2025) se fijaron en el notebook de modelado avanzado y no estan en
  resultados/, asi que se declaran abajo como constantes.
- kpi_umbrales y semaforo_kpi se leen de docs/fichas_kpi.json (umbrales exactos, sin tramos propios).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
FIG = RAIZ / "docs" / "figuras"
TAB = RAIZ / "docs" / "tablas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

AZUL, GRIS, NARANJA, ROJO = "#1f5fa8", "#9aa0a6", "#e8833a", "#c0392b"
AZUL_CLARO, GRIS_OSC, VERDE, AMBAR = "#8fb4dc", "#5f6368", "#3a9d5d", "#e0a800"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "savefig.dpi": 300, "figure.dpi": 100,
})

# ---------------------------------------------------------------- datos
A = pd.read_csv(RAIZ / "resultados" / "avanzado_metricas.csv")
B = pd.read_csv(RAIZ / "resultados" / "baseline_metricas.csv")
TUN = pd.read_csv(RAIZ / "resultados" / "avanzado_tuning.csv")
PA = pd.read_csv(RAIZ / "resultados" / "avanzado_predicciones_2025.csv", parse_dates=["semana"])
PB = pd.read_csv(RAIZ / "resultados" / "baseline_predicciones_2025.csv", parse_dates=["semana"])
META = json.load(open(RAIZ / "ia_models" / "modelos_metadata.json", encoding="utf8"))
PREP = json.load(open(RAIZ / "datos" / "datasets_preparation" / "parametros_preparacion.json", encoding="utf8"))
KPI = json.load(open(RAIZ / "datos" / "datasets_preparation" / "kpi_parametros.json", encoding="utf8"))
A["fuente"] = "avanzado"
B["fuente"] = "baseline"
B["variante"] = "baseline"
B["tratamiento"] = B["modelo"].map(lambda m: "pesos" if m == "arbol_balanceado" else "no aplica")
ALL = pd.concat([A, B], ignore_index=True)

# criterios de exito (notebook de modelado avanzado)
CRIT = {"f1_7d": (2.946, 1.165), "f1_30d": (11.044, 4.523), "f2_f1": 0.615, "f2_rc": 0.645}
SUB = {"f1_7d": "todas", "f1_30d": "todas", "f2": "con_clase_t"}


def v(frente, modelo, variante, trat, sisam, ev, metrica, df=None):
    d = ALL if df is None else df
    s = d[(d.frente == frente) & (d.modelo == modelo) & (d.variante == variante)
          & (d.tratamiento == trat) & (d.sisam == sisam) & (d.evaluacion == ev)
          & (d.metrica == metrica) & (d.subconjunto == SUB[frente])]["valor"]
    return float(s.iloc[0]) if len(s) else float("nan")


def combos(frente):
    d = ALL[ALL.frente == frente][["fuente", "modelo", "variante", "tratamiento", "sisam"]].drop_duplicates(["modelo", "variante", "tratamiento", "sisam"])
    orden = {"persistencia": 0, "lineal": 1, "arbol": 2, "arbol_balanceado": 3, "rf": 4, "xgb": 5, "lgbm": 6}
    ovar = {"baseline": 0, "defecto": 1, "ajustado": 2, "ajustado_con_parametros_de_7d": 3}
    d = d.assign(o1=d.modelo.map(orden), o2=d.variante.map(ovar))
    return d.sort_values(["o1", "o2", "tratamiento", "sisam"]).drop(columns=["o1", "o2"]).values.tolist()


NOM = {"persistencia": "Persistencia", "lineal": "Regresión Lineal", "arbol": "Árbol de Regresión",
       "arbol_balanceado": "Árbol balanceado", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
NOM_F2 = dict(NOM, arbol="Árbol de decisión")
VAR = {"baseline": "línea base", "defecto": "por defecto", "ajustado": "ajustado",
       "ajustado_con_parametros_de_7d": "ajustado con parámetros de 7 d"}
TRAT = {"no aplica": "", "pesos": "pesos de clase", "sin_remuestreo": "sin remuestreo", "smote": "SMOTE"}

FINAL = {t: META["modelos"][k] for t, k in [("f1_7d", "f1_7d"), ("f1_30d", "f1_30d"), ("f2", "f2")]}


def es_final(frente, modelo, variante, trat, sisam):
    f = FINAL[frente]
    alg = {"Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgbm"}[f["algoritmo"]]
    return (modelo == alg and variante == "ajustado" and sisam == ("con" if f["lista_variables"] == "con_sisam" else "sin")
            and (frente != "f2" or trat == (f["tratamiento_desbalance"] or "sin_remuestreo")))


def guardar(df, nombre, nota=None):
    df.to_csv(TAB / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    cols = list(df.columns)

    def fmt(x):
        if isinstance(x, float):
            return "" if math.isnan(x) else f"{x:.3f}"
        return str(x).replace("|", "/")
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lineas += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in df.itertuples(index=False)]
    if nota:
        lineas += ["", nota]
    (TAB / f"{nombre}.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("tabla", nombre, df.shape)


def guardar_fig(fig, nombre):
    fig.savefig(FIG / f"{nombre}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura", nombre)


# ================================================================ TABLAS
def tabla_comparativa(frente, metricas):
    filas = []
    for fuente, modelo, variante, trat, sisam in combos(frente):
        nombres = NOM_F2 if frente == "f2" else NOM
        r = {"modelo": nombres[modelo] + (" (final)" if es_final(frente, modelo, variante, trat, sisam) else ""),
             "variante": VAR[variante], "tratamiento_desbalance": TRAT[trat], "SISAM": sisam}
        if frente != "f2":
            del r["tratamiento_desbalance"]
        for et, m in metricas:
            r[f"{et} val. media"] = v(frente, modelo, variante, trat, sisam, "cv_promedio", m)
            r[f"{et} val. desv."] = v(frente, modelo, variante, trat, sisam, "cv_desviacion", m)
            r[f"{et} 2025"] = v(frente, modelo, variante, trat, sisam, "prueba_2025", m)
        filas.append(r)
    return pd.DataFrame(filas)


def tablas():
    # 1 criterios de exito
    f7, f30, f2 = META["modelos"]["f1_7d"]["metricas"], META["modelos"]["f1_30d"]["metricas"], META["modelos"]["f2"]["metricas"]
    filas = []

    def fila(tarea, crit, vval, v25, cval, c25, menor):
        ok = (lambda x, c: x <= c) if menor else (lambda x, c: x >= c)
        filas.append({"tarea": tarea, "criterio": crit, "valor_validacion": vval, "valor_2025": v25,
                      "cumple_validacion": "Sí" if ok(vval, cval) else "No",
                      "cumple_2025": ("Sí" if ok(v25, c25) else "No") if c25 is not None else "Referencia: " + ("Sí" if ok(v25, cval) else "No")})
    fila("Frente 1 a 7 días", f"MAE ≤ {CRIT['f1_7d'][0]} (val.) y ≤ {CRIT['f1_7d'][1]} (2025)",
         f7["validacion"]["MAE"]["media"], f7["prueba_2025"]["MAE"], CRIT["f1_7d"][0], CRIT["f1_7d"][1], True)
    fila("Frente 1 a 30 días", f"MAE ≤ {CRIT['f1_30d'][0]} (val.) y ≤ {CRIT['f1_30d'][1]} (2025)",
         f30["validacion"]["MAE"]["media"], f30["prueba_2025"]["MAE"], CRIT["f1_30d"][0], CRIT["f1_30d"][1], True)
    fila("Frente 2: F1 macro", f"F1 macro ≥ {CRIT['f2_f1']}", f2["validacion"]["F1_macro"]["media"],
         f2["prueba_2025"]["F1_macro"], CRIT["f2_f1"], None, False)
    fila("Frente 2: recall de Crítico", f"Recall de Crítico ≥ {CRIT['f2_rc']}", f2["validacion"]["recall_critico"]["media"],
         f2["prueba_2025"]["recall_critico"], CRIT["f2_rc"], None, False)
    guardar(pd.DataFrame(filas), "criterios_exito",
            "Nota: en el Frente 2 los mismos valores sirven de referencia en 2025. Modelos finales según ia_models/modelos_metadata.json.")

    # 2 y 3 comparativas
    t1 = []
    for fr, h in [("f1_7d", "7 días"), ("f1_30d", "30 días")]:
        t = tabla_comparativa(fr, [("MAE", "MAE"), ("RMSE", "RMSE"), ("R² (log1p)", "R2_log1p")])
        t.insert(0, "horizonte", h)
        t1.append(t)
    guardar(pd.concat(t1, ignore_index=True), "comparativa_frente1",
            "MAE y RMSE en focos de calor por municipio y semana; R² medido sobre log1p de los focos.")
    guardar(tabla_comparativa("f2", [("Accuracy", "accuracy"), ("F1 macro", "F1_macro"), ("Recall Crítico", "recall_critico"),
                                     ("Error ordinal", "error_ordinal"), ("ROC-AUC macro", "AUC_macro")]),
            "comparativa_frente2",
            "Evaluado en filas con clase de la semana actual. ROC-AUC macro no aplica a la persistencia ni al árbol de la línea base (vacío).")

    # 4 hiperparametros
    espacio = {
        "rf": "n_estimators 200-600; max_depth 6-20; min_samples_leaf 1-50; max_features {sqrt, 0.3, 0.5}",
        "xgb": "n_estimators 100-800; learning_rate 0.02-0.2 (log); max_depth 3-10; min_child_weight 1-20; subsample 0.6-1.0; colsample_bytree 0.5-1.0; reg_alpha 1e-3-10 (log); reg_lambda 1e-2-20 (log)",
        "lgbm": "num_leaves 15-127; learning_rate 0.02-0.2 (log); n_estimators 100-800; min_child_samples 20-200; subsample 0.6-1.0; colsample_bytree 0.5-1.0; reg_alpha 1e-3-10 (log); reg_lambda 1e-2-20 (log)"}
    filas = []
    for (tarea, modelo, trat), g in TUN.groupby(["tarea", "modelo", "tratamiento"]):
        mejor = g.loc[g.puntaje_validacion.idxmax() if tarea == "f2" else g.puntaje_validacion.idxmin()]
        fr = tarea
        filas.append({"tarea": tarea, "modelo": NOM[modelo], "tratamiento_desbalance": TRAT[trat],
                      "combinaciones": len(g), "espacio_de_busqueda": espacio[modelo],
                      "mejores_parametros": mejor["parametros"],
                      "puntaje_validacion": mejor["puntaje_validacion"],
                      "metrica_puntaje": "F1 macro (mayor es mejor)" if tarea == "f2" else "MAE (menor es mejor)",
                      "modelo_final": "Sí" if es_final(fr, modelo, "ajustado", trat if tarea == "f2" else "no aplica", "sin") or
                      (tarea == "f1_7d" and False) else ""})
    h = pd.DataFrame(filas)
    # marca de modelo final segun metadato (f1_7d y f2)
    for tk in ("f1_7d", "f2"):
        f = FINAL[tk]
        alg = NOM[{"Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgbm"}[f["algoritmo"]]]
        m = (h.tarea == tk) & (h.modelo == alg)
        if tk == "f2":
            m &= h.tratamiento_desbalance == TRAT[f["tratamiento_desbalance"]]
        h.loc[m, "modelo_final"] = "Sí"
    filas = []
    for tk, f in FINAL.items():
        filas.append({"tarea": tk, "modelo": f["algoritmo"] + " (final)", "tratamiento_desbalance": TRAT.get(f["tratamiento_desbalance"] or "no aplica", ""),
                      "combinaciones": "", "espacio_de_busqueda": "", "mejores_parametros": json.dumps(f["parametros"]),
                      "puntaje_validacion": "", "metrica_puntaje": f["origen_parametros"], "modelo_final": "Sí (metadato)"})
    h = pd.concat([h, pd.DataFrame(filas)], ignore_index=True)
    guardar(h, "hiperparametros", "Búsqueda aleatoria de 30 combinaciones con 4 folds. A 30 días se reutilizan los parámetros ajustados a 7 días (los finales están en las últimas filas, del metadato).")

    # 5 KPI: se leen tal cual de docs/fichas_kpi.json (sin tramos propios)
    fichas = json.load(open(RAIZ / "docs" / "fichas_kpi.json", encoding="utf8"))
    filas = []
    for f in fichas:
        bu = f["base_del_umbral"]
        filas.append({"KPI": f["nombre"], "qué_mide": f["que_mide"], "verde": f["umbral"]["verde"],
                      "ámbar": f["umbral"]["ambar"], "rojo": f["umbral"]["rojo"],
                      "base_del_umbral (dato pasado)": bu.get("dato pasado", ""),
                      "base_del_umbral (negocio)": bu.get("experiencia del negocio", ""),
                      "fuente": f["fuente_dato"], "acción_sugerida": f["accion_sugerida"]})
    guardar(pd.DataFrame(filas), "kpi_umbrales",
            "Fuente: docs/fichas_kpi.json. Los umbrales se copian tal cual; \"No se usa\" en ámbar significa que ese KPI tiene solo verde y rojo.")

    # 6 ablacion SISAM
    filas = []
    for fr, nom in [("f1_7d", "Frente 1 a 7 días"), ("f1_30d", "Frente 1 a 30 días")]:
        for mo in ("rf", "xgb", "lgbm"):
            con = v(fr, mo, "ajustado", "no aplica", "con", "cv_promedio", "MAE")
            sin = v(fr, mo, "ajustado", "no aplica", "sin", "cv_promedio", "MAE")
            filas.append({"tarea": nom, "modelo": NOM[mo], "métrica": "MAE (menor es mejor)", "con_SISAM_val": con, "sin_SISAM_val": sin,
                          "mejora_con_SISAM_%": (sin - con) / sin * 100,
                          "con_SISAM_2025": v(fr, mo, "ajustado", "no aplica", "con", "prueba_2025", "MAE"),
                          "sin_SISAM_2025": v(fr, mo, "ajustado", "no aplica", "sin", "prueba_2025", "MAE")})
    for mo in ("rf", "xgb", "lgbm"):
        for tr in ("sin_remuestreo", "pesos", "smote"):
            con = v("f2", mo, "ajustado", tr, "con", "cv_promedio", "F1_macro")
            sin = v("f2", mo, "ajustado", tr, "sin", "cv_promedio", "F1_macro")
            filas.append({"tarea": f"Frente 2 ({TRAT[tr]})", "modelo": NOM[mo], "métrica": "F1 macro (mayor es mejor)", "con_SISAM_val": con,
                          "sin_SISAM_val": sin, "mejora_con_SISAM_%": (con - sin) / sin * 100,
                          "con_SISAM_2025": v("f2", mo, "ajustado", tr, "con", "prueba_2025", "F1_macro"),
                          "sin_SISAM_2025": v("f2", mo, "ajustado", tr, "sin", "prueba_2025", "F1_macro")})
    ab = pd.DataFrame(filas)
    dec = {"Frente 1 a 7 días": FINAL["f1_7d"], "Frente 1 a 30 días": FINAL["f1_30d"]}
    ab["decisión"] = ab.tarea.map(lambda t: ("Con SISAM" if dec[t]["lista_variables"] == "con_sisam" else "Sin SISAM") if t in dec
                                  else ("Con SISAM" if FINAL["f2"]["lista_variables"] == "con_sisam" else "Sin SISAM"))
    guardar(ab, "ablacion_sisam",
            "Regla: SISAM entra solo si mejora la validación al menos 1 % (MAE) o 0.01 (F1 macro) y en al menos 3 de 4 folds. "
            "La decisión sale de ia_models/modelos_metadata.json (lista_variables del modelo final); 2025 no interviene. Mejora en %: positivo favorece a SISAM.")

    # 7 definiciones
    d = [
        ("MAE", "Error absoluto medio entre focos predichos y reales (focos por municipio y semana)", "Menor es mejor; está en focos de calor, la misma unidad del objetivo."),
        ("RMSE", "Raíz del error cuadrático medio, en focos", "Menor es mejor; castiga más los errores grandes (picos)."),
        ("R² (log1p)", "Proporción de la variabilidad de log(1 + focos) que explica el modelo", "Más cerca de 1 es mejor; en escala logarítmica, para no dominar por picos."),
        ("Accuracy", "Proporción de semanas con la clase de riesgo bien predicha", "Mayor es mejor; engañosa si las clases están desbalanceadas."),
        ("F1 macro", "Promedio del F1 (media de precisión y recall) de las 4 clases, con igual peso", "Mayor es mejor; trata igual a las clases raras, como Crítico."),
        ("Recall (de Crítico)", "De las semanas realmente Crítico, cuántas detectó el modelo", "Mayor es mejor; mide cuántos casos graves no se pierden."),
        ("ROC-AUC macro", "Área bajo la curva ROC de cada clase contra el resto, promediada", "0.5 es azar y 1 es perfecto; mide la separación sin fijar un umbral."),
        ("Error ordinal", "Distancia media entre clase real y predicha en la escala Bajo a Crítico (0 a 3)", "Menor es mejor; confundir Bajo con Crítico pesa más que Alto con Crítico."),
        ("Recall de alerta", "De las semanas-municipio con nivel Muy elevado real, cuántas alertó el modelo (k × focos predichos > p75 de la temporada)", "Mayor es mejor; es el KPI de detección."),
        ("Tasa de falsas alarmas", "De las semanas-municipio sin alerta real, cuántas marcó el modelo en alerta", "Menor es mejor; mide el costo de alertar de más."),
    ]
    guardar(pd.DataFrame(d, columns=["métrica", "qué_mide", "cómo_se_lee"]), "metricas_definiciones")

    # 8 particiones
    panel = pd.read_csv(RAIZ / "datos" / "datasets_preparation" / "dataset_panel_PREPARADO.csv", usecols=["semana", "particion"])
    cnt = panel.particion.value_counts().to_dict()
    p = PREP["particiones"]
    uso = {"entrenamiento": "Entrenamiento y validación cruzada (4 folds expansivos)", "margen": "Margen de 4 semanas; no se usa para entrenar ni evaluar",
           "prueba": "Prueba final 2025, evaluada una sola vez", "flujo_2026": "Flujo simulado 2026 (semanas subidas al dashboard)"}
    filas = []
    for k_, u in uso.items():
        n = next((c for c in cnt if k_.split("_")[0] in c.lower()), None)
        filas.append({"partición": k_, "desde": p[k_][0], "hasta": p[k_][1], "filas": cnt.get(n, float("nan")), "uso": u})
    filas.append({"partición": "descartadas", "desde": min(p["descartadas"]), "hasta": max(p["descartadas"]),
                  "filas": cnt.get(next((c for c in cnt if "descart" in c.lower()), ""), float("nan")),
                  "uso": "Semanas sin historia suficiente (inicio 2022) o incompletas (fin 2025)"})
    t = pd.DataFrame(filas)
    t["filas"] = t["filas"].map(lambda x: x if (isinstance(x, float) and math.isnan(x)) else int(x))
    guardar(t, "particiones", "Filas = municipio-semana del panel preparado, contadas por la columna particion. Valores de particion en el panel: " + ", ".join(f"{k}={n}" for k, n in cnt.items()))


# ================================================================ FIGURAS
def caja(ax, x, y, w, h, texto, fc=AZUL, tc="white", fs=10, bold=True, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def flecha(ax, p0, p1, color=GRIS_OSC, ls="-", lw=1.6):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def fig_arquitectura():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.set_title("Arquitectura de integración", loc="left")
    cajas = [("Módulo de IA\n(simulado)\nCSV semanal\n2026", GRIS, "white"), ("Ingesta y\nvalidación", AZUL, "white"),
             ("Base común\n(SQLite)", AZUL, "white"), ("Pipeline de\nvariables y\nmodelos\n(ia_models)", AZUL, "white"),
             ("Servicio / API", AZUL, "white"), ("Dashboard", NARANJA, "white")]
    xs = [0.2, 1.85, 3.5, 5.15, 6.8, 8.45]
    w, h, y = 1.35, 1.2, 2.2
    for (t, c, tc), x in zip(cajas, xs):
        caja(ax, x, y, w, h, t, fc=c, tc=tc, fs=8)
    for i in range(5):
        flecha(ax, (xs[i] + w + 0.02, y + h / 2), (xs[i + 1] - 0.02, y + h / 2))
    # retorno: la API lee la base
    flecha(ax, (7.45, y - 0.02), (4.2, y - 0.02), ls="--")
    ax.text(5.8, y - 0.35, "consulta predicciones y KPI guardados", ha="center", fontsize=8.5, color=GRIS_OSC)
    ax.add_patch(Rectangle((0.1, 0.25), 9.8, 0.95, fc="#f3f4f6", ec=GRIS, lw=1))
    ax.text(5, 0.72, "Datos públicos de INPE (DETER y BDQueimadas), NASA POWER y SISAM simulan la interfaz del módulo de IA.\n"
            "Cada semana nueva llega como un CSV con el mismo formato que el panel de entrenamiento.", ha="center", va="center", fontsize=9, color=GRIS_OSC)
    guardar_fig(fig, "arquitectura")


def fig_flujo():
    act = ["Usuario", "Dashboard", "API", "Validación", "Modelos\n(ia_models)", "Base\n(SQLite)"]
    xs = np.linspace(0.8, 9.2, len(act))
    pasos = [(0, 1, "1. Sube el CSV de la semana"), (1, 2, "2. Envía el archivo"), (2, 3, "3. Valida columnas, tipos y rangos"),
             (3, 2, "4. Semana válida"), (2, 4, "5. Calcula variables y predice (7 d, 30 d, riesgo)"),
             (4, 5, "6. Guarda predicciones y alertas"), (5, 1, "7. Muestra proyección y alertas"),
             (1, 5, "8. Semana siguiente: compara con lo observado"), (5, 1, "9. Actualiza KPI de control y estado del modelo")]
    n = len(pasos)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, n + 1.2); ax.axis("off")
    ax.set_title("Flujo: subo una semana y veo la proyección", loc="left")
    for x, a in zip(xs, act):
        caja(ax, x - 0.6, n + 0.35, 1.2, 0.6, a, fc=AZUL if a != "Usuario" else NARANJA, fs=8.5)
        ax.plot([x, x], [-0.2, n + 0.35], color=GRIS, ls="--", lw=1, zorder=0)
    for i, (a, b, t) in enumerate(pasos):
        y = n - i - 0.2
        sem = i >= 7
        col = NARANJA if sem else GRIS_OSC
        flecha(ax, (xs[a], y), (xs[b], y), color=col)
        ax.text((xs[a] + xs[b]) / 2, y + 0.12, t, ha="center", fontsize=8.5, color="black")
    ax.axhspan(0.55, 1.75, xmin=0.02, xmax=0.98, color="#fdf1e7", zorder=-1)
    ax.text(0.15, 1.45, "Una semana\ndespués", fontsize=8, color=NARANJA, fontweight="bold", va="center")
    guardar_fig(fig, "flujo_interaccion")


def fig_folds():
    d = pd.Timestamp
    fig, ax = plt.subplots(figsize=(10, 4.4))
    filas = []
    for p in PREP["pliegos"]:
        filas.append((f"Fold {p['pliego']}", d(p["entreno"][0]), d(p["entreno"][1]), d(p["validacion"][0]), d(p["validacion"][1])))
    for i, (et, e0, e1, v0, v1) in enumerate(filas):
        y = 4 - i
        ax.barh(y, (e1 - e0).days + 7, left=e0, height=0.55, color=AZUL)
        ax.barh(y, (v0 - e1).days - 7, left=e1 + pd.Timedelta(days=7), height=0.55, color=GRIS)
        ax.barh(y, (v1 - v0).days + 7, left=v0, height=0.55, color=NARANJA)
        ax.text(d("2021-12-01"), y, et, ha="right", va="center", fontsize=10)
    P = PREP["particiones"]
    y = 0
    ax.barh(y, (d(P["entrenamiento"][1]) - d(P["entrenamiento"][0])).days + 7, left=d(P["entrenamiento"][0]), height=0.55, color=AZUL)
    ax.barh(y, (d(P["margen"][1]) - d(P["margen"][0])).days + 7, left=d(P["margen"][0]), height=0.55, color=GRIS)
    ax.barh(y, (d(P["prueba"][1]) - d(P["prueba"][0])).days + 7, left=d(P["prueba"][0]), height=0.55, color=NARANJA, hatch="///", ec="white")
    ax.barh(y, (d(P["flujo_2026"][1]) - d(P["flujo_2026"][0])).days + 7, left=d(P["flujo_2026"][0]), height=0.55, color=AZUL_CLARO, hatch="..", ec="white")
    ax.text(d("2021-12-01"), y, "Final", ha="right", va="center", fontsize=10)
    ax.text(d(P["prueba"][0]) + (d(P["prueba"][1]) - d(P["prueba"][0])) / 2, -0.65, "Prueba 2025", ha="center", fontsize=9, color=NARANJA)
    ax.text(d(P["flujo_2026"][0]) + (d(P["flujo_2026"][1]) - d(P["flujo_2026"][0])) / 2, -0.65, "Flujo simulado 2026", ha="center", fontsize=9, color=AZUL)
    ax.set_yticks([]); ax.set_ylim(-1.0, 4.6)
    ax.set_xlim(d("2021-12-01"), d("2026-12-01"))
    ax.set_xlabel("Semana (fecha de inicio)")
    ax.set_title("Validación cruzada temporal expansiva y particiones", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, label="Entrenamiento"), Patch(color=GRIS, label="Margen de 4 semanas"),
                       Patch(color=NARANJA, label="Validación / prueba 2025"), Patch(color=AZUL_CLARO, label="Flujo 2026 (sin entrenar)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
    ax.spines["left"].set_visible(False)
    guardar_fig(fig, "folds_validacion")


def fig_crisp(cr):
    fases = [
        ("1. Comprensión del negocio", "Hecho", "Problema, KPI de alerta y criterios de éxito fijados antes de modelar"),
        ("2. Comprensión de los datos", "Hecho", "DETER, BDQueimadas, NASA POWER y SISAM; EDA de focos y clima"),
        ("3. Preparación de los datos", "Hecho", "Panel municipio-semana 2022 a 2025, folds y margen de 4 semanas"),
        ("4. Modelado", "Hecho", "Línea base y RF, XGBoost y LightGBM (30 combinaciones por modelo)"),
        ("5. Evaluación", "Hecho", cr),
        ("6. Despliegue", "Prototipo", "Modelos en ia_models, API y dashboard con flujo simulado 2026"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("Mapa CRISP-DM del proyecto", loc="left")
    for i, (t, e, txt) in enumerate(fases):
        c, r = i % 3, i // 3
        x, y = 0.15 + c * 3.3, 2.75 - r * 2.6
        ax.add_patch(FancyBboxPatch((x, y), 3.1, 2.3, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f3f4f6", ec=AZUL, lw=1.4))
        ax.text(x + 0.15, y + 1.95, t, fontsize=9, fontweight="bold", color=AZUL, va="center")
        ax.text(x + 0.15, y + 1.55, e, fontsize=9, fontweight="bold", color=VERDE if e == "Hecho" else NARANJA, va="center")
        import textwrap
        ax.text(x + 0.15, y + 1.25, "\n".join(textwrap.wrap(txt, 30)), fontsize=8.5, va="top", color="#202124")
    guardar_fig(fig, "crispdm_mapa")


def mejor(frente, modelos, variante, metrica, menor, sisam_opts=("con", "sin"), trat_opts=None):
    mejor_ = None
    for mo in modelos:
        for si in sisam_opts:
            for tr in (trat_opts or ["no aplica"]):
                val = v(frente, mo, variante, tr, si, "cv_promedio", metrica)
                if math.isnan(val):
                    continue
                if mejor_ is None or (val < mejor_[0] if menor else val > mejor_[0]):
                    mejor_ = (val, mo, si, tr)
    return mejor_


def fig_mae():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    series = [("Persistencia", GRIS), ("Mejor baseline", GRIS_OSC), ("Random Forest", AZUL_CLARO), ("XGBoost", "#5b8fc9"), ("LightGBM", AZUL)]
    for ax, ev, tit in [(axs[0], "cv_promedio", "Validación (promedio de 4 folds)"), (axs[1], "prueba_2025", "Prueba 2025")]:
        for gi, fr in enumerate(["f1_7d", "f1_30d"]):
            vals = []
            vals.append(v(fr, "persistencia", "baseline", "no aplica", "no aplica", ev, "MAE"))
            bl = mejor(fr, ["lineal", "arbol"], "baseline", "MAE", True)
            vals.append(v(fr, bl[1], "baseline", "no aplica", bl[2], ev, "MAE"))
            for mo in ("rf", "xgb", "lgbm"):
                b_ = mejor(fr, [mo], "ajustado", "MAE", True)
                vals.append(v(fr, mo, "ajustado", "no aplica", b_[2], ev, "MAE"))
            for si_, ((nm, c), val) in enumerate(zip(series, vals)):
                x = gi + (si_ - 2) * 0.16
                ax.bar(x, val, 0.15, color=c)
                ax.text(x, val, f"{val:.1f}" if val >= 10 else f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=6)
            cv = CRIT[fr][0] if ev == "cv_promedio" else CRIT[fr][1]
            ax.hlines(cv, gi - 0.45, gi + 0.45, color="black", ls="--", lw=1.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A 7 días", "A 30 días"])
        ax.set_title(tit, loc="left", fontsize=11)
    axs[0].set_ylabel("MAE (focos por municipio y semana)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(color=c, label=n) for n, c in series] + [Line2D([0], [0], color="black", ls="--", label="Criterio de éxito")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Error absoluto medio por modelo (menor es mejor)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    guardar_fig(fig, "comparativa_mae")


def fig_f2():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    series = [("Persistencia", GRIS), ("Árbol (línea base)", GRIS_OSC), ("Random Forest", AZUL_CLARO), ("XGBoost", "#5b8fc9"), ("LightGBM", AZUL)]
    TR = ["sin_remuestreo", "pesos", "smote"]
    for ax, (met, crit, tit) in zip(axs, [("F1_macro", CRIT["f2_f1"], "F1 macro"), ("recall_critico", CRIT["f2_rc"], "Recall de Crítico")]):
        for gi, ev in enumerate(["cv_promedio", "prueba_2025"]):
            vals = [v("f2", "persistencia", "baseline", "no aplica", "no aplica", ev, met)]
            ba = max(((v("f2", mo, "baseline", tr, si, "cv_promedio", "F1_macro"), mo, si, tr)
                      for mo, tr in (("arbol", "no aplica"), ("arbol_balanceado", "pesos")) for si in ("con", "sin")), key=lambda t: t[0])
            vals.append(v("f2", ba[1], "baseline", ba[3], ba[2], ev, met))
            for mo in ("rf", "xgb", "lgbm"):
                b_ = mejor("f2", [mo], "ajustado", "F1_macro", False, trat_opts=TR)
                vals.append(v("f2", mo, "ajustado", b_[3], b_[2], ev, met))
            for si_, ((nm, c), val) in enumerate(zip(series, vals)):
                x = gi + (si_ - 2) * 0.16
                ax.bar(x, val, 0.15, color=c)
                ax.text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=6)
            ax.hlines(crit, gi - 0.45, gi + 0.45, color="black", ls="--", lw=1.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Validación\n(4 folds)", "Prueba 2025"])
        ax.set_title(tit, loc="left", fontsize=11); ax.set_ylim(0, 0.85)
    axs[0].set_ylabel("Valor de la métrica (0 a 1)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(color=c, label=n) for n, c in series] + [Line2D([0], [0], color="black", ls="--", label="Criterio de éxito")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Frente 2: clase de riesgo de fuego (mayor es mejor)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    guardar_fig(fig, "comparativa_f2")


def alerta(df, k):
    """Recall y tasa de falsas alarmas en 2025; alerta si k x predicho > p75 de la temporada de la semana objetivo."""
    t = df.semana + pd.Timedelta(days=7)
    u = np.where(t.dt.month.between(7, 11), KPI["nivel_focos_municipio_semana"]["valor"]["cortes_por_temporada"]["seca"]["p75"],
                 KPI["nivel_focos_municipio_semana"]["valor"]["cortes_por_temporada"]["humeda"]["p75"])
    real = df.real > u
    al = k * df.predicho > u
    return (al & real).sum() / real.sum(), (al & ~real).sum() / (~real).sum()


def fig_punto_operacion():
    d = PA[(PA.frente == "f1_7d") & (PA.modelo == "lgbm_final_sin_sisam")]
    pers = PB[(PB.frente == "f1_7d") & (PB.modelo == "persistencia")]
    ks = np.round(np.arange(1.0, 3.0001, 0.1), 1)
    pts = [alerta(d, k) for k in ks]
    k0 = META["alerta"]["factor_k"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([p[1] * 100 for p in pts], [p[0] for p in pts], "-o", color=AZUL, ms=4, lw=1.8, label="LightGBM con factor k")
    for k, p in zip(ks, pts):
        if k in (1.0, 2.0, 3.0):
            ax.annotate(f"k = {k:.1f}", (p[1] * 100, p[0]), textcoords="offset points", xytext=(8, -14), fontsize=8, color=AZUL)
    i0 = list(ks).index(k0)
    ax.plot(pts[i0][1] * 100, pts[i0][0], "o", color=NARANJA, ms=11, zorder=5, label=f"k = {k0} (elegido con validación)")
    ax.annotate(f"k = {k0}", (pts[i0][1] * 100, pts[i0][0]), textcoords="offset points", xytext=(10, -14), fontsize=9, color=NARANJA, fontweight="bold")
    rp, fp = alerta(pers, 1.0)
    ax.plot(fp * 100, rp, "s", color=GRIS_OSC, ms=9, label="Persistencia (referencia)")
    ax.set_xlabel("Tasa de falsas alarmas (% de semanas-municipio sin alerta real)")
    ax.set_ylabel("Recall de alerta (proporción detectada)")
    ax.set_title("Punto de operación de la alerta (prueba 2025)", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    guardar_fig(fig, "punto_operacion_alerta")
    return {"k1.4": pts[i0], "pers": (rp, fp)}


def _num(txt):
    import re
    return float(re.search(r"\d+(?:\.\d+)?", txt.replace(",", "")).group())


def fig_semaforo():
    """Semaforo de los 3 KPI de control con los umbrales exactos de docs/fichas_kpi.json (solo verde y rojo)."""
    fichas = {f["id"]: f for f in json.load(open(RAIZ / "docs" / "fichas_kpi.json", encoding="utf8"))}
    c2025 = META["kpi_control"]["comprobacion_2025"]
    items = [
        ("control_recall_alerta", "Recall de alerta (mayor es mejor; ventana de 8 semanas)", False, 1.0, c2025["recall_alerta"]),
        ("control_falsas_alarmas", "Tasa de falsas alarmas (menor es mejor; ventana de 8 semanas)", True, 0.08, c2025["tasa_falsas_alarmas"]),
        ("control_mae_movil", "MAE móvil de 4 semanas, en focos (menor es mejor)", True, 10, META["modelos"]["f1_7d"]["metricas"]["prueba_2025"]["MAE"]),
    ]
    fig, axs = plt.subplots(3, 1, figsize=(9, 5.2))
    for ax, (fid, tit, menor, hi, v25) in zip(axs, items):
        u = _num(fichas[fid]["umbral"]["verde"])
        cortes = [0, u, hi]
        cols = [VERDE, ROJO] if menor else [ROJO, VERDE]
        for a_, b_, c in zip(cortes[:-1], cortes[1:], cols):
            ax.barh(0, b_ - a_, left=a_, color=c, height=0.5)
        ax.text(u, 0.42, f"{u:g}", ha="center", fontsize=9)
        ax.plot(v25, -0.38, "^", color="white", mec="black", ms=8)
        ax.text(v25, -0.62, "2025 (global)", ha="center", fontsize=8, va="top")
        ax.set_xlim(0, hi); ax.set_ylim(-1.0, 0.8); ax.set_yticks([])
        ax.set_title(tit, loc="left", fontsize=10)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Semáforo de los KPI de control (umbrales de las fichas de KPI)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    guardar_fig(fig, "semaforo_kpi")


if __name__ == "__main__":
    tablas()
    cr = pd.read_csv(TAB / "criterios_exito.csv")
    ok = (cr.cumple_validacion == "Sí").sum()
    fig_arquitectura()
    fig_flujo()
    fig_folds()
    fig_crisp(f"Criterios de éxito cumplidos en validación: {ok} de {len(cr)}; alerta con k = {META['alerta']['factor_k']}")
    fig_mae()
    fig_f2()
    res = fig_punto_operacion()
    fig_semaforo()
    print("punto de operación:", res)
    print("OK")
