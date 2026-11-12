"""KPI de resultado y de control de BIODIVERSITY-GUARD-Predict.

Funciones puras: reciben DataFrames y devuelven resultados. No usan la base SQLite.
Los umbrales se leen de `datos/datasets_preparation/*.json` y de `ia_models/modelos_metadata.json`.
Estado posible de un KPI: verde, ámbar, rojo, o sin_dato (valor None) cuando falta información.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_KPI = RAIZ / "datos" / "datasets_preparation" / "kpi_parametros.json"
RUTA_PARAM = RAIZ / "datos" / "datasets_preparation" / "parametros_preparacion.json"
RUTA_META = RAIZ / "ia_models" / "modelos_metadata.json"
RUTA_FICHAS = RAIZ / "docs" / "fichas_kpi.json"

NIVELES = ["Sin focos", "Habitual", "Elevado", "Muy elevado"]
HORIZONTE_DIAS = 7  # el modelo de focos predice la semana siguiente
VENTANA_INTERANUAL = 4  # semanas
BASE_MINIMA_INTERANUAL = 20  # focos mínimos del periodo anterior para calcular una variación (evita razones sin sentido)
INTERANUAL_AMBAR_PCT = 20.0  # supuesto del negocio indicado en la tarea
NIVEL_VERDE_MAX_PCT = 25.0  # por construcción, 25 % de las semanas con focos supera el p75
NIVEL_AMBAR_MAX_PCT = 40.0
MIN_ALERTAS_REALES_RECALL = 30  # con menos alertas reales en la ventana el recall no es evaluable (se deja sin dato)


@lru_cache(maxsize=1)
def _cargar():
    kpi = json.loads(RUTA_KPI.read_text(encoding="utf-8"))
    par = json.loads(RUTA_PARAM.read_text(encoding="utf-8"))
    meta = json.loads(RUTA_META.read_text(encoding="utf-8"))
    return kpi, par, meta


def _kpi_par():
    return _cargar()[0]


def _cortes():
    return _cargar()[1]["kpi_cortes_focos"]


def _fin_entrenamiento():
    return pd.Timestamp(_cargar()[1]["particiones"]["entrenamiento"][1])


def _inicio_entrenamiento():
    return pd.Timestamp(_cargar()[1]["particiones"]["entrenamiento"][0])


def _es_seca(fechas):
    # temporada seca = julio a noviembre
    return pd.to_datetime(fechas).dt.month.between(7, 11) if isinstance(fechas, pd.Series) else pd.Timestamp(fechas).month in range(7, 12)


# ---------------------------------------------------------------- nivel de focos
def nivel_focos_valor(focos, semana):
    """Nivel de un municipio-semana con los cortes de la temporada de `semana`."""
    if focos is None or pd.isna(focos):
        return None
    c = _cortes()["seca" if _es_seca(semana) else "humeda"]
    if focos <= 0:
        return NIVELES[0]
    if focos <= c["p50"]:
        return NIVELES[1]
    if focos <= c["p75"]:
        return NIVELES[2]
    return NIVELES[3]


def nivel_focos(df_semana, col_focos="num_incendios", col_semana="semana"):
    """Agrega la columna `nivel_focos` (Sin focos, Habitual, Elevado, Muy elevado) segun la temporada de cada semana."""
    d = df_semana.copy()
    if d.empty:
        d["nivel_focos"] = pd.Series(dtype=object)
        return d
    seca = pd.to_datetime(d[col_semana]).dt.month.between(7, 11)
    c = _cortes()
    p50 = np.where(seca, c["seca"]["p50"], c["humeda"]["p50"])
    p75 = np.where(seca, c["seca"]["p75"], c["humeda"]["p75"])
    f = pd.to_numeric(d[col_focos], errors="coerce")
    nivel = np.select([f <= 0, f <= p50, f <= p75, f > p75], NIVELES[:1] + NIVELES[1:], default="")
    d["nivel_focos"] = np.where(f.isna(), None, nivel)
    return d


# ---------------------------------------------------------------- utilidades de KPI
def _estado(valor, ambar, rojo, estricto=False):
    if valor is None or pd.isna(valor):
        return "sin_dato"
    if (valor < ambar) if estricto else (valor <= ambar):
        return "verde"
    return "ambar" if valor <= rojo else "rojo"


def _kpi(id_, nombre, valor, unidad, ambar, rojo, base, lectura, estado=None, estricto=False, **extra):
    if valor is not None and not pd.isna(valor):
        valor = round(float(valor), 3)
    else:
        valor = None
    d = {"id": id_, "nombre": nombre, "valor": valor, "unidad": unidad,
         "estado": estado or _estado(valor, ambar, rojo, estricto),
         "umbral_ambar": ambar, "umbral_rojo": rojo, "base_umbral": base, "lectura": lectura}
    d.update(extra)
    return d


def _sin_dato(id_, nombre, unidad, ambar, rojo, base, motivo, **extra):
    return _kpi(id_, nombre, None, unidad, ambar, rojo, base, f"Sin dato: {motivo}.", estado="sin_dato", **extra)


def _prep_hist(df):
    d = df.copy()
    d["semana"] = pd.to_datetime(d["semana"])
    return d


def _col(d, nombre):
    return pd.to_numeric(d[nombre], errors="coerce") if nombre in d.columns else pd.Series(np.nan, index=d.index)


def _alerta_pred(focos_7d, semana_base):
    """Alerta del modelo: factor_k x focos predichos > umbral de la temporada de la semana objetivo."""
    a = _cargar()[2]["alerta"]
    objetivo = pd.to_datetime(semana_base) + pd.Timedelta(days=HORIZONTE_DIAS)
    seca = objetivo.dt.month.between(7, 11) if isinstance(objetivo, pd.Series) else objetivo.month in range(7, 12)
    umbral = np.where(seca, a["umbral_seca"], a["umbral_humeda"])
    return (a["factor_k"] * np.asarray(focos_7d, dtype=float) > umbral)


def _normaliza_pred(pred):
    p = pred.copy()
    if "semana_base" not in p.columns and "semana" in p.columns:
        p = p.rename(columns={"semana": "semana_base"})
    if "focos_7d" not in p.columns and "predicho" in p.columns:
        p = p.rename(columns={"predicho": "focos_7d"})
    p["semana_base"] = pd.to_datetime(p["semana_base"])
    return p


# ---------------------------------------------------------------- análisis interanual
def variacion_interanual_por_estado(obs_historico, semana):
    """Focos de las últimas 4 semanas frente al mismo periodo del año anterior, por estado (uf). Devuelve DataFrame."""
    h = _prep_hist(obs_historico)
    semana = pd.Timestamp(semana)
    fin_a, ini_a = semana, semana - pd.Timedelta(weeks=VENTANA_INTERANUAL - 1)
    fin_b, ini_b = fin_a - pd.Timedelta(days=364), ini_a - pd.Timedelta(days=364)
    ahora = h[(h.semana >= ini_a) & (h.semana <= fin_a)].groupby("uf")["num_incendios"].sum()
    antes = h[(h.semana >= ini_b) & (h.semana <= fin_b)]
    n_sem_antes = antes.groupby("uf")["semana"].nunique()
    antes = antes.groupby("uf")["num_incendios"].sum()
    t = pd.DataFrame({"focos_actual": ahora, "focos_anio_anterior": antes}).reindex(sorted(set(ahora.index) | set(antes.index)))
    t["semanas_anio_anterior"] = n_sem_antes.reindex(t.index)
    ok = (t.focos_anio_anterior >= BASE_MINIMA_INTERANUAL) & (t.semanas_anio_anterior >= VENTANA_INTERANUAL) & t.focos_actual.notna()
    t["variacion_pct"] = np.where(ok, (t.focos_actual - t.focos_anio_anterior) / t.focos_anio_anterior.where(ok) * 100, np.nan)
    return t.reset_index().rename(columns={"index": "uf"})


@lru_cache(maxsize=1)
def analisis_interanual_historico():
    """Distribucion de la variación interanual (ventana de 4 semanas) por estado en 2023 y 2024, con el panel completo.

    Solo usa semanas de entrenamiento. Sirve para documentar en que percentil cae la referencia de 40 %.
    """
    ruta = RAIZ / "datos" / "datasets_eda" / "dataset_panel_semanal_COMPLETO.csv"
    pa = pd.read_csv(ruta, parse_dates=["semana"], usecols=["semana", "uf", "num_incendios"])
    pa = pa[(pa.semana >= _inicio_entrenamiento()) & (pa.semana <= _fin_entrenamiento())]
    w = pa.groupby(["semana", "uf"])["num_incendios"].sum().unstack().asfreq("7D").fillna(0)
    r4 = w.rolling(VENTANA_INTERANUAL).sum()
    prev = r4.shift(52)
    v = ((r4 - prev) / prev.where(prev >= BASE_MINIMA_INTERANUAL) * 100).loc["2023-01-01":].stack().dropna()
    ref = _kpi_par()["variacion_interanual_de_focos"]["valor"]["referencia_pct"]
    return {
        "n_observaciones": int(len(v)),
        "referencia_pct": ref,
        "percentil_de_la_referencia": round(float((v <= ref).mean() * 100), 1),
        "porcentaje_que_supera_referencia": round(float((v > ref).mean() * 100), 1),
        "porcentaje_que_supera_ambar": round(float((v > INTERANUAL_AMBAR_PCT).mean() * 100), 1),
        "p50": round(float(v.quantile(0.5)), 1), "p75": round(float(v.quantile(0.75)), 1),
        "p90": round(float(v.quantile(0.9)), 1),
        "p50_julio_a_noviembre": round(float(v[v.index.get_level_values(0).month.isin(range(7, 12))].quantile(0.5)), 1),
    }


@lru_cache(maxsize=1)
def umbrales_deter():
    """p75 y p90 del área semanal DETER (suma de los estados) en entrenamiento 2022 a 2024, con el panel completo."""
    ruta = RAIZ / "datos" / "datasets_eda" / "dataset_panel_semanal_COMPLETO.csv"
    pa = pd.read_csv(ruta, parse_dates=["semana"], usecols=["semana", "deter_area_ha"])
    pa = pa[(pa.semana >= _inicio_entrenamiento()) & (pa.semana <= _fin_entrenamiento())]
    s = pa.groupby("semana")["deter_area_ha"].sum()
    return float(s.quantile(0.75)), float(s.quantile(0.90))


# ---------------------------------------------------------------- KPI de resultado
def kpis_resultado(obs_semana, pred_semana, obs_historico, semana):
    """Los 7 KPI de resultado (4 de resultado y 3 de contexto) para la semana `semana`."""
    semana = pd.Timestamp(semana)
    par = _kpi_par()
    obs = obs_semana if obs_semana is not None else pd.DataFrame()
    pred = _normaliza_pred(pred_semana) if pred_semana is not None and len(pred_semana) else pd.DataFrame()
    base_pasado = "dato pasado 2022 a 2024"
    out = []

    # 1. nivel de focos
    base1 = f"{base_pasado}: cortes p50 y p75 por temporada (kpi_parametros.json); semáforo del porcentaje: negocio"
    n1 = "Nivel de focos por municipio (semana observada)"
    if obs.empty or _col(obs, "num_incendios").dropna().empty:
        out.append(_sin_dato("nivel_focos", n1, "% de municipios con focos", NIVEL_VERDE_MAX_PCT, NIVEL_AMBAR_MAX_PCT, base1, "no hay focos observados en la semana"))
    else:
        o = obs.copy()
        o["semana"] = semana
        o = nivel_focos(o)
        dist = o["nivel_focos"].value_counts().reindex(NIVELES).fillna(0).astype(int)
        con_focos = int(dist[NIVELES[1:]].sum())
        if con_focos == 0:
            out.append(_kpi("nivel_focos", n1, 0.0, "% de municipios con focos", NIVEL_VERDE_MAX_PCT, NIVEL_AMBAR_MAX_PCT, base1,
                            "La semana no tuvo municipios con focos.", distribucion=dist.to_dict()))
        else:
            pct = dist["Muy elevado"] / con_focos * 100
            out.append(_kpi("nivel_focos", n1, pct, "% de municipios con focos", NIVEL_VERDE_MAX_PCT, NIVEL_AMBAR_MAX_PCT, base1,
                            f"De {con_focos} municipios con focos, {int(dist['Muy elevado'])} ({pct:.0f} %) tuvieron un nivel Muy elevado para esta temporada.",
                            distribucion=dist.to_dict()))

    # 2. municipios en alerta frente a la capacidad
    cap = par["capacidad_de_atencion_municipios_en_alerta"]["valor"]
    base2 = "negocio: " + par["capacidad_de_atencion_municipios_en_alerta"]["fuente"]
    n2 = "Municipios en alerta frente a la capacidad de respuesta"
    amb2, roj2 = cap["base"], cap["sensibilidad"][1]
    if pred.empty:
        out.append(_sin_dato("alertas_vs_capacidad", n2, "municipios", amb2, roj2, base2, "no hay predicciones para la semana", capacidad_usada=cap["base"]))
    else:
        alertas = int(_alerta_pred(pred["focos_7d"], pred["semana_base"]).sum())
        e2 = _estado(alertas, amb2, roj2)
        txt = {"verde": "dentro de la capacidad", "ambar": "por encima de la capacidad base", "rojo": "muy por encima de la capacidad"}[e2]
        out.append(_kpi("alertas_vs_capacidad", n2, alertas, "municipios", amb2, roj2, base2,
                        f"Hay {alertas} municipios en alerta para la semana proyectada, {txt} de {cap['base']} municipios.",
                        capacidad_usada=cap["base"], sensibilidad=cap["sensibilidad"]))

    # 3. variación interanual por estado
    ref = par["variacion_interanual_de_focos"]["valor"]["referencia_pct"]
    base3 = f"negocio: {par['variacion_interanual_de_focos']['fuente']}; ámbar {INTERANUAL_AMBAR_PCT:.0f} % supuesto"
    n3 = "Variación interanual de focos por estado (4 semanas)"
    try:
        t = variacion_interanual_por_estado(obs_historico, semana) if obs_historico is not None and len(obs_historico) else pd.DataFrame()
    except Exception:
        t = pd.DataFrame()
    v = t.dropna(subset=["variacion_pct"]) if not t.empty else t
    if v is None or v.empty:
        out.append(_sin_dato("variacion_interanual", n3, "%", INTERANUAL_AMBAR_PCT, ref, base3, "no hay datos del año anterior"))
    else:
        peor = v.loc[v.variacion_pct.idxmax()]
        por_estado = {r.uf: (None if pd.isna(r.variacion_pct) else round(float(r.variacion_pct), 1)) for r in t.itertuples()}
        out.append(_kpi("variacion_interanual", n3, peor.variacion_pct, "%", INTERANUAL_AMBAR_PCT, ref, base3,
                        f"El estado con más aumento es {peor.uf}: {peor.variacion_pct:+.0f} % frente al mismo periodo del año anterior.",
                        por_estado=por_estado))

    # 4. anticipación
    dias_min = par["anticipacion_minima"]["valor"]["dias"]
    out.append(_kpi("anticipacion_alerta", "Anticipación de la alerta", HORIZONTE_DIAS, "días", None, dias_min,
                    "negocio: " + par["anticipacion_minima"]["fuente"],
                    f"El modelo avisa con {HORIZONTE_DIAS} días de anticipación; el mínimo esperado es {dias_min} días.",
                    estado="verde" if HORIZONTE_DIAS >= dias_min else "rojo"))

    # 5. riesgo de fuego INPE
    cortes = par["referencias_de_contexto"]["valor"]["riesgo_fuego_inpe"]
    base5 = "negocio: cortes del INPE (referencias_de_contexto); el índice se deriva del clima"
    n5 = "Riesgo de fuego INPE (contexto)"
    r = _col(obs, "riesgo_fuego_promedio") if not obs.empty else pd.Series(dtype=float)
    r5 = pd.DataFrame({"uf": obs["uf"], "v": r}).dropna() if not obs.empty else pd.DataFrame()
    if r5.empty:
        out.append(_sin_dato("riesgo_fuego_inpe", n5, "índice 0 a 1", cortes[1], cortes[2], base5, "sin dato de riesgo en la semana", cortes_inpe=cortes))
    else:
        e = r5.groupby("uf")["v"].mean()
        out.append(_kpi("riesgo_fuego_inpe", n5, e.max(), "índice 0 a 1", cortes[1], cortes[2], base5,
                        f"El estado con mayor riesgo de fuego promedio es {e.idxmax()} ({e.max():.2f}); el INPE lo calcula con el clima.",
                        estricto=True, por_estado={k: round(float(x), 3) for k, x in e.items()}, cortes_inpe=cortes))

    # 6. PM2.5
    oms, conama = par["referencias_de_contexto"]["valor"]["pm2_5_oms_ug_m3"], par["referencias_de_contexto"]["valor"]["pm2_5_conama_pi1_ug_m3"]
    base6 = "negocio: OMS (15) y CONAMA PI-1 (60)"
    n6 = "Calidad del aire, PM2.5 (contexto)"
    p6 = pd.DataFrame({"uf": obs["uf"], "v": _col(obs, "pm2_5_reanalise")}).dropna() if not obs.empty else pd.DataFrame()
    if p6.empty:
        out.append(_sin_dato("calidad_aire_pm25", n6, "ug/m3", oms, conama, base6, "sin dato de PM2.5 en la semana"))
    else:
        e = p6.groupby("uf")["v"].mean()
        out.append(_kpi("calidad_aire_pm25", n6, e.max(), "ug/m3", oms, conama, base6,
                        f"El estado con peor aire es {e.idxmax()}: {e.max():.1f} ug/m3 de PM2.5 (la OMS recomienda hasta {oms}).",
                        por_estado={k: round(float(x), 2) for k, x in e.items()}))

    # 7. DETER
    p75, p90 = umbrales_deter()
    base7 = f"{base_pasado}: p75 y p90 del área semanal DETER, solo entrenamiento. Limitacion: el DETER del Hito 2 estaba truncado; aquí se usa el panel completo"
    n7 = "Alertas de deforestación DETER (contexto)"
    a7 = _col(obs, "deter_area_ha").dropna() if not obs.empty else pd.Series(dtype=float)
    if a7.empty:
        out.append(_sin_dato("deforestacion_deter", n7, "ha en la semana", round(p75, 1), round(p90, 1), base7, "sin dato DETER en la semana"))
    else:
        tot = a7.sum()
        out.append(_kpi("deforestacion_deter", n7, tot, "ha en la semana", round(p75, 1), round(p90, 1), base7,
                        f"Se alertaron {tot:,.0f} ha de deforestación en la semana; lo habitual de 2022 a 2024 es hasta {p75:,.0f} ha."))
    return out


# ---------------------------------------------------------------- tabla de alertas
def tabla_alertas(pred_semana, obs_semana):
    """Tabla por municipio ordenada por prioridad (alertas primero, de mayor a menor focos_7d)."""
    p = _normaliza_pred(pred_semana)
    cols = ["municipio_norm", "uf", "nivel_focos", "focos_7d", "focos_30d", "clase_riesgo", "prob_critico", "es_alerta", "prioridad"]
    if p.empty:
        return pd.DataFrame(columns=cols)
    for c in ("focos_30d", "clase_riesgo", "prob_critico"):
        if c not in p.columns:
            p[c] = np.nan
    p["es_alerta"] = _alerta_pred(p["focos_7d"], p["semana_base"]).astype(int)
    if obs_semana is not None and len(obs_semana):
        o = obs_semana[["municipio_norm", "uf", "num_incendios"]].copy()
        o["semana"] = p["semana_base"].iloc[0]
        o = nivel_focos(o)[["municipio_norm", "uf", "nivel_focos"]].drop_duplicates(["municipio_norm", "uf"])
        p = p.merge(o, on=["municipio_norm", "uf"], how="left")
    else:
        p["nivel_focos"] = None
    p["nivel_focos"] = p["nivel_focos"].where(p["nivel_focos"].notna(), "Sin dato")
    p = p.sort_values(["es_alerta", "focos_7d"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    p["prioridad"] = np.arange(1, len(p) + 1)
    return p[cols]


# ---------------------------------------------------------------- KPI de control
def umbrales_control():
    """Umbrales de control. Si existe `ventanas_moviles` en el metadato (calibrado para ventanas móviles) se usan esos."""
    m = _cargar()[2]
    k = m["kpi_control"]
    vm = k.get("ventanas_moviles")
    if vm:
        mae, rec, fa = vm["umbrales"]["mae_movil_max"], vm["umbrales"]["recall_alerta_min"], vm["umbrales"]["falsas_alarmas_max"]
        v_mae, v_rf = vm["ventana_mae_semanas"], vm["ventana_recall_falsas_alarmas_semanas"]
    else:
        mae, rec, fa = k["mae_movil_4_semanas_maximo"]["final"], k["recall_alerta_minimo"]["final"], k["tasa_falsas_alarmas_maximo"]["final"]
        v_mae = v_rf = 4
    return {"mae_max": mae, "recall_min": rec, "fa_max": fa, "ventana_mae": v_mae, "ventana_recall_fa": v_rf,
            "factor_k": m["alerta"]["factor_k"], "umbral_seca": m["alerta"]["umbral_seca"], "umbral_humeda": m["alerta"]["umbral_humeda"]}


def _conteos_semanales(df):
    d = df.copy()
    if "semana_base" not in d.columns:
        d = d.rename(columns={"semana": "semana_base"})
    if "predicho" not in d.columns and "focos_7d" in d.columns:
        d = d.rename(columns={"focos_7d": "predicho"})
    d["semana_base"] = pd.to_datetime(d["semana_base"])
    u = umbrales_control()
    objetivo = d["semana_base"] + pd.Timedelta(days=HORIZONTE_DIAS)
    umbral = np.where(objetivo.dt.month.between(7, 11), u["umbral_seca"], u["umbral_humeda"])
    d["err"] = (d["real"] - d["predicho"]).abs()
    d["real_alerta"] = d["real"] > umbral
    d["pred_alerta"] = u["factor_k"] * d["predicho"] > umbral
    d["tp"] = d.real_alerta & d.pred_alerta
    d["fp"] = ~d.real_alerta & d.pred_alerta
    d["neg"] = ~d.real_alerta
    g = d.groupby("semana_base").agg(err=("err", "sum"), n=("err", "size"), tp=("tp", "sum"), pos=("real_alerta", "sum"),
                                    fp=("fp", "sum"), neg=("neg", "sum")).sort_index()
    return g


def _ventanas(g, v_mae, v_rf):
    """MAE en ventana de v_mae semanas; recall y falsas alarmas en ventana de v_rf semanas (conteos juntados). Índice: semana base."""
    a = g.rolling(v_mae, min_periods=v_mae).sum()
    b = g.rolling(v_rf, min_periods=v_rf).sum()
    return pd.DataFrame({"mae_movil_4s": a.err / a.n,
                         "recall_alerta": np.where(b.pos >= MIN_ALERTAS_REALES_RECALL, b.tp / b.pos.where(b.pos > 0), np.nan),
                         "falsas_alarmas": np.where(b.neg > 0, b.fp / b.neg.where(b.neg > 0), np.nan)}, index=g.index)


def kpis_control(df_pred_vs_real):
    """MAE móvil, recall de alerta, tasa de falsas alarmas y estado del modelo por semana base.

    Entrada: columnas semana (o semana_base), real, predicho (o focos_7d) por municipio; `real` son los focos de la semana objetivo.
    El MAE usa una ventana de 4 semanas; recall y falsas alarmas una ventana de 8 semanas (con 4 había muy pocas alertas y salía ruidoso).
    Las ventanas son las de `ventanas_moviles` del metadato. Solo salen semanas con la ventana del MAE completa; el recall y las
    falsas alarmas quedan sin dato hasta tener su ventana completa (y el recall exige MIN_ALERTAS_REALES_RECALL alertas reales).
    """
    cols = ["semana_base", "mae_movil_4s", "recall_alerta", "falsas_alarmas", "estado"]
    if df_pred_vs_real is None or df_pred_vs_real.empty:
        return pd.DataFrame(columns=cols)
    u = umbrales_control()
    t = _ventanas(_conteos_semanales(df_pred_vs_real), u["ventana_mae"], u["ventana_recall_fa"]).dropna(subset=["mae_movil_4s"])
    mal_mae = t.mae_movil_4s > u["mae_max"]
    mal_rec = t.recall_alerta < u["recall_min"]
    mal_fa = t.falsas_alarmas > u["fa_max"]
    persist = (mal_mae & mal_mae.shift(1, fill_value=False)) | (mal_rec & mal_rec.shift(1, fill_value=False))
    t["estado"] = np.where(persist, "reentrenar recomendado", np.where(mal_mae | mal_rec | mal_fa, "vigilar", "normal"))
    t = t.reset_index()
    t["semana_base"] = t["semana_base"].dt.strftime("%Y-%m-%d")
    return t[cols]


def kpis_control_global(df_pred_vs_real):
    """Valores globales de todo el periodo (para contrastar con modelos_metadata.json, comprobacion_2025)."""
    g = _conteos_semanales(df_pred_vs_real).sum()
    return {"mae": float(g.err / g.n), "recall_alerta": float(g.tp / g.pos), "falsas_alarmas": float(g.fp / g.neg)}


def estado_modelo(df_kpis_control):
    """Estado del modelo en la última semana y semanas que salieron de algún umbral."""
    if df_kpis_control is None or df_kpis_control.empty:
        return {"estado": "sin_dato", "mensaje": "Aún no hay 4 semanas con dato observado para evaluar el modelo.", "semanas_fuera_umbral": []}
    d = df_kpis_control
    fuera = d.loc[d["estado"] != "normal", "semana_base"].tolist()
    est = d["estado"].iloc[-1]
    if est == "reentrenar recomendado":
        msg = "El error o la detección de alertas están fuera de umbral dos semanas seguidas; se recomienda reentrenar."
    elif est == "vigilar":
        msg = "Algún indicador salió del umbral esta semana; conviene vigilarlo."
    elif fuera:
        msg = (f"Esta semana el modelo está dentro de los umbrales, pero {len(fuera)} de {len(d)} semanas evaluadas "
               f"salieron de algún umbral (la última, {fuera[-1]}).")
    else:
        msg = "El modelo está dentro de los umbrales de control en todas las semanas evaluadas."
    return {"estado": est, "mensaje": msg, "semanas_fuera_umbral": fuera}


# ---------------------------------------------------------------- calibración de umbrales de control
PERCENTIL_RECALL_MIN = 10  # umbral inferior del recall: p10 de las ventanas de validación
PERCENTIL_MAE_FA_MAX = 90  # umbrales superiores de MAE y falsas alarmas: p90


def predicciones_fuera_de_muestra():
    """Reajusta el modelo de focos a 7 días en cada pliego de validación con los parámetros del metadato (margen de 4 semanas).

    Devuelve semana, municipio_norm, uf, real, predicho, pliego solo de las semanas de validación de los 4 pliegos.
    """
    import sys

    from lightgbm import LGBMRegressor  # importación tardía: solo la calibración necesita el modelo

    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    import ia_models.utilidades as U

    meta, par = _cargar()[2], _cargar()[1]
    mm = meta["modelos"]["f1_7d"]
    cols = mm["variables"]
    d = pd.read_csv(RAIZ / "datos" / "datasets_preparation" / "dataset_panel_PREPARADO.csv", parse_dates=["semana"])
    d = d[d["valido_f1"].astype(bool)]
    out = []
    for p in par["pliegos"]:
        tr = d[(d.semana <= p["entreno"][1]) & d.y_7d.notna()]
        va = d[(d.semana >= p["validacion"][0]) & (d.semana <= p["validacion"][1]) & (d.pliego_validacion == p["pliego"])]
        m = U.con_log(LGBMRegressor(**mm["parametros"], subsample_freq=1, n_jobs=-1, random_state=meta["semilla"], verbose=-1))
        m.fit(tr[cols], tr.y_7d)
        o = va[["semana", "municipio_norm", "uf"]].copy()
        o["real"] = va.y_7d.values
        o["predicho"] = np.asarray(m.predict(va[cols]))
        o["pliego"] = p["pliego"]
        out.append(o)
    return pd.concat(out, ignore_index=True)


def calibrar_ventanas_moviles(oos, v_mae=4, v_rf=8):
    """Distribución de MAE móvil, recall y falsas alarmas en ventanas móviles dentro de cada pliego y umbrales por percentil."""
    res = [_ventanas(_conteos_semanales(x), v_mae, v_rf) for _, x in oos.groupby("pliego")]
    v = pd.concat(res)

    def q(s, p):
        return float(np.nanpercentile(s.dropna(), p))

    dist = {c: {f"p{p}": round(q(v[c], p), 4) for p in (5, 10, 50, 90, 95)} for c in ["mae_movil_4s", "recall_alerta", "falsas_alarmas"]}
    return {
        "ventana_mae_semanas": v_mae,
        "ventana_recall_falsas_alarmas_semanas": v_rf,
        "motivo_ventana_8": "con 4 semanas el recall sale ruidoso (rango entre p10 y p90 de 0.245 a 0.750, con pocas alertas reales); con 8 semanas se estabiliza",
        "percentiles_usados": {"mae_movil_max": PERCENTIL_MAE_FA_MAX, "recall_alerta_min": PERCENTIL_RECALL_MIN, "falsas_alarmas_max": PERCENTIL_MAE_FA_MAX},
        "umbrales": {"mae_movil_max": round(q(v["mae_movil_4s"], PERCENTIL_MAE_FA_MAX), 3),
                     "recall_alerta_min": round(q(v["recall_alerta"], PERCENTIL_RECALL_MIN), 3),
                     "falsas_alarmas_max": round(q(v["falsas_alarmas"], PERCENTIL_MAE_FA_MAX), 4)},
        "distribucion_validacion": dist,
        "n_ventanas": {"mae": int(v["mae_movil_4s"].notna().sum()), "recall_y_falsas_alarmas": int(v["recall_alerta"].notna().sum())},
        "fuente": "Predicciones fuera de muestra de los 4 pliegos de validación (modelo final reajustado en cada pliego con los parámetros del metadato y margen de 4 semanas); ventanas dentro de cada pliego",
    }


# ---------------------------------------------------------------- fichas de sustento
def generar_fichas(ruta=RUTA_FICHAS):
    """Construye docs/fichas_kpi.json con cifras leídas de los JSON y del cálculo (nada tecleado)."""
    par, u = _kpi_par(), umbrales_control()
    c = _cortes()
    cap = par["capacidad_de_atencion_municipios_en_alerta"]
    ia = analisis_interanual_historico()
    p75d, p90d = umbrales_deter()
    ant = par["anticipacion_minima"]["valor"]["dias"]
    inpe = par["referencias_de_contexto"]["valor"]["riesgo_fuego_inpe"]
    oms, conama = par["referencias_de_contexto"]["valor"]["pm2_5_oms_ug_m3"], par["referencias_de_contexto"]["valor"]["pm2_5_conama_pi1_ug_m3"]
    ref = par["variacion_interanual_de_focos"]["valor"]["referencia_pct"]
    meta = _cargar()[2]
    pasado = "dato pasado"
    neg = "experiencia del negocio"
    f = []

    def ficha(id_, nombre, que, formula, verde, ambar, rojo, base_pasado, base_negocio, fuente, lim, accion):
        f.append({"id": id_, "nombre": nombre, "que_mide": que, "formula": formula,
                  "umbral": {"verde": verde, "ambar": ambar, "rojo": rojo},
                  "base_del_umbral": {pasado: base_pasado, neg: base_negocio},
                  "fuente_dato": fuente, "limitaciones": lim, "accion_sugerida": accion})

    ficha("nivel_focos", "Nivel de focos por municipio",
          "Qué tan fuerte es la cantidad de focos de calor de un municipio en la semana, comparada con lo normal de la temporada.",
          "Sin focos si hay 0; Habitual si hay hasta p50; Elevado si hay hasta p75; Muy elevado si hay más que p75. El KPI muestra el porcentaje de municipios con focos que están en Muy elevado.",
          f"Hasta {NIVEL_VERDE_MAX_PCT:.0f} % de los municipios con focos en Muy elevado",
          f"De {NIVEL_VERDE_MAX_PCT:.0f} % hasta {NIVEL_AMBAR_MAX_PCT:.0f} %", f"Más de {NIVEL_AMBAR_MAX_PCT:.0f} %",
          f"Cortes por temporada calculados con las semanas con focos de 2022 a 2024. Temporada húmeda: p50 {c['humeda']['p50']:.0f} y p75 {c['humeda']['p75']:.0f} focos. Temporada seca ({par['nivel_focos_municipio_semana']['valor']['seca']}): p50 {c['seca']['p50']:.0f} y p75 {c['seca']['p75']:.0f} focos. Por construcción, cerca de 25 % de los casos con focos supera el p75, por eso ese es el límite verde.",
          f"El límite de {NIVEL_AMBAR_MAX_PCT:.0f} % para el rojo es un supuesto del equipo (sin dato oficial).",
          "BDQueimadas (INPE), panel semanal por municipio.",
          "El conteo depende de que el satélite pase y no haya nubes; un municipio grande tiene más focos que uno pequeño. Los cortes son nacionales, no por municipio.",
          "Ámbar o rojo: revisar la lista de municipios en Muy elevado y confirmar con la brigada local si hay incendios activos.")

    ficha("alertas_vs_capacidad", "Municipios en alerta frente a la capacidad de respuesta",
          "Cuántos municipios el modelo marca en alerta para la próxima semana y si alcanzan los equipos de respuesta.",
          f"Alerta si {u['factor_k']} x focos predichos supera el p75 de la temporada de la semana proyectada (nivel Muy elevado). Se cuentan los municipios en alerta y se comparan con la capacidad.",
          f"Hasta {cap['valor']['base']} municipios en alerta", f"De {cap['valor']['base'] + 1} a {cap['valor']['sensibilidad'][1]} municipios",
          f"Más de {cap['valor']['sensibilidad'][1]} municipios",
          f"La regla de alerta (k={u['factor_k']}, p75 por temporada) sale de la validación con datos de 2022 a 2024.",
          f"Capacidad base de {cap['valor']['base']} municipios (se probó entre {cap['valor']['sensibilidad'][0]} y {cap['valor']['sensibilidad'][1]}). {cap['fuente']}.",
          "Predicciones del modelo de focos a 7 días.",
          "La capacidad es un supuesto declarado; no hay dato oficial de brigadas. Con datos semanales no se sabe el día exacto del incendio.",
          "Ámbar o rojo: priorizar la lista de alertas por orden de prioridad, pedir apoyo de brigadas de otros estados y confirmar la capacidad real con la autoridad.")

    ficha("variacion_interanual", "Variación interanual de focos por estado",
          "Cuánto subieron o bajaron los focos de las últimas 4 semanas frente al mismo periodo del año anterior, en cada estado.",
          f"(focos de las últimas 4 semanas - focos del mismo periodo un año antes) / focos del periodo anterior x 100. Solo se calcula si el periodo anterior tuvo al menos {BASE_MINIMA_INTERANUAL} focos. El KPI muestra el estado con mayor aumento.",
          f"Hasta {INTERANUAL_AMBAR_PCT:.0f} %", f"De {INTERANUAL_AMBAR_PCT:.0f} % hasta {ref} %", f"Más de {ref} %",
          f"Con 2023 y 2024 (ventana de 4 semanas por estado, {ia['n_observaciones']} casos), la referencia de {ref} % cae en el percentil {ia['percentil_de_la_referencia']}: {ia['porcentaje_que_supera_referencia']} % de los casos la supera y {ia['porcentaje_que_supera_ambar']} % supera {INTERANUAL_AMBAR_PCT:.0f} %. Mediana {ia['p50']} %, p75 {ia['p75']} %, p90 {ia['p90']} %. Se mantiene el valor del negocio como alerta temprana, sabiendo que el rojo se activa con frecuencia en meses de subida.",
          f"{par['variacion_interanual_de_focos']['fuente']}. El {INTERANUAL_AMBAR_PCT:.0f} % del ámbar es un supuesto del equipo.",
          "BDQueimadas (INPE), panel semanal agregado por estado.",
          "Con pocos focos el porcentaje se dispara; por eso se exige una base mínima. Un año anterior muy bueno o muy malo cambia la lectura. Solo hay datos desde 2022.",
          "Ámbar o rojo: mirar el estado y el mes, comparar con las lluvias y reforzar la prevención en ese estado.")

    ficha("anticipacion_alerta", "Anticipación de la alerta",
          "Con cuántos días de aviso el modelo anuncia los municipios en alerta.",
          f"Horizonte del modelo ({HORIZONTE_DIAS} días) comparado con la anticipación mínima ({ant} días).",
          f"Horizonte de {ant} días o más", "No se usa", f"Horizonte menor a {ant} días",
          "No se usa dato pasado: el horizonte es una decisión de diseño del modelo.",
          f"Anticipación mínima de {ant} días. {par['anticipacion_minima']['fuente']}.",
          "Diseño del modelo (ia_models/modelos_metadata.json).",
          "Es informativo y un supuesto declarado: con datos semanales no se mide el tiempo real entre la alerta y el incendio.",
          "Rojo: el aviso llega tarde; revisar si el modelo cambió de horizonte antes de usar las alertas.")

    ficha("riesgo_fuego_inpe", "Riesgo de fuego INPE (contexto)",
          "Promedio del índice de riesgo de fuego del INPE por estado en la semana (0 a 1).",
          "Promedio del riesgo_fuego_promedio de los municipios del estado; el KPI muestra el estado con valor más alto.",
          f"Menor a {inpe[1]}", f"De {inpe[1]} a {inpe[2]}", f"Mayor a {inpe[2]}",
          "No se calculó con datos propios.",
          f"Cortes del INPE {inpe} (el {inpe[3]} marca riesgo crítico). El índice se deriva del clima (lluvia, humedad, temperatura), no mide focos reales.",
          "INPE, riesgo de fuego diario, promedio semanal.",
          "Es solo contexto: depende del clima y puede ser alto sin que haya focos. Algunos municipios no tienen dato y se omiten del promedio.",
          "Ámbar o rojo: tomarlo como aviso de condiciones secas y combinarlo con las alertas del modelo.")

    ficha("calidad_aire_pm25", "Calidad del aire, PM2.5 (contexto)",
          "Concentración de partículas finas en el aire, que sube con el humo de los incendios.",
          "Promedio semanal de PM2.5 de los municipios del estado; el KPI muestra el estado con mayor valor.",
          f"Hasta {oms} ug/m3", f"Más de {oms} y hasta {conama} ug/m3", f"Más de {conama} ug/m3",
          "No se calculó con datos propios.",
          f"Guía de la OMS ({oms} ug/m3) y estándar CONAMA PI-1 de Brasil ({conama} ug/m3).",
          "Reanálisis atmosférico (CAMS), promedio semanal por municipio.",
          "Es un modelo de reanálisis, no una estación de medición; puede subestimar picos locales.",
          "Ámbar o rojo: avisar a salud pública y recomendar a la población evitar esfuerzo físico al aire libre.")

    ficha("deforestacion_deter", "Alertas de deforestación DETER (contexto)",
          "Hectáreas con alerta de deforestación en la semana en los estados analizados.",
          "Suma semanal de deter_area_ha; se compara con los percentiles de la serie semanal de 2022 a 2024.",
          f"Hasta {p75d:,.0f} ha (p75)", f"De {p75d:,.0f} a {p90d:,.0f} ha", f"Más de {p90d:,.0f} ha (p90)",
          "Percentiles p75 (ámbar) y p90 (rojo) de la serie semanal de 2022 a 2024, calculados solo con entrenamiento.",
          "Sin umbral del negocio.",
          "DETER (INPE), alertas de deforestación; panel semanal completo.",
          "El DETER del Hito 2 estaba truncado; aquí se usa el panel completo. Las alertas se publican con retraso y pueden corregirse después.",
          "Ámbar o rojo: cruzar con focos en el mismo municipio, porque la deforestación suele anteceder al fuego.")

    k = meta["kpi_control"]
    vm = k["ventanas_moviles"]
    dist, pu = vm["distribucion_validacion"], vm["percentiles_usados"]
    v_mae, v_rf = vm["ventana_mae_semanas"], vm["ventana_recall_falsas_alarmas_semanas"]
    comun = (f"Umbrales calculados como percentiles de las ventanas móviles de las predicciones fuera de muestra de los 4 pliegos de validación "
             f"(2023 y 2024; modelo reajustado en cada pliego con margen de 4 semanas). Ventanas del MAE: {v_mae} semanas; recall y falsas alarmas: {v_rf} semanas, "
             f"porque con {v_mae} semanas hay pocas alertas reales y el recall sale ruidoso.")
    ficha("control_mae_movil", f"MAE móvil de {v_mae} semanas",
          f"Error promedio del modelo (en focos por municipio) en las últimas {v_mae} semanas.",
          f"Promedio de |focos reales - focos predichos| de todos los municipios en las últimas {v_mae} semanas.",
          f"Hasta {u['mae_max']:.3f}", "No se usa", f"Más de {u['mae_max']:.3f} (con dos semanas seguidas se recomienda reentrenar)",
          f"{comun} Umbral = percentil {pu['mae_movil_max']} de la distribución (mediana {dist['mae_movil_4s']['p50']:.2f}, p90 {dist['mae_movil_4s']['p90']:.2f}, {vm['n_ventanas']['mae']} ventanas).",
          "Sin umbral del negocio.", "Predicciones del modelo y focos observados de la semana objetivo.",
          f"Necesita {v_mae} semanas con focos observados. El error sube de forma natural en la temporada seca, por eso el umbral es más holgado que el error de la temporada húmeda.",
          "Si sale en vigilar, revisar los datos de entrada; si se repite dos semanas, pedir el reentrenamiento del modelo.")

    ficha("control_recall_alerta", "Recall de alerta",
          "De los municipios que sí tuvieron una semana Muy elevada, cuántos el modelo avisó.",
          f"Alertas acertadas / municipios con semana Muy elevada, en las últimas {v_rf} semanas. Alerta = {u['factor_k']} x focos predichos > p75 de la temporada ({u['umbral_humeda']:.0f} húmeda, {u['umbral_seca']:.0f} seca).",
          f"{u['recall_min']:.3f} o más", "No se usa", f"Menor a {u['recall_min']:.3f} (con dos semanas seguidas se recomienda reentrenar)",
          f"{comun} Umbral = percentil {pu['recall_alerta_min']} de la distribución (mediana {dist['recall_alerta']['p50']:.2f}, p10 {dist['recall_alerta']['p10']:.3f}, {vm['n_ventanas']['recall_y_falsas_alarmas']} ventanas).",
          "Sin umbral del negocio.", "Predicciones del modelo y focos observados.",
          f"Se calcula juntando {v_rf} semanas porque una sola tiene pocas alertas; se exige un mínimo de {MIN_ALERTAS_REALES_RECALL} alertas reales en la ventana. En 2025 el recall global fue {k['comprobacion_2025']['recall_alerta']:.3f}.",
          "Si sale en vigilar, no confiar solo en el modelo y apoyarse en el nivel de focos observado; dos semanas seguidas, reentrenar.")

    ficha("control_falsas_alarmas", "Tasa de falsas alarmas",
          "De los municipios que no tuvieron semana Muy elevada, cuántos el modelo marcó en alerta por error.",
          f"Alertas falsas / municipios sin semana Muy elevada, en las últimas {v_rf} semanas.",
          f"Hasta {u['fa_max']:.4f}", "No se usa", f"Más de {u['fa_max']:.4f} (se marca vigilar)",
          f"{comun} Umbral = percentil {pu['falsas_alarmas_max']} de la distribución (mediana {dist['falsas_alarmas']['p50']:.4f}, p90 {dist['falsas_alarmas']['p90']:.4f}).",
          "Sin umbral del negocio.", "Predicciones del modelo y focos observados.",
          "Muchas falsas alarmas gastan la capacidad de respuesta; subir el factor k las aumenta y bajarlo reduce el recall.",
          "Si sale en vigilar, revisar si la lista de alertas supera la capacidad y confirmar en terreno antes de movilizar equipos.")

    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(f, ensure_ascii=False, indent=2), encoding="utf-8")
    return f
