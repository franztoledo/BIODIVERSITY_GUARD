"""Servicio de ejemplo (stub) con el mismo contrato que backend/servicio.py.

Deriva todo de archivos existentes del repositorio. Las cifras son aproximadas:
sirve para que la interfaz funcione antes de tener el servicio real.
Estado de ingesta en memoria (se pierde al reiniciar).
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
F_PRED = RAIZ / "resultados" / "avanzado_predicciones_2025.csv"
F_PANEL = RAIZ / "datos" / "datasets_eda" / "dataset_panel_semanal_COMPLETO.csv"
F_PARAM = RAIZ / "datos" / "datasets_preparation" / "kpi_parametros.json"
F_META = RAIZ / "ia_models" / "modelos_metadata.json"
F_FICHAS = RAIZ / "docs" / "fichas_kpi.json"

CLASES = ["Bajo", "Moderado", "Alto", "Crítico"]
SIN_DATO = "Sin dato"

_PARAM = json.loads(F_PARAM.read_text(encoding="utf-8"))
_META = json.loads(F_META.read_text(encoding="utf-8"))
_K = float(_META["alerta"]["factor_k"])
_CORTES = _PARAM["nivel_focos_municipio_semana"]["valor"]["cortes_por_temporada"]
_CAP = _PARAM["capacidad_de_atencion_municipios_en_alerta"]["valor"]
_KC = _META["kpi_control"]
_FILAS_HIST: list[dict] = []
_EXTRA: dict[str, pd.DataFrame] = {}


# ----------------------------------------------------------------- carga
def _cargar_base() -> pd.DataFrame:
    p = pd.read_csv(F_PRED)
    f7 = p[(p.frente == "f1_7d") & (p.modelo == "lgbm_final_sin_sisam")]
    f30 = p[(p.frente == "f1_30d") & (p.modelo == "xgb_final_con_sisam")]
    f2 = p[(p.frente == "f2") & (p.modelo == "rf_final_sin_sisam")]
    k = ["semana", "municipio_norm", "uf"]
    b = f7[k + ["real", "predicho"]].rename(columns={"real": "focos_real_7d", "predicho": "focos_7d"})
    b = b.merge(f30[k + ["predicho"]].rename(columns={"predicho": "focos_30d"}), on=k, how="left")
    b = b.merge(f2[k + ["real", "predicho"]].rename(columns={"real": "clase_real", "predicho": "clase_idx"}),
                on=k, how="left")
    pan = pd.read_csv(F_PANEL, usecols=["municipio_norm", "uf", "lat", "lon"]).drop_duplicates(
        ["municipio_norm", "uf"])
    b = b.merge(pan, on=["municipio_norm", "uf"], how="left")
    b["focos_30d"] = b["focos_30d"].fillna(b["focos_7d"] * 4.3)
    return b


_BASE = _cargar_base()
_PANEL = pd.read_csv(F_PANEL, usecols=["semana", "uf", "num_incendios", "riesgo_fuego_promedio",
                                       "pm2_5_reanalise", "deter_area_ha"])


def _todo() -> pd.DataFrame:
    if not _EXTRA:
        return _BASE
    return pd.concat([_BASE] + list(_EXTRA.values()), ignore_index=True)


def _cortes(semana: str) -> dict:
    mes = int(semana[5:7])
    return _CORTES["seca" if 7 <= mes <= 11 else "humeda"]


def _nivel(focos_aj: pd.Series, semana: str) -> pd.Series:
    c = _cortes(semana)
    return pd.Series(np.select([focos_aj < 0.5, focos_aj <= c["p50"], focos_aj <= c["p75"]],
                               ["Sin focos", "Habitual", "Elevado"], "Muy elevado"), index=focos_aj.index)


def _probs(idx: pd.Series) -> pd.DataFrame:
    """Probabilidades aproximadas a partir de la clase predicha (el stub no tiene las reales)."""
    out = np.full((len(idx), 4), np.nan)
    for i, c in enumerate(idx.to_numpy()):
        if np.isnan(c):
            continue
        c = int(c)
        w = np.array([0.7 if j == c else (0.2 if abs(j - c) == 1 else 0.03) for j in range(4)])
        out[i] = w / w.sum()
    return pd.DataFrame(out, columns=["prob_bajo", "prob_moderado", "prob_alto", "prob_critico"], index=idx.index)


# ------------------------------------------------------------- semanas
def semanas_disponibles() -> list[str]:
    return sorted(_todo()["semana"].unique())


def ultima_semana() -> str:
    return semanas_disponibles()[-1]


def _sem(semana: str | None) -> str:
    return semana or ultima_semana()


# --------------------------------------------------------- predicciones
def predicciones(semana, uf=None) -> pd.DataFrame:
    d = _todo()
    d = d[d.semana == _sem(semana)].copy()
    if uf:
        d = d[d.uf == uf]
    d["clase_riesgo"] = d["clase_idx"].map(dict(enumerate(CLASES))).fillna(SIN_DATO)
    d = pd.concat([d, _probs(d["clase_idx"])], axis=1)
    d["es_alerta"] = _K * d["focos_7d"] > _cortes(_sem(semana))["p75"]
    d["focos_7d"] = d["focos_7d"].round(2)
    d["focos_30d"] = d["focos_30d"].round(2)
    cols = ["municipio_norm", "uf", "lat", "lon", "focos_7d", "focos_30d", "clase_riesgo", "prob_bajo",
            "prob_moderado", "prob_alto", "prob_critico", "es_alerta", "focos_real_7d"]
    return d[cols].reset_index(drop=True)


def tabla_alertas(semana) -> pd.DataFrame:
    s = _sem(semana)
    d = predicciones(s)
    d["nivel_focos"] = _nivel(_K * d["focos_7d"], s)
    d = d.sort_values(["es_alerta", "focos_7d"], ascending=False).reset_index(drop=True)
    d["prioridad"] = np.arange(1, len(d) + 1)
    d["prob_critico"] = d["prob_critico"].round(3)
    return d[["municipio_norm", "uf", "nivel_focos", "focos_7d", "focos_30d", "clase_riesgo",
              "prob_critico", "es_alerta", "prioridad"]]


def serie_real_vs_predicho(uf=None) -> pd.DataFrame:
    d = _todo()
    if uf:
        d = d[d.uf == uf]
    g = d.groupby("semana").agg(real_7d=("focos_real_7d", "sum"), pred_7d=("focos_7d", "sum"),
                                n_real=("focos_real_7d", "count")).reset_index()
    g.loc[g.n_real == 0, "real_7d"] = np.nan
    return g[["semana", "real_7d", "pred_7d"]]


# ------------------------------------------------------------- contexto
def contexto(semana) -> pd.DataFrame:
    s = _sem(semana)
    pan = _PANEL[_PANEL.semana <= s]
    if pan.empty:
        return pd.DataFrame(columns=["uf", "riesgo_fuego_promedio", "pm2_5", "deter_area_ha"])
    ref = pan.semana.max()  # semana más cercana con dato
    d = pan[pan.semana == ref].groupby("uf").agg(
        riesgo_fuego_promedio=("riesgo_fuego_promedio", "mean"), pm2_5=("pm2_5_reanalise", "mean"),
        deter_area_ha=("deter_area_ha", "sum")).reset_index()
    return d


def umbrales_contexto() -> dict:
    """EXTRA al contrato: umbrales para la página de Contexto."""
    ref = _PARAM["referencias_de_contexto"]["valor"]
    dm = _PANEL[(_PANEL.semana < "2025-01-01") & (_PANEL.deter_area_ha > 0)]
    g = dm.groupby(["semana", "uf"]).deter_area_ha.sum()
    return {
        "riesgo_fuego": {"cortes": ref["riesgo_fuego_inpe"], "clases": ["Mínimo", "Bajo", "Medio", "Alto", "Crítico"],
                         "fuente": _PARAM["referencias_de_contexto"]["fuente"]},
        "pm2_5": {"ambar": ref["pm2_5_oms_ug_m3"], "rojo": ref["pm2_5_conama_pi1_ug_m3"],
                  "unidad": "µg/m³", "fuente": "Guía OMS y estándar nacional PI-1"},
        "deter": {"ambar": float(g.quantile(0.5)), "rojo": float(g.quantile(0.75)), "unidad": "ha por estado y semana",
                  "fuente": "Mediana y P75 de semanas con alertas DETER 2022 a 2024 (stub)"},
    }


# ------------------------------------------------------------------ KPI
def _tasas() -> pd.DataFrame:
    """Métricas semanales con datos reales (semanas con focos_real_7d)."""
    d = _todo().dropna(subset=["focos_real_7d"]).copy()
    filas = []
    for s, g in d.groupby("semana"):
        p75 = _cortes(s)["p75"]
        ra, pa = g.focos_real_7d > p75, _K * g.focos_7d > p75
        crit = g.clase_real == 3
        filas.append(dict(semana=s, mae=(g.focos_real_7d - g.focos_7d).abs().mean(),
                          tp=int((ra & pa).sum()), fn=int((ra & ~pa).sum()), fp=int((~ra & pa).sum()),
                          tn=int((~ra & ~pa).sum()), ctp=int((crit & (g.clase_idx == 3)).sum()),
                          cfn=int((crit & (g.clase_idx != 3)).sum())))
    return pd.DataFrame(filas)


def _umbrales() -> dict:
    ub = {"mae": _KC["mae_movil_4_semanas_maximo"], "rec": _KC["recall_alerta_minimo"],
          "fa": _KC["tasa_falsas_alarmas_maximo"]}
    return {"mae_movil_max": ub["mae"]["final"], "recall_alerta_min": ub["rec"]["final"],
            "falsas_alarmas_max": ub["fa"]["final"],
            "mae_ambar": ub["mae"]["media_validacion"], "recall_ambar": ub["rec"]["media_validacion"],
            "fa_ambar": ub["fa"]["media_validacion"]}


def kpis_control() -> pd.DataFrame:
    t = _tasas()
    if len(t) < 4:
        return pd.DataFrame(columns=["semana_base", "mae_movil_4s", "recall_alerta", "falsas_alarmas", "estado"])
    u = _umbrales()
    r = t.rolling(4)
    out = pd.DataFrame({"semana_base": t.semana, "mae_movil_4s": r.mae.mean()})
    tp, fn, fp, tn = (r[c].sum() for c in ("tp", "fn", "fp", "tn"))
    out["recall_alerta"] = tp / (tp + fn).replace(0, np.nan)
    out["falsas_alarmas"] = fp / (fp + tn).replace(0, np.nan)
    out = out.dropna(subset=["mae_movil_4s"]).reset_index(drop=True)
    fuera = ((out.mae_movil_4s > u["mae_movil_max"]) | (out.recall_alerta < u["recall_alerta_min"])
             | (out.falsas_alarmas > u["falsas_alarmas_max"]))
    out["estado"] = np.where(fuera, "fuera", "dentro")
    return out


def estado_modelo() -> dict:
    k = kpis_control()
    u = _umbrales()
    umb = {"mae_movil_max": u["mae_movil_max"], "recall_alerta_min": u["recall_alerta_min"],
           "falsas_alarmas_max": u["falsas_alarmas_max"]}
    if k.empty:
        return dict(estado="sin_dato", mensaje="Aún no hay semanas con dato real para evaluar el modelo.",
                    semanas_fuera_umbral=0, umbrales=umb)
    n = int((k.tail(4).estado == "fuera").sum())
    if n >= 2:
        est, msg = "reentrenar", "Reentrenar recomendado: 2 o más de las últimas 4 semanas fuera de umbral."
    elif n == 1:
        est, msg = "vigilancia", "En vigilancia: 1 de las últimas 4 semanas fuera de umbral."
    else:
        est, msg = "estable", "Modelo estable: las últimas 4 semanas dentro de los umbrales."
    return dict(estado=est, mensaje=msg, semanas_fuera_umbral=n, umbrales=umb)


def _kpi(id_, nombre, valor, unidad, amb, roj, base, lectura_fn):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return dict(id=id_, nombre=nombre, valor=None, unidad=unidad, estado="sin_dato", umbral_ambar=amb,
                    umbral_rojo=roj, base_umbral=base, lectura="Sin dato real para esta semana (proyección sin observado).")
    if roj < amb:
        est = "rojo" if valor < roj else "ambar" if valor < amb else "verde"
    else:
        est = "rojo" if valor > roj else "ambar" if valor > amb else "verde"
    return dict(id=id_, nombre=nombre, valor=float(valor), unidad=unidad, estado=est, umbral_ambar=amb,
                umbral_rojo=roj, base_umbral=base, lectura=lectura_fn(est))


def resumen_kpis(semana) -> list[dict]:
    s = _sem(semana)
    u = _umbrales()
    k = kpis_control()
    fila = k[k.semana_base == s]
    f = fila.iloc[0] if len(fila) else None
    n_al = int(predicciones(s).es_alerta.sum())
    t = _tasas()
    t = t[t.semana <= s].tail(8)
    crit = None
    if len(t) and (t.ctp + t.cfn).sum() > 0:
        crit = 100 * t.ctp.sum() / (t.ctp.sum() + t.cfn.sum())
    vm = _META["modelos"]["f2"]["metricas"]["validacion"]["recall_critico"]
    cr_roj, cr_amb = 100 * (vm["media"] - vm["desv"]), 100 * vm["media"]
    L = {"verde": "dentro de lo esperado", "ambar": "cerca del límite, vigilar", "rojo": "fuera del umbral, actuar"}
    return [
        _kpi("control_mae_movil", "Error medio móvil (4 semanas)", None if f is None else f.mae_movil_4s,
             "focos por municipio", u["mae_movil_max"], u["mae_movil_max"], _KC["regla"],
             lambda e: f"Error de la proyección de focos a 7 días: {L[e]}."),
        _kpi("control_recall_alerta", "Alertas detectadas (recall)", None if f is None or np.isnan(f.recall_alerta) else 100 * f.recall_alerta,
             "%", 100 * u["recall_ambar"], 100 * u["recall_alerta_min"], _KC["regla"],
             lambda e: f"Parte de los municipios críticos que el modelo alertó: {L[e]}."),
        _kpi("control_falsas_alarmas", "Falsas alarmas", None if f is None or np.isnan(f.falsas_alarmas) else 100 * f.falsas_alarmas,
             "%", 100 * u["fa_ambar"], 100 * u["falsas_alarmas_max"], _KC["regla"],
             lambda e: f"Alertas emitidas sin foco crítico real: {L[e]}."),
        _kpi("alertas_vs_capacidad", "Municipios en alerta", n_al, "municipios", _CAP["base"], _CAP["sensibilidad"][1],
             _PARAM["capacidad_de_atencion_municipios_en_alerta"]["fuente"],
             lambda e: {"verde": "Dentro de la capacidad base de brigadas.",
                        "ambar": "Supera la capacidad base de brigadas: priorizar.",
                        "rojo": "Supera incluso el escenario alto de capacidad."}[e]),
        _kpi("recall_critico", "Críticos detectados (Frente 2)", crit, "%", cr_amb, cr_roj,
             "Validación cruzada del Frente 2: media y media menos desviación",
             lambda e: f"Clase Crítico bien detectada en las últimas 8 semanas: {L[e]}."),
    ]


# --------------------------------------------------------------- fichas
def fichas() -> dict:
    propias = _fichas_propias()
    if F_FICHAS.exists():
        d = json.loads(F_FICHAS.read_text(encoding="utf-8"))
        if isinstance(d, list):
            d = {x["id"]: x for x in d}
        propias.update(d)
    return propias


def _fichas_propias() -> dict:
    u = _umbrales()
    cuando = "Calculado con validación cruzada temporal 2022 a 2024 y comprobado con 2025."
    def f(que, formula, dato, negocio, lim, acc):
        return dict(que_mide=que, formula=formula, base_umbral={"dato_pasado": dato, "negocio": negocio},
                    limitaciones=lim, accion_sugerida=acc)
    return {
        "control_mae_movil": f("Error absoluto medio de los focos a 7 días por municipio, promediado en 4 semanas.",
                          "media(|focos reales - focos predichos|) sobre 4 semanas",
                          f"Rojo = media + desviación entre folds ({u['mae_movil_max']:.2f}); ámbar = media de validación ({u['mae_ambar']:.2f}). {cuando}",
                          "Un error alto resta confianza al orden de prioridad de los municipios.",
                          "Es un valor aproximado en el stub; los datos 2026 son simulados.",
                          "Si queda en rojo 2 semanas, reentrenar con datos recientes."),
        "control_recall_alerta": f("Parte de las alertas reales (nivel Muy elevado) que el modelo detectó.",
                           "verdaderos positivos / (verdaderos positivos + falsos negativos), ventana de 4 semanas",
                           f"Rojo = media - desviación ({100*u['recall_alerta_min']:.1f} %); ámbar = media ({100*u['recall_ambar']:.1f} %). {cuando}",
                           "Cada alerta no detectada es un incendio grave sin brigada preasignada.",
                           "Con pocas alertas reales por semana el valor es inestable.",
                           "Bajar el factor k o revisar el modelo si cae bajo el umbral."),
        "control_falsas_alarmas": f("Parte de las semanas sin alerta real en las que igual se emitió alerta.",
                            "falsos positivos / (falsos positivos + verdaderos negativos), ventana de 4 semanas",
                            f"Rojo = media + desviación ({100*u['falsas_alarmas_max']:.1f} %); ámbar = media ({100*u['fa_ambar']:.1f} %). {cuando}",
                            "Demasiadas falsas alarmas desgastan a las brigadas.",
                            "Depende del factor k elegido.",
                            "Subir el factor k o reducir el número de alertas atendidas."),
        "alertas_vs_capacidad": f("Cantidad de municipios en alerta en la semana.",
                               "conteo de municipios con factor k x focos predichos sobre el P75 de la temporada",
                               "Capacidad base de atención y rango de sensibilidad del proyecto.",
                               _PARAM["capacidad_de_atencion_municipios_en_alerta"]["fuente"],
                               "La capacidad de brigadas es un supuesto, no un dato oficial.",
                               "Si supera la capacidad, atender por prioridad y pedir refuerzo."),
        "recall_critico": f("Parte de los casos de clase Crítico que el Frente 2 acertó.",
                            "verdaderos críticos / críticos reales, últimas 8 semanas",
                            f"Rojo = media - desviación en validación; ámbar = media. {cuando}",
                            "Un crítico no detectado es el error más costoso.",
                            "Solo hay clase real en parte de las semanas.",
                            "Revisar el Frente 2 si queda en rojo."),
    }


# --------------------------------------------------------------- ingesta
REQ = ["semana", "municipio_norm", "uf", "num_incendios"]


def _leer(archivo_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archivo_bytes))


def validar_semana(archivo_bytes, nombre) -> dict:
    m: list[str] = []
    try:
        d = _leer(archivo_bytes)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, mensajes=[f"ERROR: no se pudo leer el CSV ({e})."])
    falt = [c for c in REQ if c not in d.columns]
    if falt:
        return dict(ok=False, mensajes=[f"ERROR: faltan columnas obligatorias: {', '.join(falt)}."])
    m.append(f"OK: columnas obligatorias presentes ({', '.join(REQ)}).")
    ok = True
    sem = d.semana.astype(str).unique()
    if len(sem) != 1:
        ok = False
        m.append(f"ERROR: el archivo debe tener una sola semana y trae {len(sem)}.")
    else:
        try:
            dt = datetime.strptime(sem[0], "%Y-%m-%d").date()
            if dt.weekday() != 0:
                ok = False
                m.append(f"ERROR: la semana {sem[0]} no es lunes.")
            elif sem[0] <= ultima_semana():
                ok = False
                m.append(f"ERROR: la semana {sem[0]} no es posterior a la última cargada ({ultima_semana()}).")
            else:
                m.append(f"OK: semana {sem[0]} es lunes y posterior a la última cargada.")
        except ValueError:
            ok = False
            m.append("ERROR: la columna semana debe tener formato AAAA-MM-DD.")
    if pd.to_numeric(d.num_incendios, errors="coerce").isna().any():
        ok = False
        m.append("ERROR: num_incendios tiene valores no numéricos o vacíos.")
    elif (pd.to_numeric(d.num_incendios) < 0).any():
        ok = False
        m.append("ERROR: num_incendios tiene valores negativos.")
    else:
        m.append("OK: num_incendios numérico y no negativo.")
    if d.duplicated(["municipio_norm", "uf"]).any():
        ok = False
        m.append("ERROR: hay municipios duplicados en la semana.")
    desc = set(zip(d.municipio_norm, d.uf)) - set(zip(_BASE.municipio_norm, _BASE.uf))
    if desc:
        m.append(f"AVISO: {len(desc)} municipios no conocidos se ignoran.")
    pct = len(set(zip(d.municipio_norm, d.uf)) & set(zip(_BASE.municipio_norm, _BASE.uf))) / _BASE[["municipio_norm", "uf"]].drop_duplicates().shape[0]
    if pct < 0.5:
        m.append(f"AVISO: el archivo cubre solo {pct:.0%} de los municipios; el resto usa la última proyección.")
    return dict(ok=ok, mensajes=m)


def _registrar(semana: str, nuevos: pd.DataFrame | None, modo: str) -> dict:
    ult = _todo()
    prev = ult[ult.semana == ult.semana.max()].copy()
    rng = np.random.default_rng(int(semana.replace("-", "")))
    if nuevos is not None:
        obs = nuevos.set_index(["municipio_norm", "uf"]).num_incendios
        o = prev.set_index(["municipio_norm", "uf"]).index.map(lambda k: obs.get(k, np.nan))
        base = pd.Series(np.asarray(o, dtype=float), index=prev.index)
        prev["focos_7d"] = np.where(base.notna(), 0.6 * base + 0.4 * prev["focos_7d"], prev["focos_7d"])
    else:
        prev["focos_7d"] = prev["focos_7d"] * rng.lognormal(0, 0.35, len(prev))
    prev["focos_30d"] = prev["focos_7d"] * 4.3
    prev["semana"] = semana
    prev["focos_real_7d"] = np.nan
    prev["clase_real"] = np.nan
    _EXTRA[semana] = prev
    n = int(predicciones(semana).es_alerta.sum())
    _FILAS_HIST.append(dict(momento=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), semana=semana,
                            modo=modo, n_alertas=n))
    return dict(ok=True, mensaje=f"Semana {semana} cargada y proyectada: {n} municipios en alerta.",
                semana=semana, n_alertas=n)


def ingestar_y_predecir(archivo_bytes, nombre) -> dict:
    v = validar_semana(archivo_bytes, nombre)
    if not v["ok"]:
        return dict(ok=False, mensaje="El archivo no pasó la validación.", semana=None, n_alertas=0)
    d = _leer(archivo_bytes)
    return _registrar(str(d.semana.iloc[0]), d, f"archivo: {nombre}")


def simular_siguiente_semana() -> dict:
    s = (date.fromisoformat(ultima_semana()) + timedelta(days=7)).isoformat()
    return _registrar(s, None, "simulación")


def historial_ingestas() -> pd.DataFrame:
    """EXTRA al contrato: historial de ingestas de esta ejecución."""
    return pd.DataFrame(_FILAS_HIST, columns=["momento", "semana", "modo", "n_alertas"])
