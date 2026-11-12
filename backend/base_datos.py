"""Base de datos comun (SQLite) de BIODIVERSITY-GUARD-Predict: esquema, ingesta semanal, prediccion y KPI de control.

Flujo: el modulo de IA envia un CSV por semana -> ingestar_semana (validacion minima) -> predecir_semana (pipeline + modelos)
-> kpis_control (MAE movil de 4 semanas, recall de alerta, falsas alarmas y estado).
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ia_models.pipeline import CLAVE, COLUMNAS_CRUDAS, SEMANAS_HISTORIAL, cargar_modelos, cargar_parametros, predecir, transformar_semana  # noqa: E402

RUTA_DB = Path(__file__).resolve().parent / "biodiversity_guard.db"
RUTA_CRUDO = RAIZ / "datos" / "datasets_eda" / "dataset_panel_semanal_COMPLETO.csv"
DIR_SIMULADO = RAIZ / "datos" / "modulo_ia_simulado"

TEXTO = ["semana", "municipio_norm", "uf", "fuente_sisam"]
ENTERO = ["num_incendios", "deter_num_alertas", "deter_num_alertas_area_protegida", "dias_sisam", "dias_sin_lluvia_nasa"]
COLUMNAS_PANEL = ["semana", "municipio_norm", "uf", "lat", "lon", "num_incendios", "frp_total", "frp_promedio", "precipitacion_promedio",
                  "dias_sin_lluvia_promedio", "riesgo_fuego_promedio", "riesgo_fuego_max", "deter_num_alertas", "deter_area_ha",
                  "deter_num_alertas_area_protegida", "pm10_reanalise", "pm2_5_reanalise", "o3_reanalise", "no2_reanalise", "co_reanalise",
                  "so2_reanalise", "dias_sisam", "fuente_sisam", "T2M", "RH2M", "WS2M", "T2M_MAX", "precip_nasa_sem", "dias_sin_lluvia_nasa",
                  "precip_acum_30d", "precip_acum_120d"]
NO_NULAS = [c for c in COLUMNAS_CRUDAS if c != "riesgo_fuego_promedio"] + ["fuente_sisam"]
RANGOS = {"RH2M": (0, 100), "WS2M": (0, 100), "T2M": (-60, 60), "T2M_MAX": (-60, 70), "lat": (-90, 90), "lon": (-180, 180),
          "riesgo_fuego_promedio": (0, 1), "riesgo_fuego_max": (0, 1)}
NO_NEGATIVAS = ["num_incendios", "frp_total", "precip_nasa_sem", "precip_acum_30d", "precip_acum_120d", "dias_sin_lluvia_nasa", "deter_num_alertas", "deter_area_ha"]


def _tipo(c):
    return "TEXT" if c in TEXTO else ("INTEGER" if c in ENTERO else "REAL")


def esquema():
    cols = ",\n    ".join(f"{c} {_tipo(c)}" for c in COLUMNAS_PANEL)
    return f"""
CREATE TABLE IF NOT EXISTS municipios (municipio_norm TEXT NOT NULL, uf TEXT NOT NULL, lat REAL, lon REAL, PRIMARY KEY (municipio_norm, uf));
CREATE TABLE IF NOT EXISTS observaciones_semanales (
    {cols},
    fecha_ingesta TEXT NOT NULL,
    PRIMARY KEY (semana, municipio_norm, uf));
CREATE TABLE IF NOT EXISTS predicciones (
    semana_base TEXT NOT NULL, municipio_norm TEXT NOT NULL, uf TEXT NOT NULL, focos_7d REAL, focos_30d REAL,
    prob_bajo REAL, prob_moderado REAL, prob_alto REAL, prob_critico REAL, clase_riesgo TEXT, es_alerta INTEGER,
    modelo_version TEXT, fecha_prediccion TEXT, PRIMARY KEY (semana_base, municipio_norm, uf));
CREATE TABLE IF NOT EXISTS kpis_control (semana_base TEXT PRIMARY KEY, mae_movil_4s REAL, recall_alerta REAL, falsas_alarmas REAL, estado TEXT);
CREATE TABLE IF NOT EXISTS modelos (version TEXT NOT NULL, tarea TEXT NOT NULL, algoritmo TEXT, parametros TEXT, metricas TEXT, fecha TEXT, PRIMARY KEY (version, tarea));
CREATE TABLE IF NOT EXISTS ingestas (id INTEGER PRIMARY KEY AUTOINCREMENT, semana TEXT, filas INTEGER, estado TEXT, mensaje TEXT, fecha TEXT);
"""


def conectar(ruta=None):
    conn = sqlite3.connect(str(ruta or RUTA_DB))
    conn.row_factory = None
    return conn


def crear_tablas(conn):
    conn.executescript(esquema())
    conn.commit()


def _ahora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _guardar_observaciones(conn, df):
    df = df[COLUMNAS_PANEL].copy()
    df["semana"] = pd.to_datetime(df["semana"]).dt.strftime("%Y-%m-%d")
    df["fecha_ingesta"] = _ahora()
    columnas = list(df.columns)
    filas = [tuple(None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in fila) for fila in df.itertuples(index=False, name=None)]
    conn.executemany(f"INSERT INTO observaciones_semanales ({', '.join(columnas)}) VALUES ({', '.join('?' * len(columnas))})", filas)


def cargar_municipios_y_historial(conn, hasta="2025-12-29", ruta_crudo=RUTA_CRUDO):
    """Carga municipios y el historial crudo hasta la semana `hasta` (inclusive)."""
    crudo = pd.read_csv(ruta_crudo, parse_dates=["semana"])
    muni = crudo.sort_values("semana").groupby(CLAVE, as_index=False)[["lat", "lon"]].first()
    conn.executemany("INSERT OR REPLACE INTO municipios (municipio_norm, uf, lat, lon) VALUES (?, ?, ?, ?)", muni[["municipio_norm", "uf", "lat", "lon"]].itertuples(index=False, name=None))
    hist = crudo[crudo["semana"] <= pd.Timestamp(hasta)]
    _guardar_observaciones(conn, hist)
    conn.execute("INSERT INTO ingestas (semana, filas, estado, mensaje, fecha) VALUES (?, ?, ?, ?, ?)", (str(hasta), len(hist), "carga_inicial", f"historial 2022 a {hasta}", _ahora()))
    conn.commit()
    return crudo, len(muni), len(hist)


def cargar_tabla_modelos(conn, meta):
    for tarea, info in meta["modelos"].items():
        conn.execute("INSERT OR REPLACE INTO modelos (version, tarea, algoritmo, parametros, metricas, fecha) VALUES (?, ?, ?, ?, ?, ?)",
                     (meta["version_modelos"], tarea, info["algoritmo"] + (f" ({info['tratamiento_desbalance']})" if info.get("tratamiento_desbalance") else ""),
                      json.dumps(info["parametros"], ensure_ascii=False), json.dumps(info["metricas"], ensure_ascii=False), meta["fecha_entrenamiento"]))
    conn.commit()


def ultima_semana(conn):
    r = conn.execute("SELECT MAX(semana) FROM observaciones_semanales").fetchone()[0]
    return pd.Timestamp(r) if r else None


def _registrar(conn, semana, filas, estado, mensaje):
    conn.execute("INSERT INTO ingestas (semana, filas, estado, mensaje, fecha) VALUES (?, ?, ?, ?, ?)", (semana, int(filas), estado, mensaje, _ahora()))
    conn.commit()


def validar_archivo(conn, ruta_csv):
    """Validacion minima. Devuelve (DataFrame o None, semana o None, mensaje de error o None)."""
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:  # archivo ilegible
        return None, None, f"no se pudo leer el archivo: {e}"
    faltan = [c for c in COLUMNAS_PANEL if c not in df.columns]
    if faltan:
        return None, None, f"columnas faltantes: {faltan}"
    df = df[COLUMNAS_PANEL].copy()
    # tipos numericos
    for c in [c for c in COLUMNAS_PANEL if c not in TEXTO]:
        convertido = pd.to_numeric(df[c], errors="coerce")
        if (convertido.isna() & df[c].notna()).any():
            return None, None, f"la columna {c} tiene valores no numericos"
        df[c] = convertido
    nulos = [c for c in NO_NULAS if df[c].isna().any()]
    if nulos:
        return None, None, f"valores nulos en columnas obligatorias: {nulos}"
    try:
        semanas = pd.to_datetime(df["semana"])
    except Exception:
        return None, None, "la columna semana no tiene fechas validas"
    if semanas.nunique() != 1:
        return None, None, "el archivo debe tener una sola semana"
    semana = semanas.iloc[0]
    if semana.dayofweek != 0:
        return None, None, f"la semana {semana.date()} no empieza en lunes"
    df["semana"] = semanas
    # rangos
    if (df["RH2M"] < 0).any() or (df["RH2M"] > 100).any():
        return None, None, f"humedad fuera de rango (0 a 100): maximo {df['RH2M'].max()}"
    for c, (lo, hi) in RANGOS.items():
        if ((df[c] < lo) | (df[c] > hi)).any():
            return None, None, f"{c} fuera de rango ({lo} a {hi})"
    for c in NO_NEGATIVAS:
        if (df[c] < 0).any():
            return None, None, f"{c} tiene valores negativos"
    # claves duplicadas y municipios conocidos
    if df.duplicated(CLAVE).any():
        return None, None, "claves (municipio_norm, uf) duplicadas en el archivo"
    conocidos = pd.read_sql_query("SELECT municipio_norm, uf FROM municipios", conn)
    union = df[CLAVE].merge(conocidos.assign(_ok=1), on=CLAVE, how="left")
    if union["_ok"].isna().any():
        return None, None, f"municipios desconocidos: {union.loc[union['_ok'].isna(), 'municipio_norm'].head(5).tolist()}"
    if len(df) != len(conocidos):
        return None, None, f"faltan municipios: {len(df)} filas de {len(conocidos)}"
    # semana siguiente a la ultima guardada
    ultima = ultima_semana(conn)
    if ultima is not None and semana != ultima + pd.Timedelta(weeks=1):
        return None, None, f"la semana {semana.date()} no es la siguiente a la ultima guardada ({ultima.date()}); se esperaba {(ultima + pd.Timedelta(weeks=1)).date()}"
    return df, semana, None


def ingestar_semana(conn, ruta_csv):
    """Valida y guarda una semana. Si falla no guarda nada y lo registra en `ingestas`. Devuelve (ok, mensaje, filas)."""
    df, semana, error = validar_archivo(conn, ruta_csv)
    if error:
        _registrar(conn, None if semana is None else str(semana.date()), 0, "rechazada", f"{Path(ruta_csv).name}: {error}")
        return False, error, 0
    try:
        with conn:
            _guardar_observaciones(conn, df)
    except sqlite3.Error as e:
        conn.rollback()
        _registrar(conn, str(semana.date()), 0, "rechazada", f"{Path(ruta_csv).name}: error de base de datos: {e}")
        return False, str(e), 0
    _registrar(conn, str(semana.date()), len(df), "aceptada", Path(ruta_csv).name)
    return True, "ok", len(df)


_CACHE = {}


def _modelos():
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["p"] = cargar_modelos(), cargar_parametros()
    return _CACHE["m"], _CACHE["p"]


def leer_historial(conn, semana):
    s = pd.Timestamp(semana)
    ini = (s - pd.Timedelta(weeks=SEMANAS_HISTORIAL)).strftime("%Y-%m-%d")
    df = pd.read_sql_query("SELECT * FROM observaciones_semanales WHERE semana BETWEEN ? AND ?", conn, params=(ini, s.strftime("%Y-%m-%d")), parse_dates=["semana"])
    return df


def predecir_semana(conn, semana):
    """Lee el historial de la base, transforma, predice y guarda en `predicciones`. Despues calcula el KPI de la semana base anterior."""
    modelos, P = _modelos()
    semana = pd.Timestamp(semana)
    X = transformar_semana(leer_historial(conn, semana), semana, parametros=P, meta=modelos["meta"])
    R = predecir(X, modelos)
    R["semana_base"] = R["semana_base"].dt.strftime("%Y-%m-%d")
    R["fecha_prediccion"] = _ahora()
    cols = ["semana_base", "municipio_norm", "uf", "focos_7d", "focos_30d", "prob_bajo", "prob_moderado", "prob_alto", "prob_critico", "clase_riesgo", "es_alerta", "modelo_version", "fecha_prediccion"]
    with conn:
        conn.execute("DELETE FROM predicciones WHERE semana_base = ?", (semana.strftime("%Y-%m-%d"),))
        conn.executemany(f"INSERT INTO predicciones ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [tuple(v.item() if hasattr(v, "item") else v for v in f) for f in R[cols].itertuples(index=False, name=None)])
    calcular_kpis(conn, semana - pd.Timedelta(weeks=1))
    return R


def _error_semanal(conn, semana_base):
    """Compara las predicciones de `semana_base` (focos a 7 dias) con los focos observados de la semana siguiente. None si no existe."""
    s = pd.Timestamp(semana_base)
    sig = (s + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    p = pd.read_sql_query("SELECT municipio_norm, uf, focos_7d, es_alerta FROM predicciones WHERE semana_base = ?", conn, params=(s.strftime("%Y-%m-%d"),))
    o = pd.read_sql_query("SELECT municipio_norm, uf, num_incendios FROM observaciones_semanales WHERE semana = ?", conn, params=(sig,))
    if p.empty or o.empty:
        return None
    d = p.merge(o, on=CLAVE)
    meta_alerta = _modelos()[0]["meta"]["alerta"]
    mes = (s + pd.Timedelta(weeks=1)).month
    umbral = meta_alerta["umbral_seca"] if 7 <= mes <= 11 else meta_alerta["umbral_humeda"]
    real = d["num_incendios"] > umbral
    pred = d["es_alerta"] == 1
    tp, fn, fp, tn = (real & pred).sum(), (real & ~pred).sum(), (~real & pred).sum(), (~real & ~pred).sum()
    return {"mae": float((d["focos_7d"] - d["num_incendios"]).abs().mean()), "recall": float(tp / (tp + fn)) if tp + fn else np.nan,
            "falsas": float(fp / (fp + tn)) if fp + tn else np.nan}


def calcular_kpis(conn, semana_base):
    """KPI de control de `semana_base` (cuando ya existe la semana siguiente en la base) con los umbrales finales de `kpi_control`."""
    s = pd.Timestamp(semana_base)
    actual = _error_semanal(conn, s)
    if actual is None:
        return None
    u = _modelos()[0]["meta"]["kpi_control"]
    maxi_mae, min_rec, max_fal = u["mae_movil_4_semanas_maximo"]["final"], u["recall_alerta_minimo"]["final"], u["tasa_falsas_alarmas_maximo"]["final"]
    previas = [_error_semanal(conn, s - pd.Timedelta(weeks=k)) for k in range(4)]
    mae_movil = float(np.mean([p["mae"] for p in previas])) if all(p is not None for p in previas) else np.nan

    def incumple(e, mae_m):
        return {"mae": bool(mae_m > maxi_mae) if not np.isnan(mae_m) else False, "recall": bool(e["recall"] < min_rec) if not np.isnan(e["recall"]) else False,
                "falsas": bool(e["falsas"] > max_fal) if not np.isnan(e["falsas"]) else False}
    ahora = incumple(actual, mae_movil)
    # semana anterior (para la regla de dos semanas seguidas)
    ant = conn.execute("SELECT mae_movil_4s, recall_alerta, falsas_alarmas FROM kpis_control WHERE semana_base = ?", ((s - pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"),)).fetchone()
    previo = incumple({"recall": np.nan if ant is None or ant[1] is None else ant[1], "falsas": np.nan if ant is None or ant[2] is None else ant[2]},
                      np.nan if ant is None or ant[0] is None else ant[0])
    if (ahora["mae"] and previo["mae"]) or (ahora["recall"] and previo["recall"]):
        estado = "reentrenar recomendado"
    elif any(ahora.values()):
        estado = "vigilar"
    else:
        estado = "normal"
    conn.execute("INSERT OR REPLACE INTO kpis_control (semana_base, mae_movil_4s, recall_alerta, falsas_alarmas, estado) VALUES (?, ?, ?, ?, ?)",
                 (s.strftime("%Y-%m-%d"), None if np.isnan(mae_movil) else mae_movil, None if np.isnan(actual["recall"]) else actual["recall"],
                  None if np.isnan(actual["falsas"]) else actual["falsas"], estado))
    conn.commit()
    return estado
