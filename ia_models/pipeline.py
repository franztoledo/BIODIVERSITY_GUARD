"""Pipeline de inferencia de BIODIVERSITY-GUARD-Predict.

Convierte datos crudos (panel municipio-semana sin transformar) en las variables de los modelos y predice.
Replica, en el mismo orden, los pasos de `codigo/Data Preparation/DATAPREPARATION_BIODIVERSITY_GUARD.ipynb`:
rezagos y media movil -> imputacion de riesgo_fuego_t -> recorte de atipicos -> log1p -> escalado -> variables de estado
-> epoca del anio, temporada_seca y sin_riesgo_t. Solo usa los parametros guardados (no reajusta nada), solo mira semanas
anteriores o iguales a la semana objetivo y no usa targets.

Uso:
    from ia_models.pipeline import cargar_modelos, transformar_semana, predecir
    modelos = cargar_modelos()
    X = transformar_semana(historial_crudo, semana)      # una fila por municipio
    resultado = predecir(X, modelos)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import joblib  # noqa: E402

import ia_models.utilidades  # noqa: E402,F401  (debe importarse antes de joblib.load)

CLAVE = ["municipio_norm", "uf"]
CLASES = ["Bajo", "Moderado", "Alto", "Crítico"]
REZAGOS = [1, 2, 3, 4]
VARS_REZAGO_NASA = ["precip_acum_30d", "dias_sin_lluvia_nasa", "RH2M", "T2M_MAX", "T2M", "precip_nasa_sem"]
SEMANAS_HISTORIAL = 4          # semanas anteriores necesarias (rezagos hasta 4 y media movil de 4)
COLUMNAS_CRUDAS = ["semana", "municipio_norm", "uf", "lat", "lon", "num_incendios", "frp_total", "riesgo_fuego_promedio",
                   "pm10_reanalise", "pm2_5_reanalise", "o3_reanalise", "no2_reanalise", "co_reanalise", "so2_reanalise",
                   "T2M", "RH2M", "WS2M", "T2M_MAX", "precip_nasa_sem", "dias_sin_lluvia_nasa", "precip_acum_30d", "precip_acum_120d"]
DIR_PREPARATION = RAIZ / "datos" / "datasets_preparation"
DIR_MODELOS = RAIZ / "ia_models"


def cargar_parametros(directorio=DIR_PREPARATION):
    return json.load(open(Path(directorio) / "parametros_preparacion.json", encoding="utf-8"))


def cargar_metadata(directorio=DIR_MODELOS):
    return json.load(open(Path(directorio) / "modelos_metadata.json", encoding="utf-8"))


def cargar_modelos(directorio=DIR_MODELOS):
    """Carga los tres modelos y el metadato (variables, factor k y umbrales de la alerta)."""
    meta = cargar_metadata(directorio)
    modelos = {t: joblib.load(Path(directorio) / info["archivo"]) for t, info in meta["modelos"].items()}
    return {"modelos": modelos, "meta": meta}


def columnas_de_modelos(meta):
    """Union ordenada de las variables que piden los tres modelos."""
    columnas = []
    for info in meta["modelos"].values():
        columnas += [c for c in info["variables"] if c not in columnas]
    return columnas


def redondear_como_referencia(tabla, columnas):
    """float32 y 6 cifras significativas: la representacion de dataset_panel_PREPARADO.csv, con la que se entrenaron y evaluaron los modelos."""
    for c in columnas:
        if tabla[c].dtype.kind == "f":
            tabla[c] = [float("%.6g" % x) for x in tabla[c].to_numpy(np.float32).astype(float)]
    return tabla


def transformar_semana(historial_crudo, semana, parametros=None, meta=None, redondear=True):
    """Variables de los modelos para todos los municipios de `semana`.

    historial_crudo: DataFrame crudo con la semana objetivo y, al menos, las 4 semanas anteriores (se ignora todo lo posterior).
    Devuelve semana, municipio_norm, uf y las columnas de los modelos (ordenadas) sin nulos en las variables usadas.
    redondear=True (por defecto) deja las variables en la misma representacion de dataset_panel_PREPARADO.csv (float32 con 6 cifras
    significativas), de modo que las predicciones sean identicas a las del notebook; con False se conserva la precision completa.
    """
    P = parametros or cargar_parametros()
    meta = meta or cargar_metadata()
    semana = pd.Timestamp(semana)
    faltan = [c for c in COLUMNAS_CRUDAS if c not in historial_crudo.columns]
    if faltan:
        raise ValueError(f"Faltan columnas crudas: {faltan}")
    h = historial_crudo[COLUMNAS_CRUDAS].copy()
    h["semana"] = pd.to_datetime(h["semana"])
    h = h[(h["semana"] <= semana) & (h["semana"] >= semana - pd.Timedelta(weeks=SEMANAS_HISTORIAL))]
    esperadas = {semana - pd.Timedelta(weeks=k) for k in range(SEMANAS_HISTORIAL + 1)}
    if set(h["semana"].unique()) != esperadas:
        raise ValueError(f"El historial no tiene las {SEMANAS_HISTORIAL + 1} semanas consecutivas hasta {semana.date()}")
    h = h.sort_values(CLAVE + ["semana"]).reset_index(drop=True)
    assert not h.duplicated(CLAVE + ["semana"]).any(), "Claves duplicadas en el historial"

    # 1. rezagos y media movil (sobre datos crudos); riesgo, estado, epoca del anio
    g = h.groupby(CLAVE)
    nuevas = {}
    focos = g["num_incendios"]
    for k in REZAGOS:
        nuevas[f"num_incendios_rez{k}"] = focos.shift(k)
    nuevas["num_incendios_mm4"] = focos.transform(lambda s: s.rolling(4).mean())
    for variable in VARS_REZAGO_NASA:
        for k in REZAGOS:
            nuevas[f"{variable}_rez{k}"] = g[variable].shift(k)
    nuevas["riesgo_fuego_t"] = h["riesgo_fuego_promedio"]
    nuevas["sin_riesgo_t"] = h["riesgo_fuego_promedio"].isna().astype("int8")
    mes = h["semana"].dt.month
    sem_anio = (h["semana"].dt.day_of_year - 1) // 7 + 1
    nuevas["temporada_seca"] = mes.between(7, 11).astype("int8")
    nuevas["sem_anio_sin"] = np.sin(2 * np.pi * sem_anio / 52)
    nuevas["sem_anio_cos"] = np.cos(2 * np.pi * sem_anio / 52)
    d = pd.concat([h, pd.DataFrame(nuevas, index=h.index)], axis=1)
    d = d[d["semana"] == semana].reset_index(drop=True)

    # 2. imputacion de riesgo_fuego_t con la mediana del entrenamiento
    d["riesgo_fuego_t"] = d["riesgo_fuego_t"].fillna(P["imputacion"]["riesgo_fuego_t_mediana_entrenamiento"])

    # 3. recorte de atipicos (variable base y sus rezagos), solo las que el Data Preparation decidio recortar
    for variable, datos in P["atipicos"].items():
        if not datos["recortar"]:
            continue
        for columna in [variable] + [f"{variable}_rez{k}" for k in REZAGOS if f"{variable}_rez{k}" in d.columns]:
            d[columna] = d[columna].clip(datos["inferior"], datos["superior"])

    # 4. log1p de las variables asimetricas
    for c in P["log1p"]["variables"]:
        d[c] = np.log1p(d[c])

    # 5. escalado con la media y la desviacion del entrenamiento
    for c in P["escalador"]["columnas"]:
        d[c] = (d[c] - P["escalador"]["media"][c]) / P["escalador"]["desviacion"][c]

    # 6. variables de estado (one-hot; las columnas salen de las listas de variables de los modelos)
    for c in [c for c in columnas_de_modelos(meta) if c.startswith("uf_")]:
        d[c] = (d["uf"] == c[3:]).astype("int8")

    columnas = columnas_de_modelos(meta)
    salida = d[["semana"] + CLAVE + columnas].copy()
    nulos = salida[columnas].isna().sum()
    if nulos.any():
        raise ValueError(f"Variables con nulos para {semana.date()}: {nulos[nulos > 0].to_dict()}")
    return redondear_como_referencia(salida, columnas) if redondear else salida


def predecir(features, modelos):
    """Predicciones por municipio: focos a 7 y 30 dias, probabilidades y clase de riesgo, y alerta calibrada."""
    meta, M = modelos["meta"], modelos["modelos"]
    v = {t: meta["modelos"][t]["variables"] for t in M}
    salida = features[["semana"] + CLAVE].rename(columns={"semana": "semana_base"}).reset_index(drop=True)
    salida["focos_7d"] = np.asarray(M["f1_7d"].predict(features[v["f1_7d"]]), dtype=float)
    salida["focos_30d"] = np.asarray(M["f1_30d"].predict(features[v["f1_30d"]]), dtype=float)
    proba = M["f2"].predict_proba(features[v["f2"]])
    clases = list(M["f2"].classes_)
    for i, nombre in enumerate(CLASES):
        salida[f"prob_{nombre.lower().replace('í', 'i')}"] = proba[:, clases.index(i)]
    salida["clase_riesgo"] = [CLASES[int(c)] for c in M["f2"].predict(features[v["f2"]])]
    a = meta["alerta"]
    mes_objetivo = (pd.to_datetime(salida["semana_base"]) + pd.Timedelta(weeks=1)).dt.month
    umbral = np.where(mes_objetivo.between(7, 11), a["umbral_seca"], a["umbral_humeda"])
    salida["es_alerta"] = (a["factor_k"] * salida["focos_7d"] > umbral).astype(int)
    salida["modelo_version"] = meta["version_modelos"]
    return salida
