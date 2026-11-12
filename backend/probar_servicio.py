"""Prueba de backend/servicio.py con una copia temporal de la base (no toca backend/biodiversity_guard.db).

Uso: python backend/probar_servicio.py        (crea la base de demostracion en un directorio temporal, unos 30 s)
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

TMP = Path(tempfile.mkdtemp(prefix="bg_servicio_"))
os.environ["BG_DB"] = str(TMP / "prueba.db")

from backend import servicio as S  # noqa: E402
from backend.base_datos import DIR_SIMULADO  # noqa: E402

fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'OK   ' if cond else 'FALLA'} {nombre}{(' | ' + str(detalle)) if detalle else ''}")
    if not cond:
        fallos.append(nombre)


def csv_semana(s):
    return (Path(DIR_SIMULADO) / f"semana_{s}.csv").read_bytes()


def mod(b, f):
    d = pd.read_csv(io.BytesIO(b))
    d = f(d)
    return d.to_csv(index=False).encode("utf-8")


try:
    print("0. Arranque de la base")
    import time
    t0 = time.time()
    S._asegurar_base()
    t_arr = time.time() - t0
    check("la base existe en BG_DB", Path(os.environ["BG_DB"]).exists())
    if S.SEMILLA.exists():
        check("con semilla el arranque es rapido (copia)", t_arr < 10, f"{t_arr:.1f} s")
    else:
        print(f"  (sin semilla: se creo la base en {t_arr:.0f} s)")
    # errores claros de crear_base_demo
    import backend.servicio as SV
    vacio = TMP / "vacia"
    vacio.mkdir()
    orig = SV.DIR_SIMULADO
    SV.DIR_SIMULADO = vacio
    try:
        SV.crear_base_demo(TMP / "x.db")
        check("sin semana_*.csv lanza error", False)
    except RuntimeError as e:
        check("sin semana_*.csv lanza error claro", "semana_*.csv" in str(e), str(e)[:80])
    SV.DIR_SIMULADO = orig
    try:
        SV.crear_base_demo(TMP / "x.db", hasta="2030-01-07")
        check("hasta inalcanzable lanza error", False)
    except RuntimeError as e:
        check("hasta inalcanzable lanza error claro y no deja base", "no alcanzan" in str(e) and not (TMP / "x.db").exists(), str(e)[:90])

    print("\n1. Funciones de lectura")
    sem = S.semanas_disponibles()
    u = S.ultima_semana()
    check("semanas_disponibles es lista de str ordenada", isinstance(sem, list) and sem == sorted(sem) and isinstance(sem[0], str), f"{len(sem)} semanas, de {sem[0]} a {sem[-1]}")
    check(f"la precarga llega a {S.PRECARGA_HASTA}", u == S.PRECARGA_HASTA, u)

    p = S.predicciones(u)
    cols_p = ["municipio_norm", "uf", "lat", "lon", "focos_7d", "focos_30d", "clase_riesgo", "prob_bajo", "prob_moderado", "prob_alto", "prob_critico", "es_alerta", "focos_real_7d"]
    check("predicciones: columnas del contrato", list(p.columns) == cols_p, f"{len(p)} municipios")
    check("predicciones: es_alerta booleana y sin lat/lon nulos", p.es_alerta.dtype == bool and p[["lat", "lon"]].notna().all().all())
    check("predicciones(uf) filtra", set(S.predicciones(u, "PA").uf) == {"PA"})
    check("la ultima semana no tiene focos reales (aun no ingestada la siguiente)", p.focos_real_7d.isna().all())
    check("una semana anterior si tiene focos reales", S.predicciones(sem[-3]).focos_real_7d.notna().all())

    a = S.tabla_alertas(u)
    cols_a = ["municipio_norm", "uf", "nivel_focos", "focos_7d", "focos_30d", "clase_riesgo", "prob_critico", "es_alerta", "prioridad"]
    check("tabla_alertas: columnas del contrato", list(a.columns) == cols_a)
    check("tabla_alertas: prioridad 1..n y alertas primero", list(a.prioridad) == list(range(1, len(a) + 1)) and a.es_alerta.is_monotonic_decreasing and a.es_alerta.dtype == bool)
    check("tabla_alertas: mismas alertas que predicciones", int(a.es_alerta.sum()) == int(p.es_alerta.sum()), f"{int(a.es_alerta.sum())} alertas")

    s_ = S.serie_real_vs_predicho()
    check("serie_real_vs_predicho: columnas", list(s_.columns) == ["semana", "real_7d", "pred_7d"] and len(s_) == len(sem))
    check("serie: la ultima semana sin real (nulo) y las anteriores con real", pd.isna(s_.real_7d.iloc[-1]) and s_.real_7d.iloc[:-1].notna().all())
    check("serie(uf) filtra", S.serie_real_vs_predicho("PA").pred_7d.sum() < s_.pred_7d.sum())

    c = S.contexto(u)
    check("contexto: columnas", list(c.columns) == ["uf", "riesgo_fuego_promedio", "pm2_5", "deter_area_ha"] and len(c) > 0, f"{len(c)} estados")

    k = S.kpis_control()
    check("kpis_control: columnas", list(k.columns) == ["semana_base", "mae_movil_4s", "recall_alerta", "falsas_alarmas", "estado"] and len(k) > 0, f"{len(k)} semanas")
    check("kpis_control: estado dentro o fuera", set(k.estado) <= {"dentro", "fuera"})

    e = S.estado_modelo()
    check("estado_modelo: claves y estado valido", {"estado", "mensaje", "semanas_fuera_umbral", "umbrales"} <= set(e) and e["estado"] in ("estable", "vigilancia", "reentrenar", "sin_dato")
          and {"mae_movil_max", "recall_alerta_min", "falsas_alarmas_max"} <= set(e["umbrales"]), f"{e['estado']} | fuera {e['semanas_fuera_umbral']}")

    f = S.fichas()
    r = S.resumen_kpis(u)
    claves = {"id", "nombre", "valor", "unidad", "estado", "umbral_ambar", "umbral_rojo", "base_umbral", "lectura"}
    check("fichas: lista de 10", isinstance(f, list) and len(f) == 10)
    check("resumen_kpis: claves del contrato", all(claves <= set(x) for x in r), f"{len(r)} KPI")
    check("resumen_kpis: ids iguales a los de fichas", {x['id'] for x in r} == {x['id'] for x in f}, sorted({x['id'] for x in f} ^ {x['id'] for x in r}))
    check("resumen_kpis: estados validos", all(x["estado"] in ("verde", "ambar", "rojo", "sin_dato") for x in r), {x["id"]: x["estado"] for x in r})
    check("resumen_kpis de una semana antigua funciona", len(S.resumen_kpis(sem[10])) == 10)

    um = S.umbrales_contexto()
    check("umbrales_contexto: claves", {"riesgo_fuego", "pm2_5", "deter"} <= set(um) and len(um["riesgo_fuego"]["cortes"]) == 4 and um["deter"]["ambar"] < um["deter"]["rojo"])

    h0 = S.historial_ingestas()
    check("historial_ingestas: columnas", list(h0.columns) == ["momento", "semana", "modo", "n_alertas"] and len(h0) > 0, f"{len(h0)} filas")

    print("\n2. Ingesta por el servicio")
    sig = S._mas7(u)
    b = csv_semana(sig)
    v = S.validar_semana(b, f"semana_{sig}.csv")
    check("validar_semana: archivo bueno ok, mensajes con prefijo", v["ok"] and all(m.split(":")[0] in ("OK", "AVISO", "ERROR") for m in v["mensajes"]), v["mensajes"][:2])
    check("validar_semana no guarda nada", S.ultima_semana() == u)
    r1 = S.ingestar_y_predecir(b, f"semana_{sig}.csv")
    check("ingestar_y_predecir: ok y claves", r1["ok"] and r1["semana"] == sig and isinstance(r1["n_alertas"], int), r1["mensaje"])
    check("la semana nueva aparece en ultima_semana()", S.ultima_semana() == sig and sig in S.semanas_disponibles())
    check("la semana nueva ya tiene alertas y KPI", len(S.tabla_alertas(sig)) == 543 and len(S.resumen_kpis(sig)) == 10)
    check("el historial registra la ingesta", S.historial_ingestas().iloc[0].semana == sig)

    print("\n3. Casos de rechazo (la base no cambia)")

    def foto():
        c_ = S._conn()
        try:
            return (c_.execute("SELECT COUNT(*) FROM observaciones_semanales").fetchone()[0], c_.execute("SELECT COUNT(*) FROM predicciones").fetchone()[0], S.ultima_semana())
        finally:
            c_.close()

    antes = foto()
    sig2 = S._mas7(sig)
    sig3 = S._mas7(sig2)
    casos = {
        "semana repetida": (csv_semana(sig), "repetida.csv"),
        "semana saltada": (csv_semana(sig3), "saltada.csv"),
        "columna faltante": (mod(csv_semana(sig2), lambda d: d.drop(columns=["num_incendios"])), "sin_columna.csv"),
        "humedad 150": (mod(csv_semana(sig2), lambda d: d.assign(RH2M=150.0)), "humedad.csv"),
    }
    for nombre, (bb, fn) in casos.items():
        v = S.validar_semana(bb, fn)
        r = S.ingestar_y_predecir(bb, fn)
        err = [m for m in v["mensajes"] if m.startswith("ERROR")]
        check(f"{nombre}: validar rechaza con ERROR y ingestar no guarda", (not v["ok"]) and bool(err) and (not r["ok"]), err[0] if err else v["mensajes"])
    check("la base no cambio tras los 4 rechazos", foto() == antes, foto())
    check("los rechazos quedan en el historial", (S.historial_ingestas().modo.str.startswith("rechazada")).sum() >= 4)

    print("\n4. Simular siguiente semana")
    r2 = S.simular_siguiente_semana()
    check("simular_siguiente_semana: toma la siguiente", r2["ok"] and r2["semana"] == sig2, r2["mensaje"])
    check("ultima_semana avanza", S.ultima_semana() == sig2)
    check("el KPI de control se recalcula con la nueva semana", len(S.kpis_control()) >= len(k))
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if fallos:
    print(f"FALLA: {len(fallos)} comprobaciones: {fallos}")
    sys.exit(1)
print("OK: todas las comprobaciones del servicio pasaron")
