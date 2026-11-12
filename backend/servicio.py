"""Servicio real de BIODIVERSITY-GUARD-Predict: mismo contrato que frontend/servicio_stub.py, leyendo de la base SQLite y de kpis.py.

Base de datos: ruta en la variable de entorno BG_DB (por defecto backend/biodiversity_guard.db). Si no existe se crea al primer uso
(crear_base + precarga de las semanas de 2026 hasta PRECARGA_HASTA, unos 30 a 90 s). Las semanas siguientes de datos/modulo_ia_simulado/
quedan para "Simular siguiente semana" y para subir por la pagina de Ingesta.

Los umbrales y reglas de KPI se toman siempre de backend/kpis.py y de los JSON; aqui no se copian constantes.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend import kpis as K  # noqa: E402
from backend.base_datos import COLUMNAS_PANEL, DIR_SIMULADO, RUTA_DB, conectar, ingestar_semana, predecir_semana, validar_archivo  # noqa: E402
from ia_models.pipeline import cargar_metadata  # noqa: E402

PRECARGA_HASTA = "2026-08-31"  # ultima semana de 2026 ya ingestada en la base de demostracion; las siguientes se simulan o se suben
SEMANA_FIN_HISTORIAL = "2025-12-29"
CLASES = ["Bajo", "Moderado", "Alto", "Crítico"]
SEMILLA = RAIZ / "backend" / "semilla" / "biodiversity_guard_semilla.db"  # base ya creada hasta PRECARGA_HASTA (se versiona en git)
_BLOQUEO = threading.RLock()
_RUTA_ALTERNA: list = []  # si no se puede escribir en BG_DB, se usa un directorio temporal
_CACHE: dict = {}
ESTADO_BASE = {"mensaje": "sin iniciar"}


def ruta_db() -> Path:
    if _RUTA_ALTERNA:
        return _RUTA_ALTERNA[0]
    return Path(os.environ.get("BG_DB") or RUTA_DB)


def _progreso(msg: str):
    ESTADO_BASE["mensaje"] = msg
    print(f"[servicio] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ base
def _ultima_semana_de(ruta) -> str | None:
    try:
        c = conectar(ruta)
        try:
            return c.execute("SELECT MAX(semana_base) FROM predicciones").fetchone()[0]
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None


def crear_base_demo(destino=None, hasta: str = PRECARGA_HASTA, progreso=_progreso) -> Path:
    """Crea la base de demostracion: historial hasta 2025-12-29 y semanas de 2026 ingestadas y predichas hasta `hasta`.

    Lanza RuntimeError si no hay semana_*.csv o si las semanas disponibles no llegan a `hasta`.
    """
    from backend.crear_base import crear_base  # import tardio: solo hace falta al crear

    destino = Path(destino or ruta_db())
    todos = sorted(Path(DIR_SIMULADO).glob("semana_*.csv"))
    if not todos:
        raise RuntimeError(f"No hay archivos semana_*.csv en {DIR_SIMULADO}; no se puede precargar la base de demostración.")
    archivos = [a for a in todos if a.stem.replace("semana_", "") <= hasta]
    ultima_disp = archivos[-1].stem.replace("semana_", "") if archivos else None
    if ultima_disp != hasta:
        raise RuntimeError(f"Las semanas simuladas disponibles llegan hasta {ultima_disp or 'ninguna'} y no alcanzan {hasta} "
                           f"(falta semana_{hasta}.csv en {DIR_SIMULADO}).")
    tmp = destino.with_name(destino.name + ".creando")
    t0 = time.time()
    progreso(f"Creando la base {destino.name} (unos 30 a 90 s la primera vez)...")
    try:
        with tempfile.TemporaryDirectory(prefix="bg_sim_") as carpeta:
            conn = crear_base(tmp, carpeta_simulada=carpeta)  # la simulacion se escribe en una carpeta temporal, no en el repositorio
            for i, a in enumerate(archivos, 1):
                ok, msg, _ = ingestar_semana(conn, a)
                if not ok:
                    raise RuntimeError(f"precarga: {a.name} rechazada: {msg}")
                predecir_semana(conn, a.stem.replace("semana_", ""))
                conn.execute("UPDATE ingestas SET mensaje = ? WHERE id = (SELECT MAX(id) FROM ingestas)", (f"precarga: {a.name}",))
                conn.commit()
                if i % 5 == 0 or i == len(archivos):
                    progreso(f"Precarga de 2026: {i} de {len(archivos)} semanas...")
            real = conn.execute("SELECT MAX(semana_base) FROM predicciones").fetchone()[0]
            conn.close()
        if real != hasta:
            raise RuntimeError(f"La precarga terminó en {real} y no en {hasta}.")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    destino.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, destino)
    progreso(f"Base lista en {time.time() - t0:,.0f} s (precargada hasta {real}).")
    return destino


def crear_semilla() -> Path:
    """Regenera backend/semilla/biodiversity_guard_semilla.db (base ya creada hasta PRECARGA_HASTA)."""
    return crear_base_demo(SEMILLA)


def _copiar_semilla(destino: Path) -> bool:
    """Copia la semilla a `destino` si existe y llega a PRECARGA_HASTA. True si la copio."""
    if not SEMILLA.exists() or _ultima_semana_de(SEMILLA) != PRECARGA_HASTA:
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".copiando")
    shutil.copyfile(SEMILLA, tmp)
    os.replace(tmp, destino)
    return True


def _asegurar_base():
    r = ruta_db()
    if r.exists():
        return
    with _BLOQUEO:
        if r.exists():
            return
        for destino in (r, Path(tempfile.gettempdir()) / "biodiversity_guard" / "biodiversity_guard.db"):
            try:
                if _copiar_semilla(destino):
                    if destino != r:
                        _RUTA_ALTERNA[:] = [destino]
                    _progreso(f"Base copiada de la semilla (hasta {PRECARGA_HASTA}) en {destino}.")
                    return
                break  # no hay semilla valida: se crea
            except OSError as e:  # sin permiso de escritura en BG_DB: probar el directorio temporal
                _progreso(f"No se pudo copiar la semilla a {destino}: {e}")
        try:
            crear_base_demo(r)
        except OSError:
            alt = Path(tempfile.gettempdir()) / "biodiversity_guard" / "biodiversity_guard.db"
            _RUTA_ALTERNA[:] = [alt]
            crear_base_demo(alt)


def _conn():
    _asegurar_base()
    return conectar(ruta_db())


def _q(sql, params=(), parse_dates=None) -> pd.DataFrame:
    c = _conn()
    try:
        return pd.read_sql_query(sql, c, params=params, parse_dates=parse_dates)
    finally:
        c.close()


def _mas7(semana: str) -> str:
    return (date.fromisoformat(semana) + timedelta(days=7)).isoformat()


def _sem(semana) -> str:
    return str(pd.Timestamp(semana).date()) if semana else ultima_semana()


# ---------------------------------------------------------------- semanas
def semanas_disponibles() -> list[str]:
    return _q("SELECT DISTINCT semana_base FROM predicciones ORDER BY semana_base")["semana_base"].tolist()


def ultima_semana() -> str:
    return semanas_disponibles()[-1]


def _version_cache() -> str:
    """Cambia cuando se ingesta una semana (sirve de llave de cache interna)."""
    return ultima_semana()


# ------------------------------------------------------------ predicciones
def _obs_semana(semana: str) -> pd.DataFrame:
    return _q("SELECT * FROM observaciones_semanales WHERE semana = ?", (semana,))


def _pred_semana(semana: str) -> pd.DataFrame:
    return _q("SELECT * FROM predicciones WHERE semana_base = ?", (semana,))


def predicciones(semana, uf=None) -> pd.DataFrame:
    s = _sem(semana)
    d = _q("""SELECT p.municipio_norm, p.uf, m.lat, m.lon, p.focos_7d, p.focos_30d, p.clase_riesgo, p.prob_bajo, p.prob_moderado,
                     p.prob_alto, p.prob_critico, p.es_alerta, o.num_incendios AS focos_real_7d
              FROM predicciones p
              LEFT JOIN municipios m ON m.municipio_norm = p.municipio_norm AND m.uf = p.uf
              LEFT JOIN observaciones_semanales o ON o.semana = ? AND o.municipio_norm = p.municipio_norm AND o.uf = p.uf
              WHERE p.semana_base = ?""", (_mas7(s), s))
    if uf:
        d = d[d.uf == uf]
    d["es_alerta"] = d["es_alerta"].astype(bool)
    d["clase_riesgo"] = d["clase_riesgo"].fillna("Sin dato")
    return d.reset_index(drop=True)


def tabla_alertas(semana) -> pd.DataFrame:
    s = _sem(semana)
    t = K.tabla_alertas(_pred_semana(s), _obs_semana(s))
    t["es_alerta"] = t["es_alerta"].astype(bool)
    return t


def _real_vs_pred_todo() -> pd.DataFrame:
    """semana_base, municipio, uf, predicho (focos 7 d) y real (focos observados en la semana siguiente, nulo si aun no existe)."""
    return _q("""SELECT p.semana_base, p.municipio_norm, p.uf, p.focos_7d AS predicho, o.num_incendios AS real
                 FROM predicciones p
                 LEFT JOIN observaciones_semanales o ON o.municipio_norm = p.municipio_norm AND o.uf = p.uf
                      AND o.semana = date(p.semana_base, '+7 days')""")


def serie_real_vs_predicho(uf=None) -> pd.DataFrame:
    d = _real_vs_pred_todo()
    if uf:
        d = d[d.uf == uf]
    g = d.groupby("semana_base").agg(real_7d=("real", "sum"), pred_7d=("predicho", "sum"), n_real=("real", "count")).reset_index()
    g.loc[g.n_real == 0, "real_7d"] = np.nan
    return g.rename(columns={"semana_base": "semana"})[["semana", "real_7d", "pred_7d"]]


# ---------------------------------------------------------------- contexto
def contexto(semana) -> pd.DataFrame:
    s = _sem(semana)
    return _q("""SELECT uf, AVG(riesgo_fuego_promedio) AS riesgo_fuego_promedio, AVG(pm2_5_reanalise) AS pm2_5, SUM(deter_area_ha) AS deter_area_ha
                 FROM observaciones_semanales WHERE semana = ? GROUP BY uf ORDER BY uf""", (s,))


def umbrales_contexto() -> dict:
    ref = json.loads(K.RUTA_KPI.read_text(encoding="utf-8"))["referencias_de_contexto"]
    v = ref["valor"]
    p75, p90 = K.umbrales_deter()
    return {
        "riesgo_fuego": {"cortes": v["riesgo_fuego_inpe"], "clases": ["Mínimo", "Bajo", "Medio", "Alto", "Crítico"], "fuente": "Cortes del INPE; " + ref["fuente"]},
        "pm2_5": {"ambar": v["pm2_5_oms_ug_m3"], "rojo": v["pm2_5_conama_pi1_ug_m3"], "unidad": "µg/m³", "fuente": "Guía OMS (15) y estándar CONAMA PI-1 (60)"},
        "deter": {"ambar": float(p75), "rojo": float(p90), "unidad": "ha por semana (suma de los estados)",
                  "fuente": "P75 y P90 del área semanal DETER en 2022 a 2024 (solo entrenamiento)"},
    }


# ------------------------------------------------------------- KPI control
def _control_df() -> pd.DataFrame:
    """KPI de control de kpis.py sobre las predicciones con dato real. Cache por ultima semana."""
    clave = (str(ruta_db()), ultima_semana())
    if _CACHE.get("ctrl_clave") != clave:
        d = _real_vs_pred_todo().dropna(subset=["real"])
        _CACHE["ctrl"] = K.kpis_control(d[["semana_base", "real", "predicho"]]) if len(d) else pd.DataFrame(columns=["semana_base", "mae_movil_4s", "recall_alerta", "falsas_alarmas", "estado"])
        _CACHE["ctrl_clave"] = clave
    return _CACHE["ctrl"].copy()


def kpis_control() -> pd.DataFrame:
    k = _control_df()
    k["estado"] = np.where(k["estado"] == "normal", "dentro", "fuera")
    return k


def estado_modelo() -> dict:
    k = _control_df()
    u = K.umbrales_control()
    umb = {"mae_movil_max": u["mae_max"], "recall_alerta_min": u["recall_min"], "falsas_alarmas_max": u["fa_max"]}
    if k.empty:
        return dict(estado="sin_dato", mensaje="Aún no hay 4 semanas con dato real para evaluar el modelo.", semanas_fuera_umbral=0, umbrales=umb)
    e = K.estado_modelo(k)
    n = int((k.tail(4).estado != "normal").sum())
    est = {"normal": "estable", "vigilar": "vigilancia", "reentrenar recomendado": "reentrenar"}[e["estado"]]
    msg = {"estable": "Modelo estable: " + e["mensaje"], "vigilancia": "En vigilancia: " + e["mensaje"],
           "reentrenar": "Reentrenar recomendado: " + e["mensaje"]}[est]
    return dict(estado=est, mensaje=msg, semanas_fuera_umbral=n, umbrales=umb)


# ---------------------------------------------------------------------- KPI
def _kpis_control_tarjetas(semana: str) -> list[dict]:
    u = K.umbrales_control()
    k = _control_df()
    k = k[k.semana_base <= semana]
    f = k.iloc[-1] if len(k) else None
    base = K._cargar()[2]["kpi_control"]["regla"]
    nota = "" if f is None else f" (ventana de 4 semanas que termina en la semana base {f.semana_base})"

    def tarjeta(id_, nombre, valor, unidad, umbral, tipo_min, lectura_ok, lectura_mal):
        if valor is None or pd.isna(valor):
            return dict(id=id_, nombre=nombre, valor=None, unidad=unidad, estado="sin_dato", umbral_ambar=umbral, umbral_rojo=umbral,
                        base_umbral=base, lectura="Sin dato: faltan semanas con dato real para esta ventana.")
        malo = valor < umbral if tipo_min else valor > umbral
        return dict(id=id_, nombre=nombre, valor=round(float(valor), 3), unidad=unidad, estado="rojo" if malo else "verde",
                    umbral_ambar=umbral, umbral_rojo=umbral, base_umbral=base, lectura=(lectura_mal if malo else lectura_ok) + nota + ".")

    v = lambda c, esc=1: None if f is None or pd.isna(f[c]) else f[c] * esc  # noqa: E731
    return [
        tarjeta("control_mae_movil", "Error medio móvil (4 semanas)", v("mae_movil_4s"), "focos por municipio", u["mae_max"], False,
                "Error de la proyección de focos a 7 días dentro del umbral", "Error de la proyección fuera del umbral"),
        tarjeta("control_recall_alerta", "Alertas detectadas (recall)", v("recall_alerta", 100), "%", 100 * u["recall_min"], True,
                "Se detecta la parte esperada de los municipios en nivel Muy elevado", "Se están dejando sin alertar municipios en nivel Muy elevado"),
        tarjeta("control_falsas_alarmas", "Falsas alarmas", v("falsas_alarmas", 100), "%", 100 * u["fa_max"], False,
                "Pocas alertas sin nivel Muy elevado real", "Demasiadas alertas sin nivel Muy elevado real"),
    ]


def resumen_kpis(semana) -> list[dict]:
    s = _sem(semana)
    ini = (date.fromisoformat(s) - timedelta(weeks=60)).isoformat()
    hist = _q("""SELECT semana, uf, SUM(num_incendios) AS num_incendios FROM observaciones_semanales
                 WHERE semana BETWEEN ? AND ? GROUP BY semana, uf""", (ini, s))
    res = K.kpis_resultado(_obs_semana(s), _pred_semana(s), hist, s)
    for x in res:  # el dashboard compara los dos umbrales: sin ambar se iguala al rojo (la tarjeta dice "no se usa")
        if x.get("umbral_ambar") is None:
            x["umbral_ambar"] = x["umbral_rojo"]
    return res + _kpis_control_tarjetas(s)


# ------------------------------------------------------------------ fichas
def fichas() -> list:
    return json.loads(K.RUTA_FICHAS.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ ingesta
def _guardar_temp(archivo_bytes, nombre) -> Path:
    carpeta = Path(tempfile.mkdtemp(prefix="bg_ingesta_"))
    ruta = carpeta / (Path(str(nombre)).name or "semana.csv")
    ruta.write_bytes(archivo_bytes)
    return ruta


def _limpiar(ruta: Path):
    try:
        ruta.unlink(missing_ok=True)
        ruta.parent.rmdir()
    except OSError:
        pass


def validar_semana(archivo_bytes, nombre) -> dict:
    """Valida sin guardar nada. Mensajes con prefijo OK:, AVISO: o ERROR:."""
    ruta = _guardar_temp(archivo_bytes, nombre)
    try:
        conn = _conn()
        try:
            df, semana, error = validar_archivo(conn, ruta)
            if error:
                return dict(ok=False, mensajes=[f"ERROR: {error[0].upper()}{error[1:]}."])
            ult = ultima_semana()
            m = [f"OK: columnas obligatorias presentes ({len(COLUMNAS_PANEL)} del panel crudo).",
                 f"OK: una sola semana, {semana.date()} (lunes), siguiente a la última cargada ({ult}).",
                 "OK: tipos numéricos, sin nulos obligatorios y valores dentro de rango.",
                 f"OK: {len(df)} municipios, todos conocidos y sin duplicados."]
            if (df["num_incendios"] == 0).all():
                m.append("AVISO: todos los municipios tienen 0 focos; revise si el archivo es correcto.")
            return dict(ok=True, mensajes=m)
        finally:
            conn.close()
    finally:
        _limpiar(ruta)


def _ingestar(ruta: Path, modo: str) -> dict:
    with _BLOQUEO:
        conn = _conn()
        try:
            df, semana, error = validar_archivo(conn, ruta)
            ok, msg, _ = ingestar_semana(conn, ruta)  # si falla deja el registro 'rechazada' y no guarda nada
            if not ok:
                conn.execute("UPDATE ingestas SET mensaje = ? WHERE id = (SELECT MAX(id) FROM ingestas)", (f"{modo}: {msg}",))
                conn.commit()
                return dict(ok=False, mensaje=f"Semana rechazada: {msg}.", semana=None if semana is None else str(semana.date()), n_alertas=0)
            conn.execute("UPDATE ingestas SET mensaje = ? WHERE id = (SELECT MAX(id) FROM ingestas)", (f"{modo}: {ruta.name}",))
            conn.commit()
            try:
                R = predecir_semana(conn, semana)
            except Exception as e:  # noqa: BLE001  la semana ya esta guardada pero sin prediccion: se deshace para no dejar la base a medias
                with conn:
                    conn.execute("DELETE FROM observaciones_semanales WHERE semana = ?", (str(semana.date()),))
                    conn.execute("UPDATE ingestas SET estado = 'rechazada', mensaje = ? WHERE id = (SELECT MAX(id) FROM ingestas)", (f"{modo}: error al predecir: {e}",))
                return dict(ok=False, mensaje=f"No se pudo calcular la proyección: {e}", semana=str(semana.date()), n_alertas=0)
            n = int(R["es_alerta"].sum())
            return dict(ok=True, mensaje=f"Semana {semana.date()} cargada y proyectada: {n} municipios en alerta.", semana=str(semana.date()), n_alertas=n)
        finally:
            conn.close()


def ingestar_y_predecir(archivo_bytes, nombre) -> dict:
    ruta = _guardar_temp(archivo_bytes, nombre)
    try:
        return _ingestar(ruta, "subida")
    finally:
        _limpiar(ruta)


def simular_siguiente_semana() -> dict:
    s = _mas7(ultima_semana())
    ruta = Path(DIR_SIMULADO) / f"semana_{s}.csv"
    if not ruta.exists():
        return dict(ok=False, mensaje=f"No hay más semanas simuladas: falta {ruta.name} en datos/modulo_ia_simulado.", semana=None, n_alertas=0)
    return _ingestar(ruta, "simulación")


def predecir(semana) -> dict:
    """Recalcula y guarda la prediccion de una semana ya ingestada (para la API). No ingesta nada."""
    s = _sem(semana)
    with _BLOQUEO:
        conn = _conn()
        try:
            if not conn.execute("SELECT 1 FROM observaciones_semanales WHERE semana = ? LIMIT 1", (s,)).fetchone():
                return dict(ok=False, mensaje=f"La semana {s} no está ingestada; no hay datos para predecir.", semana=s, n_alertas=0)
            R = predecir_semana(conn, s)
            n = int(R["es_alerta"].sum())
            return dict(ok=True, mensaje=f"Semana {s} proyectada: {n} municipios en alerta.", semana=s, n_alertas=n)
        finally:
            conn.close()


def historial_ingestas() -> pd.DataFrame:
    h = _q("""SELECT i.fecha AS momento, i.semana, i.estado, i.mensaje,
                     (SELECT SUM(es_alerta) FROM predicciones p WHERE p.semana_base = i.semana) AS n_alertas
              FROM ingestas i ORDER BY i.id DESC""")
    h["modo"] = np.where(h.estado == "aceptada", h.mensaje, h.estado + ": " + h.mensaje.fillna(""))
    h["n_alertas"] = h["n_alertas"].where(h.estado != "rechazada")
    return h[["momento", "semana", "modo", "n_alertas"]]


# ---------------------------------------------------------------- estado
def info_modelo() -> dict:
    m = cargar_metadata()
    return {"version": m["version_modelos"], "fecha_entrenamiento": m["fecha_entrenamiento"]}


def salud() -> dict:
    """Comprobacion de conexion para la API."""
    c = _conn()
    try:
        c.execute("SELECT 1").fetchone()
    finally:
        c.close()
    return {"base": "ok", "ruta_base": str(ruta_db()), "ultima_semana": ultima_semana(), "modelo": info_modelo()}


if __name__ == "__main__":
    if "--crear-semilla" in sys.argv:
        print(crear_semilla())
    else:
        print("Uso: python backend/servicio.py --crear-semilla")
