"""BIODIVERSITY-GUARD-Predict: dashboard de KPI, alertas y proyección.

Solo usa lo que devuelve conexion.obtener_servicio(); no lee CSV ni modelos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conexion import obtener_servicio, semana_ejemplo  # noqa: E402

st.set_page_config(page_title="BIODIVERSITY-GUARD-Predict", page_icon="🌳", layout="wide")

@st.cache_resource(show_spinner="Preparando la base de datos...")
def _iniciar():
    sv, aviso = obtener_servicio()
    sv.semanas_disponibles()  # la primera llamada crea/precarga la base si hace falta
    return sv, aviso


SV, AVISO_SERVICIO = _iniciar()

# ------------------------------------------------------------ estilo
ROJO, AMBAR, VERDE, GRIS = "#B71C1C", "#B26A00", "#2E7D32", "#6B7280"
AZUL, AZUL_CLARO = "#1F5A8C", "#9DB7CF"
ICONO = {"verde": "🟢", "ambar": "🟠", "rojo": "🔴", "sin_dato": "⚪"}
TEXTO = {"verde": "Verde", "ambar": "Ámbar", "rojo": "Rojo", "sin_dato": "Sin dato"}
COLOR_E = {"verde": VERDE, "ambar": AMBAR, "rojo": ROJO, "sin_dato": GRIS}
# Rojo solo para Muy elevado y Crítico; el resto en tonos neutros
COLOR_NIVEL = {"Sin focos": "#E5E7EB", "Habitual": "#B8C4D0", "Elevado": "#5B7A99", "Muy elevado": ROJO}
COLOR_CLASE = {"Bajo": "#D6DEE6", "Moderado": "#9DB7CF", "Alto": "#4F7399", "Crítico": ROJO, "Sin dato": "#E5E7EB"}
ORDEN_CLASE = ["Bajo", "Moderado", "Alto", "Crítico", "Sin dato"]
ORDEN_NIVEL = ["Sin focos", "Habitual", "Elevado", "Muy elevado"]

st.markdown(f"""<style>
.block-container {{padding-top: 1.6rem; max-width: 1300px;}}
.kpi {{border:1px solid #D9DEE4; border-top:5px solid var(--c); border-radius:8px; padding:12px 14px; background:#FFFFFF; min-height:335px;}}
.kpi .n {{font-size:0.82rem; color:#374151; font-weight:600; min-height:2.4em;}}
.kpi .v {{font-size:2rem; font-weight:700; color:#111827; line-height:1.15;}}
.kpi .u {{font-size:0.8rem; color:#4B5563;}}
.kpi .e {{display:inline-block; margin:6px 0; padding:1px 9px; border-radius:10px; font-size:0.8rem; font-weight:700; color:#fff; background:var(--c);}}
.kpi .t {{font-size:0.76rem; color:#374151;}}
.kpi .l {{font-size:0.78rem; color:#111827; margin-top:6px;}}
.banner {{border-radius:8px; padding:12px 16px; margin:10px 0; background:#F3F4F6; border-left:6px solid var(--c); color:#111827;}}
.estado {{border-radius:8px; padding:8px 14px; background:#F3F4F6; border-left:6px solid var(--c); color:#111827;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------ datos (con caché)
# v = última semana: invalida la caché cuando se ingresa una semana nueva
@st.cache_data(show_spinner=False)
def c_resumen(s, v): return SV.resumen_kpis(s)
@st.cache_data(show_spinner=False)
def c_alertas(s, v): return SV.tabla_alertas(s)
@st.cache_data(show_spinner=False)
def c_pred(s, uf, v): return SV.predicciones(s, uf)
@st.cache_data(show_spinner=False)
def c_serie(uf, v): return SV.serie_real_vs_predicho(uf)
@st.cache_data(show_spinner=False)
def c_ctx(s, v): return SV.contexto(s)
@st.cache_data(show_spinner=False)
def c_ctrl(v): return SV.kpis_control()
@st.cache_data(show_spinner=False)
def c_estado(v): return SV.estado_modelo()
@st.cache_data(show_spinner=False)
def c_fichas(v):
    d = SV.fichas()
    return {x.get("id"): x for x in d} if isinstance(d, list) else d
@st.cache_data(show_spinner=False)
def c_umbr_ctx(v): return SV.umbrales_contexto() if hasattr(SV, "umbrales_contexto") else None


def vacio(df, msg="No hay datos para mostrar con esta selección."):
    if df is None or len(df) == 0:
        st.info(msg)
        return True
    return False


def fmt(v, unidad=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Sin dato"
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}" if abs(v) < 10 else f"{v:,.1f}"


# ------------------------------------------------------------ barra lateral
PAGINAS = ["Panel de control", "Proyección", "Riesgo de fuego", "Contexto", "Control del modelo", "Ingesta",
           "Ayuda y límites"]
if "pagina" not in st.session_state:
    q = st.query_params.get("pagina")
    st.session_state["pagina"] = q if q in PAGINAS else PAGINAS[0]
st.sidebar.title("🌳 BIODIVERSITY-GUARD")
st.sidebar.caption("Predicción de focos de incendio y alertas")
pagina = st.sidebar.radio("Página", PAGINAS, key="pagina")

V = SV.ultima_semana()
semanas = SV.semanas_disponibles()
if "semana_sel" not in st.session_state or st.session_state["semana_sel"] not in semanas:
    st.session_state["semana_sel"] = V
SEM = st.sidebar.selectbox("Semana (lunes)", semanas[::-1], key="semana_sel")
if AVISO_SERVICIO:
    st.warning(AVISO_SERVICIO)
st.sidebar.caption("Servicio: " + ("de ejemplo (stub)" if "stub" in SV.__name__ else "real"))

EST = c_estado(V)
ESTADO_COLOR = {"estable": VERDE, "vigilancia": AMBAR, "reentrenar": ROJO, "sin_dato": GRIS}
ESTADO_ICONO = {"estable": "🟢", "vigilancia": "🟠", "reentrenar": "🔴", "sin_dato": "⚪"}


def cabecera(titulo):
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"## {titulo}")
    e = EST.get("estado", "sin_dato")
    c2.markdown(f"<div class='estado' style='--c:{ESTADO_COLOR.get(e, GRIS)}'><b>Semana {SEM}</b> · "
                f"Estado del modelo: {ESTADO_ICONO.get(e, '⚪')} <b>{e.capitalize()}</b><br>"
                f"<span style='font-size:.85rem'>{EST.get('mensaje', '')}</span></div>", unsafe_allow_html=True)


def tarjeta(k):
    e = k["estado"]
    if e == "sin_dato" or k["valor"] is None:
        val = "Sin dato"
    else:
        val = fmt(k["valor"])
    inv = k["umbral_rojo"] < k["umbral_ambar"]
    sg = "≤" if inv else "≥"
    sentido = "Menos de" if inv else "Más de"
    return (f"<div class='kpi' style='--c:{COLOR_E[e]}'><div class='n'>{k['nombre']}</div>"
            f"<div class='v'>{val}</div><div class='u'>{k['unidad'] if val != 'Sin dato' else ''}</div>"
            f"<span class='e'>{ICONO[e]} {TEXTO[e]}</span>"
            f"<div class='t'>Ámbar: {'no se usa' if k['umbral_ambar'] == k['umbral_rojo'] else sentido.lower() + ' ' + fmt(k['umbral_ambar'])} · Rojo: {sentido.lower()} {fmt(k['umbral_rojo'])}</div>"
            f"<div class='l'>{k['lectura']}</div></div>")


# ------------------------------------------------------------ 1. Panel
def p_panel():
    cabecera("Panel de control")
    kp = c_resumen(SEM, V)
    if not kp:
        st.info("El servicio no devolvió KPI para esta semana.")
    else:
        for i in range(0, len(kp), 5):
            cols = st.columns(5)
            for c, k in zip(cols, kp[i:i + 5]):
                c.markdown(tarjeta(k), unsafe_allow_html=True)
        with st.expander("¿De dónde sale este umbral?"):
            fi = c_fichas(V)
            tabs = st.tabs([k["nombre"] for k in kp])
            for t, k in zip(tabs, kp):
                with t:
                    f = fi.get(k["id"]) if isinstance(fi, dict) else None
                    if not f:
                        st.info("Este KPI aún no tiene ficha de sustento.")
                        continue
                    b = f.get("base_umbral") or f.get("base_del_umbral") or {}
                    if not isinstance(b, dict):
                        b = {"dato_pasado": str(b)}
                    um = f.get("umbral")
                    st.markdown(f"**Qué mide.** {f.get('que_mide', '')}")
                    st.markdown(f"**Fórmula.** {f.get('formula', '')}")
                    if isinstance(um, dict):
                        st.markdown("**Umbral.** " + " · ".join(f"{ICONO.get(n, '')} {n.capitalize()}: {tx}" for n, tx in um.items()))
                    st.markdown(f"**Base del umbral, dato pasado.** {b.get('dato_pasado') or b.get('dato pasado', '')}")
                    st.markdown(f"**Base del umbral, negocio.** {b.get('negocio') or b.get('experiencia del negocio', '')}")
                    st.markdown(f"**Limitaciones.** {f.get('limitaciones', '')}")
                    st.markdown(f"**Acción sugerida.** {f.get('accion_sugerida', '')}")
    al = c_alertas(SEM, V)
    if vacio(al, "No hay alertas disponibles para esta semana."):
        return
    sel = al[al.es_alerta].sort_values("prioridad")
    cap = next((k["umbral_ambar"] for k in kp if k["id"] == "alertas_vs_capacidad"), None)
    est_al = next((k["estado"] for k in kp if k["id"] == "alertas_vs_capacidad"), "sin_dato")
    txt = f"<b>{len(sel)} municipios en alerta</b>" + (f"; capacidad base {fmt(cap)}" if cap is not None else "")
    if cap is not None:
        txt += " · " + ("dentro de la capacidad" if len(sel) <= cap else "supera la capacidad base: atender por prioridad")
    st.markdown(f"<div class='banner' style='--c:{COLOR_E[est_al]}'>{ICONO[est_al]} {txt}</div>", unsafe_allow_html=True)
    if sel.empty:
        st.success("Ningún municipio en alerta esta semana.")
        return
    ufs = ["Todos"] + sorted(sel.uf.unique())
    uf = st.selectbox("Filtrar por estado", ufs, key="uf_panel")
    t = sel if uf == "Todos" else sel[sel.uf == uf]
    t = t.rename(columns={"prioridad": "Prioridad", "municipio_norm": "Municipio", "uf": "Estado",
                          "nivel_focos": "Nivel de focos", "focos_7d": "Focos 7 d", "focos_30d": "Focos 30 d",
                          "clase_riesgo": "Clase de riesgo", "prob_critico": "Prob. crítico"})
    cols = ["Prioridad", "Municipio", "Estado", "Nivel de focos", "Focos 7 d", "Focos 30 d", "Clase de riesgo", "Prob. crítico"]

    def pinta(v):
        return f"color:{ROJO};font-weight:700" if v in ("Muy elevado", "Crítico") else ""
    st.dataframe(t[cols].style.map(pinta, subset=["Nivel de focos", "Clase de riesgo"])
                 .format({"Focos 7 d": "{:.1f}", "Focos 30 d": "{:.1f}", "Prob. crítico": "{:.0%}"}, na_rep="Sin dato"),
                 hide_index=True, use_container_width=True, height=360)


# ------------------------------------------------------------ 2. Proyección
def p_proy():
    cabecera("Proyección de focos")
    al = c_alertas(SEM, V)
    ufs = sorted(al.uf.dropna().unique()) if len(al) else []
    uf = st.selectbox("Estado", ["Todos"] + ufs, key="uf_proy")
    ufp = None if uf == "Todos" else uf
    d = c_pred(SEM, ufp, V)
    if vacio(d):
        return
    a, b = st.columns(2)
    a.metric("Focos proyectados a 7 días", fmt(d.focos_7d.sum()))
    b.metric("Focos proyectados a 30 días", fmt(d.focos_30d.sum()))
    g = c_pred(SEM, None, V).groupby("uf")[["focos_7d", "focos_30d"]].sum().reset_index().sort_values("focos_7d", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Por estado**")
        m = g.melt("uf", var_name="Horizonte", value_name="Focos").replace({"focos_7d": "7 días", "focos_30d": "30 días"})
        f = px.bar(m, x="uf", y="Focos", color="Horizonte", barmode="group", color_discrete_map={"7 días": AZUL, "30 días": AZUL_CLARO},
                   labels={"uf": "Estado"})
        f.update_layout(height=340, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, use_container_width=True)
    with c2:
        st.markdown("**Por municipio (los 15 con más focos a 7 días)**")
        top = d.sort_values("focos_7d", ascending=False).head(15).iloc[::-1]
        f = go.Figure([go.Bar(y=top.municipio_norm + " (" + top.uf + ")", x=top.focos_7d, name="7 días", orientation="h", marker_color=AZUL),
                       go.Bar(y=top.municipio_norm + " (" + top.uf + ")", x=top.focos_30d, name="30 días", orientation="h", marker_color=AZUL_CLARO)])
        f.update_layout(height=340, margin=dict(t=10, b=10), barmode="group", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, use_container_width=True)
    st.markdown("**Real contra predicho (focos por semana, 7 días)**")
    s = c_serie(ufp, V)
    if not vacio(s):
        f = go.Figure()
        f.add_scatter(x=s.semana, y=s.real_7d, name="Real", line=dict(color=AZUL, width=2), connectgaps=False)
        f.add_scatter(x=s.semana, y=s.pred_7d, name="Predicho", line=dict(color="#E69F00", width=2, dash="dot"))
        f.update_layout(height=320, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1), yaxis_title="Focos")
        st.plotly_chart(f, use_container_width=True)
        if s.real_7d.isna().any():
            st.caption("Las semanas sin línea azul son proyecciones sin dato real todavía.")
    st.markdown("**Mapa por nivel de focos**")
    al2 = al if ufp is None else al[al.uf == ufp]
    m = d.merge(al2[["municipio_norm", "uf", "nivel_focos"]], on=["municipio_norm", "uf"], how="left").dropna(subset=["lat", "lon"])
    if vacio(m, "Sin coordenadas para dibujar el mapa."):
        return
    m["nivel_focos"] = pd.Categorical(m.nivel_focos.fillna("Sin focos"), ORDEN_NIVEL, ordered=True)
    m = m.sort_values("nivel_focos")
    m["tam"] = m.nivel_focos.map({"Sin focos": 4, "Habitual": 6, "Elevado": 9, "Muy elevado": 13}).astype(float)
    f = px.scatter_map(m, lat="lat", lon="lon", color="nivel_focos", size="tam", size_max=13, color_discrete_map=COLOR_NIVEL,
                       category_orders={"nivel_focos": ORDEN_NIVEL}, hover_name="municipio_norm",
                       hover_data={"uf": True, "focos_7d": ":.1f", "focos_30d": ":.1f", "lat": False, "lon": False, "tam": False},
                       zoom=3.2 if ufp is None else 5, height=520, map_style="carto-positron", labels={"nivel_focos": "Nivel de focos"})
    f.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(f, use_container_width=True)


# ------------------------------------------------------------ 3. Riesgo de fuego
def p_riesgo():
    cabecera("Riesgo de fuego (Frente 2)")
    d = c_pred(SEM, None, V)
    if vacio(d):
        return
    sd = (d.clase_riesgo == "Sin dato").sum()
    uf = st.selectbox("Estado", ["Todos"] + sorted(d.uf.unique()), key="uf_riesgo")
    x = d if uf == "Todos" else d[d.uf == uf]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Distribución de clases por estado (municipios)**")
        g = d.groupby(["uf", "clase_riesgo"]).size().reset_index(name="Municipios")
        f = px.bar(g, x="uf", y="Municipios", color="clase_riesgo", color_discrete_map=COLOR_CLASE,
                   category_orders={"clase_riesgo": ORDEN_CLASE}, labels={"uf": "Estado", "clase_riesgo": "Clase"})
        f.update_layout(height=380, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f, use_container_width=True)
    with c2:
        st.markdown("**Municipios con mayor probabilidad de clase Crítico**")
        t = x.dropna(subset=["prob_critico"]).sort_values("prob_critico", ascending=False).head(15)
        if vacio(t, "Sin probabilidades para esta selección."):
            pass
        else:
            f = px.bar(t.iloc[::-1], x="prob_critico", y=t.iloc[::-1].municipio_norm + " (" + t.iloc[::-1].uf + ")",
                       labels={"prob_critico": "Prob. crítico", "y": ""}, color_discrete_sequence=[ROJO])
            f.update_layout(height=380, margin=dict(t=10, b=10), xaxis_tickformat=".0%")
            st.plotly_chart(f, use_container_width=True)
    if sd:
        st.caption(f"{sd} municipios sin clase de riesgo disponible esta semana (se muestran como «Sin dato»).")
    st.markdown("**Clase y probabilidades por municipio**")
    t = x.sort_values("prob_critico", ascending=False)[["municipio_norm", "uf", "clase_riesgo", "prob_bajo", "prob_moderado",
                                                       "prob_alto", "prob_critico"]]
    t.columns = ["Municipio", "Estado", "Clase", "Bajo", "Moderado", "Alto", "Crítico"]
    pc = lambda v: f"color:{ROJO};font-weight:700" if v == "Crítico" else ""  # noqa: E731
    st.dataframe(t.style.map(pc, subset=["Clase"]).format({c: "{:.0%}" for c in ["Bajo", "Moderado", "Alto", "Crítico"]}, na_rep="Sin dato"),
                 hide_index=True, use_container_width=True, height=380)


# ------------------------------------------------------------ 4. Contexto
def p_ctx():
    cabecera("Contexto observado")
    c = c_ctx(SEM, V)
    if vacio(c, "No hay datos de contexto para esta semana."):
        return
    um = c_umbr_ctx(V)
    if not um:
        st.info("El servicio no entrega umbrales de contexto; se muestran solo los valores.")
        um = {}
    st.caption("Contexto: no entra al cálculo de alertas. Rojo solo para el nivel crítico; los demás niveles en tonos neutros.")

    def nivel3(v, a, r):
        if v is None or np.isnan(v):
            return "Sin dato"
        return "Sobre el umbral crítico" if v > r else "Sobre el umbral intermedio" if v > a else "Dentro de lo habitual"
    COL3 = {"Sin dato": "#E5E7EB", "Dentro de lo habitual": "#B8C4D0", "Sobre el umbral intermedio": "#5B7A99",
            "Sobre el umbral crítico": ROJO}
    t1, t2, t3 = st.tabs(["Riesgo de fuego INPE", "PM2.5", "Deforestación DETER"])
    with t1:
        u = um.get("riesgo_fuego")
        d = c.dropna(subset=["riesgo_fuego_promedio"]).sort_values("riesgo_fuego_promedio", ascending=False)
        if vacio(d, "Sin dato de riesgo de fuego INPE para esta semana."):
            pass
        else:
            if u:
                cz, cl = u["cortes"], u["clases"]
                d["Clase"] = [cl[int(np.searchsorted(cz, v, side="left"))] for v in d.riesgo_fuego_promedio]
                cmap = dict(zip(cl, ["#E5E7EB", "#B8C4D0", "#7F9BB8", "#3F6590", ROJO]))
                f = px.bar(d, x="uf", y="riesgo_fuego_promedio", color="Clase", color_discrete_map=cmap, category_orders={"Clase": cl})
                for z in cz:
                    f.add_hline(y=z, line_dash="dot", line_color=GRIS, line_width=1)
                st.caption("Líneas punteadas: límites de clase (" + ", ".join(fmt(z) for z in cz) + ").")
            else:
                f = px.bar(d, x="uf", y="riesgo_fuego_promedio", color_discrete_sequence=[AZUL])
            f.update_layout(height=360, margin=dict(t=10), xaxis_title="Estado", yaxis_title="Riesgo de fuego promedio")
            st.plotly_chart(f, use_container_width=True)
    with t2:
        d = c.dropna(subset=["pm2_5"]).sort_values("pm2_5", ascending=False)
        if vacio(d, "Sin dato de PM2.5 para esta semana."):
            pass
        else:
            u = um.get("pm2_5", {})
            if u:
                d["Nivel"] = [nivel3(v, u["ambar"], u["rojo"]) for v in d.pm2_5]
                f = px.bar(d, x="uf", y="pm2_5", color="Nivel", color_discrete_map=COL3)
                f.add_hline(y=u["ambar"], line_dash="dot", line_color=GRIS, annotation_text=f"Intermedio {fmt(u['ambar'])}")
                f.add_hline(y=u["rojo"], line_dash="dash", line_color=ROJO, annotation_text=f"Crítico {fmt(u['rojo'])}")
                st.caption(f"Umbrales ({u.get('unidad', '')}): {u.get('fuente', '')}.")
            else:
                f = px.bar(d, x="uf", y="pm2_5", color_discrete_sequence=[AZUL])
            f.update_layout(height=360, margin=dict(t=10), xaxis_title="Estado", yaxis_title="PM2.5 (µg/m³)")
            st.plotly_chart(f, use_container_width=True)
    with t3:
        d = c.dropna(subset=["deter_area_ha"]).sort_values("deter_area_ha", ascending=False)
        if vacio(d, "Sin dato DETER para esta semana."):
            pass
        else:
            u = um.get("deter", {})
            if u:
                d["Nivel"] = [nivel3(v, u["ambar"], u["rojo"]) for v in d.deter_area_ha]
                f = px.bar(d, x="uf", y="deter_area_ha", color="Nivel", color_discrete_map=COL3)
                f.add_hline(y=u["ambar"], line_dash="dot", line_color=GRIS, annotation_text=f"Intermedio {fmt(u['ambar'])}")
                f.add_hline(y=u["rojo"], line_dash="dash", line_color=ROJO, annotation_text=f"Crítico {fmt(u['rojo'])}")
                st.caption(f"Umbrales ({u.get('unidad', '')}): {u.get('fuente', '')}.")
            else:
                f = px.bar(d, x="uf", y="deter_area_ha", color_discrete_sequence=[AZUL])
            f.update_layout(height=360, margin=dict(t=10), xaxis_title="Estado", yaxis_title="Área deforestada (ha)")
            st.plotly_chart(f, use_container_width=True)


# ------------------------------------------------------------ 5. Control
def p_ctrl():
    cabecera("Control del modelo")
    k = c_ctrl(V)
    if vacio(k, "Aún no hay semanas con dato real para calcular los KPI de control."):
        return
    e = EST.get("estado")
    if e == "reentrenar":
        st.error(f"🔴 Reentrenar recomendado. {EST.get('mensaje', '')}")
    elif e == "vigilancia":
        st.warning(f"🟠 {EST.get('mensaje', '')}")
    else:
        st.success(f"🟢 {EST.get('mensaje', '')}")
    u = EST.get("umbrales", {})
    st.caption(f"Semanas fuera de umbral entre las últimas 4: {EST.get('semanas_fuera_umbral', 0)}.")
    series = [("mae_movil_4s", "Error medio móvil de 4 semanas (focos)", u.get("mae_movil_max"), "máximo", 1),
              ("recall_alerta", "Recall de alerta (%)", u.get("recall_alerta_min"), "mínimo", 100),
              ("falsas_alarmas", "Tasa de falsas alarmas (%)", u.get("falsas_alarmas_max"), "máximo", 100)]
    cols = st.columns(3)
    for c, (col, tit, umb, tipo, esc) in zip(cols, series):
        with c:
            f = go.Figure()
            f.add_scatter(x=k.semana_base, y=k[col] * esc, mode="lines", name="Valor", line=dict(color=AZUL, width=2))
            if umb is not None:
                f.add_hline(y=umb * esc, line_dash="dash", line_color=ROJO,
                            annotation_text=f"Umbral {tipo}: {fmt(umb * esc)}", annotation_position="top left")
            f.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), title=dict(text=tit, font=dict(size=14)), showlegend=False)
            st.plotly_chart(f, use_container_width=True)
    t = k.rename(columns={"semana_base": "Semana", "mae_movil_4s": "Error móvil 4 s", "recall_alerta": "Recall", "falsas_alarmas": "Falsas alarmas", "estado": "Estado"})
    t["Estado"] = t["Estado"].map({"dentro": "🟢 Dentro", "fuera": "🔴 Fuera de umbral"}).fillna(t["Estado"])
    with st.expander("Ver tabla de control"):
        st.dataframe(t.sort_values("Semana", ascending=False).style.format({"Error móvil 4 s": "{:.2f}", "Recall": "{:.1%}", "Falsas alarmas": "{:.1%}"}, na_rep="Sin dato"),
                     hide_index=True, use_container_width=True)


# ------------------------------------------------------------ 6. Ingesta
def p_ingesta():
    cabecera("Ingesta de una semana nueva")
    ej = semana_ejemplo(SV)
    if ej:
        d0 = pd.read_csv(pd.io.common.BytesIO(ej[1]))
        det = f" Debe traer las {d0.shape[1]} columnas del panel crudo y los {len(d0)} municipios."
    else:
        det = " Debe traer las columnas del panel crudo y todos los municipios."
    st.write("Suba un CSV con una sola semana (lunes, posterior a la última cargada)." + det +
             " Se valida y, si pasa, se calcula la proyección. Si falta una columna o un municipio, se rechaza.")
    if ej:
        st.download_button("Descargar semana de ejemplo", ej[1], file_name=ej[0], mime="text/csv",
                           help="Es la próxima semana. Súbala tal cual, o edítela (por ejemplo, borre una columna) para ver un rechazo.")
    else:
        st.info("No quedan más semanas de ejemplo para descargar.")
    st.caption(f"Última semana cargada: {V}")
    if "historial" not in st.session_state:
        st.session_state["historial"] = []
    st.session_state.setdefault("up_n", 0)
    up = st.file_uploader("Archivo CSV de la semana", type=["csv"], key=f"up{st.session_state['up_n']}")
    resultado = None
    if up is not None:
        b = up.getvalue()
        v = SV.validar_semana(b, up.name)
        st.markdown("**Resultado de la validación**")
        for m in v.get("mensajes", []):
            if m.startswith("ERROR"):
                st.error("✖ " + m)
            elif m.startswith("AVISO"):
                st.warning("⚠ " + m)
            else:
                st.success("✔ " + m.removeprefix("OK: ") if m.startswith("OK") else m)
        if v.get("ok"):
            if st.button("Calcular proyección", type="primary"):
                resultado = SV.ingestar_y_predecir(b, up.name)
                st.session_state["historial"].append({"archivo": up.name, **{k: resultado.get(k) for k in ("ok", "semana", "n_alertas")}})
                st.session_state["resultado_prev"] = resultado
                st.session_state["up_n"] += 1
                st.rerun()
        else:
            st.info("Corrija el archivo y vuélvalo a subir.")
    resultado = resultado or st.session_state.pop("resultado_prev", None)
    st.divider()
    if st.button("Simular siguiente semana (demostración)"):
        resultado = SV.simular_siguiente_semana()
        st.session_state["historial"].append({"archivo": "simulación", **{k: resultado.get(k) for k in ("ok", "semana", "n_alertas")}})
    if resultado:
        if resultado.get("ok"):
            st.success(resultado.get("mensaje", "Listo."))
            c_resumen.clear(); c_alertas.clear(); c_pred.clear(); c_serie.clear(); c_ctx.clear(); c_ctrl.clear(); c_estado.clear()
            al = SV.tabla_alertas(resultado["semana"])
            st.markdown(f"**Proyección de la semana {resultado['semana']}: municipios en alerta**")
            t = al[al.es_alerta].sort_values("prioridad")[["prioridad", "municipio_norm", "uf", "nivel_focos", "focos_7d", "focos_30d", "clase_riesgo"]]
            t.columns = ["Prioridad", "Municipio", "Estado", "Nivel de focos", "Focos 7 d", "Focos 30 d", "Clase de riesgo"]
            if not vacio(t, "Sin municipios en alerta en la semana nueva."):
                st.dataframe(t.style.format({"Focos 7 d": "{:.1f}", "Focos 30 d": "{:.1f}"}), hide_index=True, use_container_width=True, height=300)
            st.caption("Para verla en el resto del dashboard, elija esa semana en la barra lateral.")
        else:
            st.error(resultado.get("mensaje", "No se pudo procesar."))
    st.markdown("**Historial de ingestas**")
    h = SV.historial_ingestas() if hasattr(SV, "historial_ingestas") else pd.DataFrame(st.session_state["historial"])
    if vacio(h, "Aún no hay ingestas registradas."):
        return
    st.dataframe(h, hide_index=True, use_container_width=True)


# ------------------------------------------------------------ 7. Ayuda
def p_ayuda():
    cabecera("Ayuda y límites")
    st.markdown("""
### Cómo leer los colores
Cada color va siempre con su texto o ícono.

| Estado | Significa | Qué hacer |
|---|---|---|
| 🟢 **Verde** | El indicador está dentro de lo esperado. | Seguir con la rutina semanal. |
| 🟠 **Ámbar** | Cerca del límite. | Vigilar la semana siguiente y preparar refuerzos. |
| 🔴 **Rojo** | Fuera del umbral. | Actuar: atender por prioridad y, si es un KPI de control, evaluar reentrenar. |
| ⚪ **Sin dato** | No hay dato real para calcular. | Esperar el dato observado; la proyección sigue siendo válida. |

El rojo se reserva a los niveles **Muy elevado** y **Crítico** y a los KPI en rojo. El resto va en tonos azules y grises.

### Estado del modelo
- **Estable**: las últimas 4 semanas dentro de umbral.
- **Vigilancia**: una semana fuera de umbral.
- **Reentrenar recomendado**: dos o más semanas fuera de umbral.

### Límites declarados
- Los datos de 2026 son **simulados** con fuentes públicas (INPE, NASA POWER, SISAM).
- El SISAM de 2026 es un **pronóstico sin calibrar**.
- La **capacidad de brigadas** es un supuesto; no existe dato oficial.
- El **riesgo de fuego INPE** está derivado del clima.
- Los umbrales de los KPI salen de validación con datos 2022 a 2024 y se comprueban con 2025; ver «¿De dónde sale este umbral?» en el panel.
""")
    with st.expander("Parámetros del modelo y del servicio"):
        st.write("Servicio en uso: " + SV.__name__)


{"Panel de control": p_panel, "Proyección": p_proy, "Riesgo de fuego": p_riesgo, "Contexto": p_ctx,
 "Control del modelo": p_ctrl, "Ingesta": p_ingesta, "Ayuda y límites": p_ayuda}[pagina]()
