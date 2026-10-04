"""
BIODIVERSITY-GUARD-Predict: Dashboard preliminar (Hito 2)

Tres vistas, cada una con sus propios filtros:
  Frente 1: Hectáreas en riesgo (regresión; unidad: semana y municipio)
  Frente 2: Alertas por nivel de riesgo (clasificación; unidad: alerta individual)
  Contexto observado: riesgo de fuego (INPE) y calidad del aire (PM2.5)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="BIODIVERSITY-GUARD-Predict",
    page_icon="🌳",
    layout="wide",
)

# ------------------------------------------------------------------ Parámetros
# Umbrales PROPIOS iniciales (editables en la barra lateral de cada vista)
F1_ELEVADO, F1_MUY_ELEVADO = 45.0, 163.0   # ha por municipio y semana (EDA: mediana y P75)
F2_ELEVADO, F2_MUY_ELEVADO = 49.0, 86.5    # alertas Alto o Crítico por semana (156 semanas
                                           # completas 2022-2024: mediana y P75)

# Umbrales OFICIALES (fijos)
FUEGO_CLASES = ["Mínimo", "Bajo", "Medio", "Alto", "Crítico"]
FUEGO_LIMITES = [0.15, 0.40, 0.70, 0.95]   # INPE, cada límite pertenece a la clase inferior
PM_OMS = 15.0        # guía OMS 2021, media de 24 h (µg/m³)
PM_NACIONAL = 60.0   # PI-1, Resolución CONAMA 506/2024, vigente hasta el 31-dic-2024

PRIORITARIOS = ["SAO FELIX DO XINGU", "ALTAMIRA", "NOVO PROGRESSO"]
TOP_N = 15

NIVELES = ["Habitual", "Elevado", "Muy elevado"]
MD_COLOR = {"Habitual": "blue", "Elevado": "orange", "Muy elevado": "red"}
COLOR_NIVEL = {"Habitual": "#6BAED6", "Elevado": "#FD8D3C", "Muy elevado": "#A50F15"}

CLASES = ["Bajo", "Moderado", "Alto", "Crítico"]
COLOR_CLASE = {"Bajo": "#C6DBEF", "Moderado": "#6BAED6", "Alto": "#FD8D3C", "Crítico": "#A50F15"}
GRAVES = ["Alto", "Crítico"]

COLOR_FUEGO = {"Mínimo": "#C6DBEF", "Bajo": "#9ECAE1", "Medio": "#FDAE6B",
               "Alto": "#E6550D", "Crítico": "#7F0000"}
PM_NIVELES = ["Dentro de la guía OMS", "Sobre la guía OMS", "Sobre el estándar nacional"]
COLOR_PM = {"Dentro de la guía OMS": "#C6DBEF", "Sobre la guía OMS": "#FD8D3C",
            "Sobre el estándar nacional": "#A50F15"}
COLOR_SIN_DATO = "#BDBDBD"

COLOR_PRED, COLOR_OBS = "#E69F00", "#0072B2"
DATA_DIR = Path(__file__).parent / "datos"


# ---------------------------------------------------------------------- Utilidades
def nivel(valor, elevado, muy_elevado):
    """Habitual si valor <= elevado; Elevado si <= muy_elevado; Muy elevado si lo supera."""
    if valor > muy_elevado:
        return "Muy elevado"
    if valor > elevado:
        return "Elevado"
    return "Habitual"


def clase_fuego(valor):
    """Clase oficial de INPE: Mínimo <= 0,15 < Bajo <= 0,40 < Medio <= 0,70 < Alto <= 0,95 < Crítico."""
    return FUEGO_CLASES[int(np.searchsorted(FUEGO_LIMITES, valor, side="left"))]


def nivel_pm(dias_oms, dias_nacional):
    """Nivel semanal de PM2.5 según los días que superan cada umbral de 24 horas."""
    if dias_nacional >= 1:
        return "Sobre el estándar nacional"
    if dias_oms >= 1:
        return "Sobre la guía OMS"
    return "Dentro de la guía OMS"


def etiqueta_nivel(n):
    """Nombre del nivel con color de refuerzo (el significado lo da el nombre)."""
    return f":{MD_COLOR[n]}[**{n}**]"


def mostrar_leyenda(unidad, elevado, muy, origen):
    """Tabla fija con el nombre, el rango y el significado de cada nivel."""
    tabla = pd.DataFrame(
        {
            "Nivel": NIVELES,
            "Rango": [
                f"hasta {elevado:g} {unidad}",
                f"más de {elevado:g} hasta {muy:g} {unidad}",
                f"más de {muy:g} {unidad}",
            ],
            "Significado respecto al historial": [
                "En la mitad más baja del historial",
                "Por encima de la mediana, sin llegar al 25 % más alto",
                "Dentro del 25 % más alto del historial",
            ],
            "Acción sugerida (provisional)": [
                "Monitoreo regular",
                "Vigilancia reforzada",
                "Priorizar para inspección o respuesta",
            ],
        }
    ).set_index("Nivel")
    st.table(tabla)
    st.caption(
        f"{origen} Las acciones son provisionales y deben validarse con BIODIVERSITY-GUARD. "
        "Si modificas los umbrales, el significado respecto al historial deja de aplicar. "
        "Solo se consideran semanas completas (lunes a domingo)."
    )


def semanas_completas(semanas, fecha_fin_datos, inicio=None, fin=None):
    """Semanas (lunes) con sus 7 días dentro de los datos y, si se indica, del rango."""
    limite = fecha_fin_datos if fin is None else min(fecha_fin_datos, fin)
    salida = []
    for s in semanas:
        s = pd.Timestamp(s)
        if s + pd.Timedelta(days=6) > limite:
            continue
        if inicio is not None and s < inicio:
            continue
        salida.append(s)
    return salida


def marca_semana(fig, semana):
    """Línea vertical punteada en la semana de análisis."""
    fig.add_shape(
        type="line", xref="x", yref="paper", x0=semana, x1=semana, y0=0, y1=1,
        line=dict(color="gray", dash="dot", width=1.5),
    )


def grafico_ranking(df, col_pred, col_obs, orden, titulo, eje_x, formato):
    """Ranking horizontal de municipios (predicho vs. observado). ★ = prioritario."""
    columna = col_pred if orden == "Predicho" else col_obs
    base = df.sort_values(columna, ascending=False).head(TOP_N).iloc[::-1]
    etiquetas = [
        ("★ " if m in PRIORITARIOS else "") + f"{m} ({u})"
        for m, u in zip(base["municipio_norm"], base["uf"])
    ]
    fig = go.Figure()
    fig.add_bar(
        name="Predicho", y=etiquetas, x=base[col_pred], orientation="h",
        marker_color=COLOR_PRED, text=[formato.format(v) for v in base[col_pred]],
        textposition="outside", cliponaxis=False,
    )
    fig.add_bar(
        name="Observado", y=etiquetas, x=base[col_obs], orientation="h",
        marker_color=COLOR_OBS, text=[formato.format(v) for v in base[col_obs]],
        textposition="outside", cliponaxis=False,
    )
    maximo = float(max(base[col_pred].max(), base[col_obs].max()))
    fig.update_layout(
        barmode="group", title=titulo, xaxis_title=eje_x, height=560,
        margin=dict(t=50, b=20, r=30), legend=dict(orientation="h", y=-0.12),
        xaxis_range=[0, max(maximo, 1) * 1.22],
    )
    return fig


def construir_mapa(df, col_categoria, categorias, colores, hover, titulo, tamano, altura):
    """Mapa con un trazo por categoría, para que la leyenda muestre los nombres."""
    fig = go.Figure()
    for c in categorias:
        d = df[df[col_categoria] == c]
        if d.empty:
            continue
        fig.add_trace(
            go.Scattermap(
                lat=d["lat_real"], lon=d["lon_real"], mode="markers", name=c,
                marker=dict(size=tamano, color=colores[c], opacity=0.8),
                text=hover[d.index], hoverinfo="text",
            )
        )
    lat, lon = df["lat_real"], df["lon_real"]
    amplitud = max(lat.max() - lat.min(), (lon.max() - lon.min()) * 0.6)
    zoom = 3.2 if amplitud > 14 else 4 if amplitud > 8 else 5 if amplitud > 3 else 7
    fig.update_layout(
        title=titulo, height=altura, margin=dict(l=0, r=0, t=50, b=0),
        map=dict(style="open-street-map", zoom=zoom,
                 center=dict(lat=float(lat.mean()), lon=float(lon.mean()))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def firma_archivos():
    """Nombre y fecha de modificación de los CSV: renueva la caché si cambian."""
    return tuple((a.name, a.stat().st_mtime) for a in sorted(DATA_DIR.glob("*.csv")))


@st.cache_data
def cargar_datos(firma):
    """Lee los CSV de predicciones (julio a diciembre de 2024) y el contexto, si existe."""
    f1 = pd.read_csv(DATA_DIR / "predicciones_frente1.csv", parse_dates=["semana"])
    f2 = pd.read_csv(DATA_DIR / "predicciones_frente2.csv", parse_dates=["view_date"])
    f2["semana"] = f2["view_date"].dt.to_period("W").dt.start_time
    ruta = DATA_DIR / "contexto_semanal.csv"
    ctx = pd.read_csv(ruta, parse_dates=["semana"]) if ruta.exists() else None
    return f1, f2, ctx


# =========================================================================== FRENTE 1
def vista_frente1(f1, fecha_fin_datos):
    st.header("Frente 1: Hectáreas en riesgo")
    st.markdown(
        "**Pregunta:** ¿cuántas hectáreas se espera afectar esta semana y qué tan fuera "
        "de lo habitual es en cada municipio?"
    )

    # ---- filtros y umbrales de esta vista
    st.sidebar.header("Filtros")
    municipios = sorted(f1["municipio_norm"].unique())
    seleccion = st.sidebar.multiselect(
        "Municipios", municipios, default=[], key="f1_municipios",
        help="Si no eliges ninguno, se muestran todos los municipios.",
    )
    semanas = semanas_completas(sorted(f1["semana"].unique()), fecha_fin_datos)
    if not semanas:
        st.warning("No hay semanas completas en los datos.")
        return
    semana_sel = st.sidebar.selectbox(
        "Semana de análisis", semanas, index=len(semanas) - 1, key="f1_semana",
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        help="Lunes de inicio de la semana.",
    )
    st.sidebar.header("Umbrales (editables)")
    elevado = st.sidebar.number_input(
        "Inicio de Elevado (ha)", min_value=0.0, value=F1_ELEVADO, step=5.0, key="f1_elevado"
    )
    muy = st.sidebar.number_input(
        "Inicio de Muy elevado (ha)", min_value=0.0, value=F1_MUY_ELEVADO, step=5.0, key="f1_muy"
    )
    if muy <= elevado:
        st.sidebar.error("El umbral de Muy elevado debe ser mayor que el de Elevado.")
        return

    mostrar_leyenda(
        "ha por municipio", elevado, muy,
        "Umbrales propios del proyecto: mediana y percentil 75 de las hectáreas semanales "
        "por municipio (EDA, 2022-2024).",
    )
    with st.expander("¿Por qué estos umbrales? Referencias oficiales analizadas"):
        st.table(
            pd.DataFrame(
                {
                    "Referencia": ["6.25 ha", "25 ha", "20 ha"],
                    "Qué mide": [
                        "Área mínima de los polígonos que INPE publica y que PRODES cuenta "
                        "como deforestación",
                        "Tamaño de parche desde el cual el DETER de baja resolución se "
                        "considera preciso",
                        "Corte de deforestación de gran escala en un estudio sobre la "
                        "fiscalización de IBAMA",
                    ],
                    "Fuente": ["INPE", "Börner et al. (2015)", "Börner et al. (2015)"],
                }
            ).set_index("Referencia")
        )
        st.caption(
            "Son umbrales de detección por parche, no de alerta semanal por municipio; "
            "por eso el proyecto usa la mediana y el percentil 75 de su propio historial."
        )

    f1_sel = f1[f1["municipio_norm"].isin(seleccion)] if seleccion else f1
    sem1 = f1_sel[f1_sel["semana"] == semana_sel].copy()

    # ---- serie semanal (usa todas las semanas completas)
    serie = (
        f1_sel[f1_sel["semana"].isin(semanas)]
        .groupby("semana")[["area_ha_total_predicha", "area_ha_total"]].sum()
        .reindex(semanas).fillna(0)
    )
    fig_serie = go.Figure()
    fig_serie.add_scatter(x=serie.index, y=serie["area_ha_total_predicha"], name="Predicho",
                          mode="lines+markers", line=dict(color=COLOR_PRED))
    fig_serie.add_scatter(x=serie.index, y=serie["area_ha_total"], name="Observado",
                          mode="lines+markers", line=dict(color=COLOR_OBS))
    marca_semana(fig_serie, semana_sel)
    fig_serie.update_layout(
        title="Hectáreas semanales (suma de los municipios elegidos)",
        yaxis_title="Hectáreas", height=360, margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
    )

    st.subheader(f"Semana del {pd.Timestamp(semana_sel).strftime('%Y-%m-%d')}")
    if sem1.empty:
        st.info("Los municipios elegidos no tienen alertas registradas en esta semana.")
        st.plotly_chart(fig_serie, key="f1_serie")
        return

    sem1["nivel_pred"] = sem1["area_ha_total_predicha"].apply(lambda v: nivel(v, elevado, muy))
    sem1["nivel_obs"] = sem1["area_ha_total"].apply(lambda v: nivel(v, elevado, muy))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hectáreas (predicho)", f"{sem1['area_ha_total_predicha'].sum():,.0f} ha")
    m2.metric("Hectáreas (observado)", f"{sem1['area_ha_total'].sum():,.0f} ha")
    m3.metric("Municipios en Muy elevado (predicho)", int((sem1["nivel_pred"] == "Muy elevado").sum()))
    m4.metric("Municipios en Muy elevado (observado)", int((sem1["nivel_obs"] == "Muy elevado").sum()))
    st.caption(f"{len(sem1)} municipios con alertas en la semana. El nivel se asigna a cada municipio.")

    # ---- gráfico 1: municipios por nivel
    conteo = pd.DataFrame(
        {"Predicho": sem1["nivel_pred"].value_counts(), "Observado": sem1["nivel_obs"].value_counts()}
    ).reindex(NIVELES).fillna(0).astype(int)
    fig_niv = go.Figure()
    for nombre, color in (("Predicho", COLOR_PRED), ("Observado", COLOR_OBS)):
        fig_niv.add_bar(name=nombre, x=NIVELES, y=conteo[nombre], marker_color=color,
                        text=conteo[nombre], textposition="outside", cliponaxis=False)
    fig_niv.update_layout(
        barmode="group", title="Municipios por nivel: predicho vs. observado",
        yaxis_title="Número de municipios", height=360, margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        yaxis_range=[0, max(int(conteo.values.max()), 1) * 1.2],
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_niv, key="f1_niveles")
    c2.plotly_chart(fig_serie, key="f1_serie")

    # ---- ranking y mapa
    c3, c4 = st.columns(2)
    with c3:
        orden = st.radio("Ordenar el ranking por", ["Observado", "Predicho"],
                         horizontal=True, key="f1_orden")
        st.plotly_chart(
            grafico_ranking(sem1, "area_ha_total_predicha", "area_ha_total", orden,
                            f"Top {TOP_N} municipios de la semana", "Hectáreas", "{:,.0f}"),
            key="f1_ranking",
        )
        st.caption("★ = municipio prioritario (arco de deforestación).")
    with c4:
        sem1 = sem1.reset_index(drop=True)
        hover = (
            sem1["municipio_norm"] + " (" + sem1["uf"] + ") | Predicho: "
            + sem1["area_ha_total_predicha"].round(0).astype(int).astype(str)
            + " ha (" + sem1["nivel_pred"] + ") | Observado: "
            + sem1["area_ha_total"].round(0).astype(int).astype(str) + " ha (" + sem1["nivel_obs"] + ")"
        )
        st.write("")
        st.write("")
        st.plotly_chart(
            construir_mapa(sem1, "nivel_pred", NIVELES, COLOR_NIVEL, hover,
                           "Municipios por nivel predicho", 12, 560),
            key="f1_mapa",
        )
        st.caption("Posición: centroide de las alertas del municipio. Clic en la leyenda para ocultar un nivel.")


# =========================================================================== FRENTE 2
def vista_frente2(f2, fecha_fin_datos):
    st.header("Frente 2: Alertas por nivel de riesgo")
    st.markdown(
        "**Pregunta:** ¿cuántas alertas graves (Alto o Crítico) hay esta semana y cómo se "
        "distribuyen las alertas por nivel de riesgo?"
    )

    # ---- filtros y umbrales de esta vista (sin filtro de municipio)
    st.sidebar.header("Filtros")
    f_min, f_max = f2["view_date"].min().date(), f2["view_date"].max().date()
    rango = st.sidebar.date_input(
        "Rango de fechas", value=(f_min, f_max), min_value=f_min, max_value=f_max, key="f2_rango"
    )
    if len(rango) != 2:
        st.info("Selecciona la fecha de inicio y la de fin para continuar.")
        return
    inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])

    calendario = list(pd.date_range(f2["semana"].min(), f2["semana"].max(), freq="7D"))
    semanas = semanas_completas(calendario, fecha_fin_datos, inicio, fin)

    st.sidebar.header("Umbrales (editables)")
    elevado = st.sidebar.number_input(
        "Inicio de Elevado (alertas por semana)", min_value=0.0, value=F2_ELEVADO, step=1.0,
        key="f2_elevado",
    )
    muy = st.sidebar.number_input(
        "Inicio de Muy elevado (alertas por semana)", min_value=0.0, value=F2_MUY_ELEVADO,
        step=1.0, key="f2_muy",
    )
    if muy <= elevado:
        st.sidebar.error("El umbral de Muy elevado debe ser mayor que el de Elevado.")
        return

    mostrar_leyenda(
        "alertas por semana", elevado, muy,
        "Los niveles se aplican al número de alertas Alto o Crítico de la semana. Umbrales "
        "propios del proyecto: mediana y percentil 75 de esa serie (156 semanas completas, "
        "2022-2024).",
    )
    st.caption(
        "Los umbrales no distinguen entre temporada seca y húmeda: en los meses de quema es "
        "esperable ver con frecuencia el nivel Muy elevado."
    )

    f2_per = f2[(f2["view_date"] >= inicio) & (f2["view_date"] <= fin)].copy()
    if f2_per.empty:
        st.info("No hay alertas en el rango elegido.")
        return
    f2_per["grave_obs"] = f2_per["Nivel_Riesgo_Amenaza"].isin(GRAVES)
    f2_per["grave_pred"] = f2_per["Nivel_Riesgo_Amenaza_predicho"].isin(GRAVES)

    # ---- KPI semanal y serie (semanas completas dentro del rango)
    st.subheader("Alertas graves de la semana")
    fig_serie = None
    if not semanas:
        st.warning("El rango elegido no contiene ninguna semana completa (lunes a domingo).")
    else:
        por_semana = (
            f2_per.groupby("semana")[["grave_pred", "grave_obs"]].sum()
            .reindex(semanas).fillna(0).astype(int)
        )
        totales = f2_per.groupby("semana").size().reindex(semanas).fillna(0).astype(int)
        semana_sel = st.sidebar.selectbox(
            "Semana de análisis", semanas, index=len(semanas) - 1, key="f2_semana",
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
            help="Semanas completas dentro del rango de fechas.",
        )
        n_pred = int(por_semana.loc[semana_sel, "grave_pred"])
        n_obs = int(por_semana.loc[semana_sel, "grave_obs"])

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Alto o Crítico (predicho)", f"{n_pred:,}")
            st.markdown(f"Nivel: {etiqueta_nivel(nivel(n_pred, elevado, muy))}")
        with k2:
            st.metric("Alto o Crítico (observado)", f"{n_obs:,}")
            st.markdown(f"Nivel: {etiqueta_nivel(nivel(n_obs, elevado, muy))}")
        with k3:
            st.metric("Alertas totales de la semana", f"{int(totales.loc[semana_sel]):,}")

        extremos = [por_semana.index[0], por_semana.index[-1]]
        fig_serie = go.Figure()
        fig_serie.add_scatter(x=por_semana.index, y=por_semana["grave_pred"], name="Predicho",
                              mode="lines+markers", line=dict(color=COLOR_PRED))
        fig_serie.add_scatter(x=por_semana.index, y=por_semana["grave_obs"], name="Observado",
                              mode="lines+markers", line=dict(color=COLOR_OBS))
        fig_serie.add_scatter(x=extremos, y=[elevado, elevado], mode="lines",
                              name=f"Umbral Elevado: más de {elevado:g}",
                              line=dict(color="gray", dash="dash"))
        fig_serie.add_scatter(x=extremos, y=[muy, muy], mode="lines",
                              name=f"Umbral Muy elevado: más de {muy:g}",
                              line=dict(color="#A50F15", dash="dash"))
        marca_semana(fig_serie, semana_sel)
        fig_serie.update_layout(
            title="Alertas Alto o Crítico por semana", yaxis_title="Número de alertas",
            height=400, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.3),
        )

    st.subheader("Distribución de las alertas en el periodo elegido")
    st.caption(
        f"{len(f2_per):,} alertas entre {inicio.strftime('%Y-%m-%d')} y {fin.strftime('%Y-%m-%d')}. "
        "Estos gráficos usan todas las alertas del rango, no solo las semanas completas."
    )

    # ---- gráfico: distribución de los 4 niveles
    cp = f2_per["Nivel_Riesgo_Amenaza_predicho"].value_counts().reindex(CLASES).fillna(0).astype(int)
    co = f2_per["Nivel_Riesgo_Amenaza"].value_counts().reindex(CLASES).fillna(0).astype(int)
    total = len(f2_per)
    fig_dist = go.Figure()
    for nombre, serie_c, color in (("Predicho", cp, COLOR_PRED), ("Observado", co, COLOR_OBS)):
        fig_dist.add_bar(
            name=nombre, x=CLASES, y=serie_c, marker_color=color,
            text=[f"{v:,}<br>({v / total * 100:.1f}%)" for v in serie_c],
            textposition="outside", textfont=dict(size=11), cliponaxis=False,
        )
    fig_dist.update_layout(
        barmode="group", title="Alertas por nivel de riesgo: predicho vs. observado",
        yaxis_title="Número de alertas", height=400, margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        yaxis_range=[0, max(int(cp.max()), int(co.max()), 1) * 1.25],
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_dist, key="f2_distribucion")
    if fig_serie is not None:
        c2.plotly_chart(fig_serie, key="f2_serie")

    # ---- ranking y mapa
    rk = (
        f2_per.groupby(["municipio_norm", "uf"])[["grave_pred", "grave_obs"]]
        .sum().astype(int).reset_index()
    )
    c3, c4 = st.columns(2)
    with c3:
        orden = st.radio("Ordenar el ranking por", ["Observado", "Predicho"],
                         horizontal=True, key="f2_orden")
        st.plotly_chart(
            grafico_ranking(rk, "grave_pred", "grave_obs", orden,
                            f"Top {TOP_N} municipios por alertas Alto o Crítico",
                            "Número de alertas", "{:,.0f}"),
            key="f2_ranking",
        )
        st.caption("★ = municipio prioritario (arco de deforestación).")
    with c4:
        colorear = st.radio("Colorear el mapa por", ["Predicho", "Observado"],
                            horizontal=True, key="f2_color")
        col_cat = "Nivel_Riesgo_Amenaza_predicho" if colorear == "Predicho" else "Nivel_Riesgo_Amenaza"
        mapa_df = f2_per.reset_index(drop=True)
        hover = (
            mapa_df["municipio_norm"] + " (" + mapa_df["uf"] + ") | "
            + mapa_df["view_date"].dt.strftime("%Y-%m-%d")
            + " | Observado: " + mapa_df["Nivel_Riesgo_Amenaza"]
            + " | Predicho: " + mapa_df["Nivel_Riesgo_Amenaza_predicho"]
        )
        st.plotly_chart(
            construir_mapa(mapa_df, col_cat, CLASES, COLOR_CLASE, hover,
                           f"Alertas del periodo, coloreadas por nivel {colorear.lower()}", 5, 560),
            key="f2_mapa",
        )
        st.caption("Clic en la leyenda para ocultar un nivel y ver mejor los demás.")

    # ---- referencia técnica: calidad del modelo (cerrada por defecto)
    with st.expander("Calidad del modelo (referencia técnica)"):
        st.caption(
            "Esta matriz evalúa el modelo, no es un indicador de negocio. Sirve para entender "
            "qué tan bien identifica cada nivel."
        )
        cm = pd.crosstab(
            f2_per["Nivel_Riesgo_Amenaza"], f2_per["Nivel_Riesgo_Amenaza_predicho"]
        ).reindex(index=CLASES, columns=CLASES, fill_value=0)
        pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fig_cm = go.Figure(
            go.Heatmap(
                z=pct.values, x=CLASES, y=CLASES,
                text=[[f"{v:,}" for v in fila] for fila in cm.values],
                texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
                colorbar=dict(title="% de la fila"),
                hovertemplate=("Observado: %{y}<br>Predicho: %{x}<br>"
                               "% de la fila: %{z:.1f}%<br>Alertas: %{text}<extra></extra>"),
            )
        )
        fig_cm.update_layout(
            title="Matriz de confusión del periodo", xaxis_title="Predicho",
            yaxis_title="Observado", yaxis_autorange="reversed",
            height=480, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_cm, key="f2_matriz")
        st.caption(
            "Cada fila suma 100 %: la diagonal muestra qué porcentaje de las alertas "
            "observadas de cada nivel fue identificado correctamente."
        )


# ===================================================================== CONTEXTO OBSERVADO
def vista_contexto(f1, ctx, fecha_fin_datos):
    st.header("Contexto observado: fuego y calidad del aire")
    st.markdown(
        "**Pregunta:** ¿qué condiciones ambientales acompañan esta semana en cada municipio? "
        "Son datos medidos, no predicciones del modelo."
    )
    if ctx is None:
        st.info(
            "Para ver esta vista, agrega el archivo datos/contexto_semanal.csv "
            "(riesgo de fuego y PM2.5 por semana y municipio)."
        )
        return

    f1c = f1.merge(ctx, on=["semana", "municipio_norm", "uf"], how="left")
    if len(f1c) != len(f1):
        st.error(
            "El archivo contexto_semanal.csv duplica filas al unirse con las predicciones. "
            "Revisa que tenga una sola fila por semana, municipio y estado."
        )
        return

    # ---- filtros de esta vista
    st.sidebar.header("Filtros")
    indicador = st.sidebar.radio(
        "Indicador", ["Riesgo de fuego (INPE)", "Calidad del aire (PM2.5)"], key="ctx_indicador"
    )
    municipios = sorted(f1c["municipio_norm"].unique())
    seleccion = st.sidebar.multiselect(
        "Municipios", municipios, default=[], key="ctx_municipios",
        help="Si no eliges ninguno, se muestran todos los municipios.",
    )
    semanas = semanas_completas(sorted(f1c["semana"].unique()), fecha_fin_datos)
    if not semanas:
        st.warning("No hay semanas completas en los datos.")
        return
    semana_sel = st.sidebar.selectbox(
        "Semana de análisis", semanas, index=len(semanas) - 1, key="ctx_semana",
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        help="Lunes de inicio de la semana.",
    )
    st.sidebar.caption("Los umbrales de esta vista son oficiales y por eso no son editables.")

    base = f1c[f1c["municipio_norm"].isin(seleccion)] if seleccion else f1c
    sem = base[base["semana"] == semana_sel].copy()
    en_rango = base[base["semana"].isin(semanas)].copy()
    st.subheader(f"Semana del {pd.Timestamp(semana_sel).strftime('%Y-%m-%d')}")
    if sem.empty:
        st.info("Los municipios elegidos no tienen registros en esta semana.")
        return

    if indicador.startswith("Riesgo"):
        _contexto_fuego(sem, en_rango, semanas, semana_sel)
    else:
        _contexto_pm(sem, en_rango, semanas, semana_sel)


def _serie_lineas(serie, columnas, colores, titulo, eje_y, semana_sel):
    fig = go.Figure()
    for c in columnas:
        fig.add_scatter(x=serie.index, y=serie[c], name=c, mode="lines+markers",
                        line=dict(color=colores[c]))
    marca_semana(fig, semana_sel)
    fig.update_layout(title=titulo, yaxis_title=eje_y, height=380, margin=dict(t=50, b=20),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _contexto_fuego(sem, en_rango, semanas, semana_sel):
    st.table(
        pd.DataFrame(
            {
                "Clase (INPE)": [f"Riesgo de fuego: {c}" for c in FUEGO_CLASES],
                "Rango del riesgo de fuego": [
                    "hasta 0,15", "más de 0,15 hasta 0,40", "más de 0,40 hasta 0,70",
                    "más de 0,70 hasta 0,95", "más de 0,95",
                ],
            }
        ).set_index("Clase (INPE)")
    )
    st.caption(
        "Escala oficial de INPE. El riesgo de fuego estima la susceptibilidad de la vegetación "
        "a quemarse según los días sin lluvia, la humedad y la temperatura; no predice la "
        "actividad humana. El dato es el promedio semanal del riesgo previsto en los focos de "
        "calor detectados en el municipio, por lo que los municipios sin focos no tienen dato. "
        "INPE documenta que, en su producto de riesgo observado, eleva a Alto el riesgo donde "
        "se detecta un foco; no puede descartarse un sesgo al alza del promedio."
    )

    con = sem.dropna(subset=["riesgo_fuego_promedio"]).copy()
    n_sin = len(sem) - len(con)
    con["clase"] = con["riesgo_fuego_promedio"].apply(clase_fuego)

    m1, m2, m3 = st.columns(3)
    m1.metric("Municipios con dato", f"{len(con)} de {len(sem)}")
    m2.metric("Municipios sin focos (sin dato)", n_sin)
    m3.metric("Riesgo de fuego en Alto o Crítico", int(con["clase"].isin(["Alto", "Crítico"]).sum()))

    if con.empty:
        st.info("Ningún municipio de la selección tiene focos de calor detectados en esta semana.")
        return

    conteo = con["clase"].value_counts().reindex(FUEGO_CLASES).fillna(0).astype(int)
    etiquetas = FUEGO_CLASES + ["Sin dato"]
    valores = list(conteo.values) + [n_sin]
    colores = [COLOR_FUEGO[c] for c in FUEGO_CLASES] + [COLOR_SIN_DATO]
    fig_cl = go.Figure(
        go.Bar(x=etiquetas, y=valores, marker_color=colores, text=valores,
               textposition="outside", cliponaxis=False)
    )
    fig_cl.update_layout(
        title="Riesgo de fuego (INPE): municipios por clase", yaxis_title="Número de municipios",
        height=380, margin=dict(t=50, b=20), showlegend=False,
        yaxis_range=[0, max(max(valores), 1) * 1.2],
    )

    todas = en_rango.dropna(subset=["riesgo_fuego_promedio"]).copy()
    todas["clase"] = todas["riesgo_fuego_promedio"].apply(clase_fuego)
    serie = (
        todas.groupby(["semana", "clase"]).size().unstack(fill_value=0)
        .reindex(index=semanas, columns=FUEGO_CLASES, fill_value=0)
    )
    fig_se = _serie_lineas(serie, ["Alto", "Crítico"], COLOR_FUEGO,
                           "Municipios con riesgo de fuego Alto o Crítico por semana",
                           "Número de municipios", semana_sel)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_cl, key="ctx_fuego_clases")
    c2.plotly_chart(fig_se, key="ctx_fuego_serie")

    mp = con.reset_index(drop=True)
    hover = (
        mp["municipio_norm"] + " (" + mp["uf"] + ") | Riesgo de fuego: "
        + mp["riesgo_fuego_promedio"].round(2).astype(str) + " (" + mp["clase"] + ")"
    )
    st.plotly_chart(
        construir_mapa(mp, "clase", FUEGO_CLASES, COLOR_FUEGO, hover,
                       "Municipios por clase de riesgo de fuego", 12, 520),
        key="ctx_fuego_mapa",
    )
    st.caption("Posición: centroide de las alertas del municipio. Clic en la leyenda para ocultar una clase.")


def _contexto_pm(sem, en_rango, semanas, semana_sel):
    st.table(
        pd.DataFrame(
            {
                "Nivel": PM_NIVELES,
                "Criterio (PM2.5 diario)": [
                    f"Ningún día de la semana sobre {PM_OMS:g} µg/m³",
                    f"Al menos un día sobre {PM_OMS:g} µg/m³ y ninguno sobre {PM_NACIONAL:g} µg/m³",
                    f"Al menos un día sobre {PM_NACIONAL:g} µg/m³",
                ],
                "Referencia": [
                    "Guía OMS 2021 (media de 24 h)",
                    "Guía OMS 2021 (media de 24 h)",
                    "Estándar PI-1, Resolución CONAMA 506/2024",
                ],
            }
        ).set_index("Nivel")
    )
    st.caption(
        "Los valores diarios provienen de un reanálisis (modelo), no de estaciones de monitoreo. "
        "El estándar nacional PI-1 estuvo vigente hasta el 31 de diciembre de 2024; desde 2025 "
        "rige PI-2 (50 µg/m³). La guía OMS es más estricta que el estándar nacional, por lo que "
        "superarla es frecuente en temporada de quema."
    )

    sem["nivel_pm"] = [
        nivel_pm(a, b) for a, b in zip(sem["pm25_dias_sobre_oms"], sem["pm25_dias_sobre_nacional"])
    ]
    conteo = sem["nivel_pm"].value_counts().reindex(PM_NIVELES).fillna(0).astype(int)

    m1, m2, m3 = st.columns(3)
    for col, n in zip((m1, m2, m3), PM_NIVELES):
        col.metric(f"Municipios {n[0].lower() + n[1:]}", int(conteo[n]))

    fig_cl = go.Figure(
        go.Bar(x=PM_NIVELES, y=conteo.values, marker_color=[COLOR_PM[n] for n in PM_NIVELES],
               text=conteo.values, textposition="outside", cliponaxis=False)
    )
    fig_cl.update_layout(
        title="PM2.5: municipios por nivel", yaxis_title="Número de municipios",
        height=380, margin=dict(t=50, b=20), showlegend=False,
        yaxis_range=[0, max(int(conteo.max()), 1) * 1.2],
    )

    todas = en_rango.copy()
    todas["nivel_pm"] = [
        nivel_pm(a, b) for a, b in zip(todas["pm25_dias_sobre_oms"], todas["pm25_dias_sobre_nacional"])
    ]
    serie = (
        todas.groupby(["semana", "nivel_pm"]).size().unstack(fill_value=0)
        .reindex(index=semanas, columns=PM_NIVELES, fill_value=0)
    )
    fig_se = _serie_lineas(serie, ["Sobre la guía OMS", "Sobre el estándar nacional"], COLOR_PM,
                           "Municipios sobre la guía OMS o el estándar nacional por semana",
                           "Número de municipios", semana_sel)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_cl, key="ctx_pm_niveles")
    c2.plotly_chart(fig_se, key="ctx_pm_serie")

    mp = sem.reset_index(drop=True)
    hover = (
        mp["municipio_norm"] + " (" + mp["uf"] + ") | Días sobre la guía OMS: "
        + mp["pm25_dias_sobre_oms"].astype(int).astype(str)
        + " | Días sobre el estándar nacional: " + mp["pm25_dias_sobre_nacional"].astype(int).astype(str)
    )
    st.plotly_chart(
        construir_mapa(mp, "nivel_pm", PM_NIVELES, COLOR_PM, hover,
                       "Municipios por nivel de PM2.5", 12, 520),
        key="ctx_pm_mapa",
    )
    st.caption("Posición: centroide de las alertas del municipio. Clic en la leyenda para ocultar un nivel.")


# ================================================================================ Página
f1, f2, ctx = cargar_datos(firma_archivos())
fecha_fin_datos = f2["view_date"].max()

st.title("BIODIVERSITY-GUARD-Predict")
st.caption(
    "Dashboard preliminar de alerta temprana. Demostración retrospectiva sobre el periodo "
    "de prueba (julio a diciembre de 2024)."
)

VISTAS = [
    "Frente 1: Hectáreas en riesgo",
    "Frente 2: Alertas por nivel de riesgo",
    "Contexto observado: fuego y calidad del aire",
]
vista = st.sidebar.radio("Vista", VISTAS, key="vista")
st.sidebar.markdown("---")

if vista == VISTAS[0]:
    vista_frente1(f1, fecha_fin_datos)
elif vista == VISTAS[1]:
    vista_frente2(f2, fecha_fin_datos)
else:
    vista_contexto(f1, ctx, fecha_fin_datos)