# BIODIVERSITY-GUARD-Predict

Sistema de Machine Learning para la gestión forestal y alerta temprana de amenazas en la Amazonía, desarrollado para el caso BIODIVERSITY-GUARD (curso 1ACC0057 – Machine Learning, UPC).

El proyecto integra cuatro fuentes abiertas de datos —**DETER**, **BDQueimadas** y **SISAM** (INPE) y **NASA POWER**— para construir dos modelos predictivos:

- **Frente 1 (Regresión / Series de tiempo):** predicción de hectáreas en riesgo de deforestación a horizonte de 7 y 30 días.
- **Frente 2 (Clasificación multiclase):** clasificación del nivel de riesgo de amenaza (Bajo / Moderado / Alto / Crítico) de cada alerta detectada.

El pipeline sigue la metodología **CRISP-DM**, desde la extracción y limpieza de datos hasta el análisis exploratorio (EDA), con documentación de cada decisión metodológica (tratamiento de duplicados, ventanas móviles sin fuga de información, corrección de claves de cruce entre municipios homónimos, entre otras).

## Estructura del repositorio
- `/codigo` — notebooks de limpieza, merge y EDA
- `/datos` — datasets procesados (anonimizados)
- `/docs` — informe técnico, diccionario de datos y manual de usuario
