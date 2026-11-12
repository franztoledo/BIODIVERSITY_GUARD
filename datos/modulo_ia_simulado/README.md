# Simulacion del modulo de IA

Cada archivo `semana_AAAA-MM-DD.csv` simula lo que el modulo de IA envia al backend una vez por semana: las columnas del panel crudo
municipio-semana (una fila por municipio de esa semana), sin targets y sin columnas preparadas.

**Son datos publicos de INPE (BDQueimadas, DETER), NASA POWER y SISAM** del panel `datos/datasets_eda/dataset_panel_semanal_COMPLETO.csv`,
usados para simular la interfaz. No son datos de una integracion en vivo. Las semanas de 2026 de SISAM son un pronostico sin calibrar.

Los archivos los genera `backend/crear_base.py` (semanas de 2026-01-05 a 2026-10-19) y los consume `backend/demo_flujo.py`.
