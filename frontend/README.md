# Dashboard BIODIVERSITY-GUARD-Predict

Dashboard Streamlit con KPI y umbrales, alertas, proyección, control del modelo e ingesta de una semana nueva.

## Correr

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Desde la raíz del repositorio. Por defecto usa el servicio real (`backend/servicio.py`; la primera carga prepara la base de datos). Si falla, cae al stub con un aviso.

## Servicio de ejemplo (stub)

```bash
BG_SERVICIO=stub streamlit run frontend/app.py
```

(PowerShell: `$env:BG_SERVICIO="stub"`). Si `backend/servicio.py` no existe o falla al importar, la app cae al stub y muestra un aviso.

El dashboard solo usa lo que devuelve `conexion.obtener_servicio()`. Funciones extra al contrato (opcionales, la app tiene respaldo): `umbrales_contexto()` e `historial_ingestas()`.

Páginas: Panel de control, Proyección, Riesgo de fuego, Contexto, Control del modelo, Ingesta, Ayuda y límites. Se puede abrir una directamente con `?pagina=Ingesta`.
