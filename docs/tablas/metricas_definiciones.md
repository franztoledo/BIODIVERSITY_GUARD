| métrica | qué_mide | cómo_se_lee |
|---|---|---|
| MAE | Error absoluto medio entre focos predichos y reales (focos por municipio y semana) | Menor es mejor; está en focos de calor, la misma unidad del objetivo. |
| RMSE | Raíz del error cuadrático medio, en focos | Menor es mejor; castiga más los errores grandes (picos). |
| R² (log1p) | Proporción de la variabilidad de log(1 + focos) que explica el modelo | Más cerca de 1 es mejor; en escala logarítmica, para no dominar por picos. |
| Accuracy | Proporción de semanas con la clase de riesgo bien predicha | Mayor es mejor; engañosa si las clases están desbalanceadas. |
| F1 macro | Promedio del F1 (media de precisión y recall) de las 4 clases, con igual peso | Mayor es mejor; trata igual a las clases raras, como Crítico. |
| Recall (de Crítico) | De las semanas realmente Crítico, cuántas detectó el modelo | Mayor es mejor; mide cuántos casos graves no se pierden. |
| ROC-AUC macro | Área bajo la curva ROC de cada clase contra el resto, promediada | 0.5 es azar y 1 es perfecto; mide la separación sin fijar un umbral. |
| Error ordinal | Distancia media entre clase real y predicha en la escala Bajo a Crítico (0 a 3) | Menor es mejor; confundir Bajo con Crítico pesa más que Alto con Crítico. |
| Recall de alerta | De las semanas-municipio con nivel Muy elevado real, cuántas alertó el modelo (k × focos predichos > p75 de la temporada) | Mayor es mejor; es el KPI de detección. |
| Tasa de falsas alarmas | De las semanas-municipio sin alerta real, cuántas marcó el modelo en alerta | Menor es mejor; mide el costo de alertar de más. |
