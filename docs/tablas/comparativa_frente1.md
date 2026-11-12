| horizonte | modelo | variante | SISAM | MAE val. media | MAE val. desv. | MAE 2025 | RMSE val. media | RMSE val. desv. | RMSE 2025 | R² (log1p) val. media | R² (log1p) val. desv. | R² (log1p) 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 días | Persistencia | línea base | no aplica | 3.461 | 3.130 | 1.451 | 14.205 | 11.899 | 5.291 | 0.399 | 0.113 | 0.357 |
| 7 días | Regresión Lineal | línea base | con | 3.274 | 3.134 | 1.295 | 15.376 | 15.049 | 4.849 | 0.561 | 0.073 | 0.533 |
| 7 días | Regresión Lineal | línea base | sin | 3.281 | 3.160 | 1.299 | 15.728 | 15.873 | 4.823 | 0.562 | 0.076 | 0.532 |
| 7 días | Árbol de Regresión | línea base | con | 3.383 | 3.115 | 1.382 | 15.094 | 13.672 | 5.611 | 0.490 | 0.105 | 0.502 |
| 7 días | Árbol de Regresión | línea base | sin | 3.357 | 3.098 | 1.376 | 14.980 | 13.690 | 5.475 | 0.493 | 0.103 | 0.505 |
| 7 días | Random Forest | por defecto | con | 3.099 | 2.828 | 1.357 | 14.115 | 13.429 | 5.346 | 0.559 | 0.087 | 0.522 |
| 7 días | Random Forest | ajustado | con | 3.040 | 2.812 | 1.304 | 14.027 | 13.501 | 4.976 | 0.587 | 0.077 | 0.552 |
| 7 días | Random Forest | ajustado | sin | 3.041 | 2.813 | 1.291 | 14.063 | 13.535 | 4.884 | 0.586 | 0.077 | 0.556 |
| 7 días | XGBoost | por defecto | con | 3.200 | 2.979 | 1.364 | 15.198 | 14.363 | 6.024 | 0.548 | 0.076 | 0.531 |
| 7 días | XGBoost | ajustado | con | 3.004 | 2.803 | 1.303 | 14.204 | 13.997 | 5.223 | 0.588 | 0.076 | 0.558 |
| 7 días | XGBoost | ajustado | sin | 3.026 | 2.833 | 1.286 | 14.368 | 14.335 | 5.062 | 0.587 | 0.076 | 0.561 |
| 7 días | LightGBM | por defecto | con | 3.015 | 2.805 | 1.290 | 13.954 | 13.329 | 5.010 | 0.585 | 0.080 | 0.555 |
| 7 días | LightGBM | ajustado | con | 2.982 | 2.767 | 1.280 | 14.102 | 13.883 | 4.896 | 0.592 | 0.077 | 0.562 |
| 7 días | LightGBM (final) | ajustado | sin | 3.001 | 2.786 | 1.270 | 14.113 | 13.857 | 4.936 | 0.589 | 0.080 | 0.565 |
| 30 días | Persistencia | línea base | no aplica | 14.752 | 13.881 | 5.802 | 63.183 | 60.743 | 20.373 | 0.324 | 0.049 | 0.302 |
| 30 días | Regresión Lineal | línea base | con | 12.451 | 12.168 | 4.655 | 53.254 | 54.313 | 16.377 | 0.594 | 0.074 | 0.597 |
| 30 días | Regresión Lineal | línea base | sin | 12.471 | 12.126 | 4.651 | 53.688 | 54.285 | 16.304 | 0.593 | 0.073 | 0.596 |
| 30 días | Árbol de Regresión | línea base | con | 12.444 | 11.211 | 5.118 | 53.215 | 48.882 | 22.765 | 0.548 | 0.114 | 0.590 |
| 30 días | Árbol de Regresión | línea base | sin | 12.271 | 11.134 | 5.025 | 52.938 | 48.826 | 21.429 | 0.568 | 0.085 | 0.595 |
| 30 días | Random Forest | por defecto | con | 10.465 | 9.301 | 4.566 | 46.587 | 44.972 | 18.141 | 0.646 | 0.066 | 0.641 |
| 30 días | Random Forest | ajustado | con | 10.361 | 9.295 | 4.410 | 46.634 | 46.059 | 16.941 | 0.664 | 0.058 | 0.660 |
| 30 días | Random Forest | ajustado | sin | 10.410 | 9.355 | 4.361 | 47.032 | 46.484 | 16.377 | 0.663 | 0.057 | 0.663 |
| 30 días | Random Forest | ajustado con parámetros de 7 d | con | 10.361 | 9.295 |  | 46.634 | 46.059 |  | 0.664 | 0.058 |  |
| 30 días | XGBoost | por defecto | con | 11.157 | 10.136 | 4.825 | 54.435 | 51.564 | 21.285 | 0.659 | 0.049 | 0.655 |
| 30 días | XGBoost (final) | ajustado | con | 9.942 | 8.904 | 4.616 | 45.434 | 45.143 | 20.288 | 0.691 | 0.048 | 0.673 |
| 30 días | XGBoost | ajustado | sin | 10.172 | 9.095 | 4.366 | 47.806 | 46.420 | 18.183 | 0.689 | 0.050 | 0.680 |
| 30 días | XGBoost | ajustado con parámetros de 7 d | con | 9.942 | 8.904 |  | 45.434 | 45.143 |  | 0.691 | 0.048 |  |
| 30 días | LightGBM | por defecto | con | 10.151 | 9.193 | 4.479 | 45.555 | 46.123 | 18.493 | 0.679 | 0.051 | 0.673 |
| 30 días | LightGBM | ajustado | con | 10.031 | 9.053 | 4.448 | 46.292 | 46.707 | 18.040 | 0.689 | 0.048 | 0.674 |
| 30 días | LightGBM | ajustado | sin | 10.040 | 9.040 | 4.304 | 46.386 | 47.093 | 17.382 | 0.688 | 0.050 | 0.682 |
| 30 días | LightGBM | ajustado con parámetros de 7 d | con | 10.031 | 9.053 |  | 46.292 | 46.707 |  | 0.689 | 0.048 |  |

MAE y RMSE en focos de calor por municipio y semana; R² medido sobre log1p de los focos.
