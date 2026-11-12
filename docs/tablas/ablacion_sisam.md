| tarea | modelo | métrica | con_SISAM_val | sin_SISAM_val | mejora_con_SISAM_% | con_SISAM_2025 | sin_SISAM_2025 | decisión |
|---|---|---|---|---|---|---|---|---|
| Frente 1 a 7 días | Random Forest | MAE (menor es mejor) | 3.040 | 3.041 | 0.048 | 1.304 | 1.291 | Sin SISAM |
| Frente 1 a 7 días | XGBoost | MAE (menor es mejor) | 3.004 | 3.026 | 0.726 | 1.303 | 1.286 | Sin SISAM |
| Frente 1 a 7 días | LightGBM | MAE (menor es mejor) | 2.982 | 3.001 | 0.629 | 1.280 | 1.270 | Sin SISAM |
| Frente 1 a 30 días | Random Forest | MAE (menor es mejor) | 10.361 | 10.410 | 0.478 | 4.410 | 4.361 | Con SISAM |
| Frente 1 a 30 días | XGBoost | MAE (menor es mejor) | 9.942 | 10.172 | 2.258 | 4.616 | 4.366 | Con SISAM |
| Frente 1 a 30 días | LightGBM | MAE (menor es mejor) | 10.031 | 10.040 | 0.090 | 4.448 | 4.304 | Con SISAM |
| Frente 2 (sin remuestreo) | Random Forest | F1 macro (mayor es mejor) | 0.546 | 0.546 | 0.058 | 0.572 | 0.575 | Sin SISAM |
| Frente 2 (pesos de clase) | Random Forest | F1 macro (mayor es mejor) | 0.570 | 0.570 | 0.023 | 0.589 | 0.595 | Sin SISAM |
| Frente 2 (SMOTE) | Random Forest | F1 macro (mayor es mejor) | 0.573 | 0.572 | 0.148 | 0.591 | 0.591 | Sin SISAM |
| Frente 2 (sin remuestreo) | XGBoost | F1 macro (mayor es mejor) | 0.536 | 0.531 | 0.849 | 0.580 | 0.578 | Sin SISAM |
| Frente 2 (pesos de clase) | XGBoost | F1 macro (mayor es mejor) | 0.553 | 0.557 | -0.687 | 0.602 | 0.594 | Sin SISAM |
| Frente 2 (SMOTE) | XGBoost | F1 macro (mayor es mejor) | 0.572 | 0.567 | 0.877 | 0.593 | 0.590 | Sin SISAM |
| Frente 2 (sin remuestreo) | LightGBM | F1 macro (mayor es mejor) | 0.540 | 0.531 | 1.697 | 0.569 | 0.570 | Sin SISAM |
| Frente 2 (pesos de clase) | LightGBM | F1 macro (mayor es mejor) | 0.560 | 0.557 | 0.393 | 0.599 | 0.594 | Sin SISAM |
| Frente 2 (SMOTE) | LightGBM | F1 macro (mayor es mejor) | 0.562 | 0.561 | 0.098 | 0.588 | 0.585 | Sin SISAM |

Regla: SISAM entra solo si mejora la validación al menos 1 % (MAE) o 0.01 (F1 macro) y en al menos 3 de 4 folds. La decisión sale de ia_models/modelos_metadata.json (lista_variables del modelo final); 2025 no interviene. Mejora en %: positivo favorece a SISAM.
