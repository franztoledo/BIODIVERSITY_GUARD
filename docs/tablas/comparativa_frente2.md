| modelo | variante | tratamiento_desbalance | SISAM | Accuracy val. media | Accuracy val. desv. | Accuracy 2025 | F1 macro val. media | F1 macro val. desv. | F1 macro 2025 | Recall Crítico val. media | Recall Crítico val. desv. | Recall Crítico 2025 | Error ordinal val. media | Error ordinal val. desv. | Error ordinal 2025 | ROC-AUC macro val. media | ROC-AUC macro val. desv. | ROC-AUC macro 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistencia | línea base |  | no aplica | 0.686 | 0.064 | 0.582 | 0.555 | 0.060 | 0.562 | 0.545 | 0.261 | 0.635 | 0.393 | 0.072 | 0.527 |  |  |  |
| Árbol de decisión | línea base |  | con | 0.651 | 0.076 | 0.579 | 0.519 | 0.058 | 0.541 | 0.456 | 0.203 | 0.635 | 0.450 | 0.105 | 0.561 |  |  |  |
| Árbol de decisión | línea base |  | sin | 0.650 | 0.079 | 0.582 | 0.521 | 0.059 | 0.544 | 0.457 | 0.198 | 0.649 | 0.449 | 0.108 | 0.566 |  |  |  |
| Árbol balanceado | línea base | pesos de clase | con | 0.615 | 0.084 | 0.560 | 0.511 | 0.053 | 0.553 | 0.425 | 0.251 | 0.578 | 0.477 | 0.113 | 0.539 |  |  |  |
| Árbol balanceado | línea base | pesos de clase | sin | 0.604 | 0.081 | 0.569 | 0.497 | 0.060 | 0.562 | 0.404 | 0.277 | 0.606 | 0.491 | 0.106 | 0.528 |  |  |  |
| Random Forest | por defecto | sin remuestreo | con | 0.661 | 0.069 | 0.591 | 0.506 | 0.040 | 0.544 | 0.481 | 0.253 | 0.622 | 0.452 | 0.102 | 0.553 | 0.849 | 0.032 | 0.828 |
| Random Forest | ajustado | pesos de clase | con | 0.686 | 0.060 | 0.606 | 0.570 | 0.065 | 0.589 | 0.509 | 0.278 | 0.616 | 0.380 | 0.072 | 0.482 | 0.871 | 0.027 | 0.843 |
| Random Forest | ajustado | pesos de clase | sin | 0.685 | 0.057 | 0.611 | 0.570 | 0.066 | 0.595 | 0.517 | 0.283 | 0.622 | 0.381 | 0.069 | 0.476 | 0.871 | 0.026 | 0.843 |
| Random Forest | ajustado | sin remuestreo | con | 0.689 | 0.065 | 0.608 | 0.546 | 0.056 | 0.572 | 0.543 | 0.274 | 0.633 | 0.409 | 0.097 | 0.523 | 0.870 | 0.027 | 0.840 |
| Random Forest | ajustado | sin remuestreo | sin | 0.689 | 0.065 | 0.612 | 0.546 | 0.061 | 0.575 | 0.549 | 0.276 | 0.631 | 0.409 | 0.098 | 0.522 | 0.871 | 0.026 | 0.840 |
| Random Forest | ajustado | SMOTE | con | 0.678 | 0.064 | 0.603 | 0.573 | 0.066 | 0.591 | 0.506 | 0.284 | 0.621 | 0.385 | 0.075 | 0.483 | 0.871 | 0.028 | 0.843 |
| Random Forest (final) | ajustado | SMOTE | sin | 0.680 | 0.062 | 0.602 | 0.572 | 0.062 | 0.591 | 0.503 | 0.290 | 0.627 | 0.382 | 0.071 | 0.487 | 0.871 | 0.027 | 0.841 |
| XGBoost | por defecto | sin remuestreo | con | 0.656 | 0.072 | 0.606 | 0.515 | 0.039 | 0.574 | 0.469 | 0.196 | 0.621 | 0.456 | 0.108 | 0.522 | 0.852 | 0.032 | 0.838 |
| XGBoost | ajustado | pesos de clase | con | 0.663 | 0.052 | 0.610 | 0.553 | 0.068 | 0.602 | 0.483 | 0.267 | 0.615 | 0.400 | 0.062 | 0.468 | 0.866 | 0.027 | 0.848 |
| XGBoost | ajustado | pesos de clase | sin | 0.666 | 0.055 | 0.603 | 0.557 | 0.063 | 0.594 | 0.487 | 0.271 | 0.607 | 0.397 | 0.066 | 0.477 | 0.867 | 0.027 | 0.848 |
| XGBoost | ajustado | sin remuestreo | con | 0.676 | 0.067 | 0.616 | 0.536 | 0.042 | 0.580 | 0.484 | 0.240 | 0.644 | 0.424 | 0.101 | 0.499 | 0.866 | 0.028 | 0.846 |
| XGBoost | ajustado | sin remuestreo | sin | 0.674 | 0.066 | 0.613 | 0.531 | 0.042 | 0.578 | 0.482 | 0.250 | 0.646 | 0.427 | 0.102 | 0.504 | 0.866 | 0.028 | 0.845 |
| XGBoost | ajustado | SMOTE | con | 0.676 | 0.058 | 0.607 | 0.572 | 0.062 | 0.593 | 0.518 | 0.276 | 0.636 | 0.391 | 0.070 | 0.480 | 0.868 | 0.029 | 0.842 |
| XGBoost | ajustado | SMOTE | sin | 0.672 | 0.059 | 0.603 | 0.567 | 0.060 | 0.590 | 0.513 | 0.269 | 0.625 | 0.395 | 0.071 | 0.487 | 0.868 | 0.028 | 0.841 |
| LightGBM | por defecto | sin remuestreo | con | 0.671 | 0.072 | 0.612 | 0.539 | 0.036 | 0.577 | 0.516 | 0.163 | 0.632 | 0.430 | 0.103 | 0.510 | 0.860 | 0.029 | 0.842 |
| LightGBM | ajustado | pesos de clase | con | 0.667 | 0.056 | 0.608 | 0.560 | 0.066 | 0.599 | 0.507 | 0.262 | 0.627 | 0.399 | 0.068 | 0.474 | 0.866 | 0.029 | 0.845 |
| LightGBM | ajustado | pesos de clase | sin | 0.666 | 0.056 | 0.605 | 0.557 | 0.071 | 0.594 | 0.505 | 0.283 | 0.616 | 0.400 | 0.066 | 0.476 | 0.866 | 0.028 | 0.845 |
| LightGBM | ajustado | sin remuestreo | con | 0.678 | 0.074 | 0.605 | 0.540 | 0.040 | 0.569 | 0.511 | 0.216 | 0.654 | 0.427 | 0.116 | 0.524 | 0.864 | 0.031 | 0.840 |
| LightGBM | ajustado | sin remuestreo | sin | 0.673 | 0.075 | 0.604 | 0.531 | 0.048 | 0.570 | 0.502 | 0.234 | 0.649 | 0.434 | 0.117 | 0.524 | 0.862 | 0.032 | 0.838 |
| LightGBM | ajustado | SMOTE | con | 0.678 | 0.059 | 0.611 | 0.562 | 0.056 | 0.588 | 0.530 | 0.238 | 0.635 | 0.397 | 0.072 | 0.487 | 0.866 | 0.028 | 0.843 |
| LightGBM | ajustado | SMOTE | sin | 0.681 | 0.060 | 0.608 | 0.561 | 0.057 | 0.585 | 0.504 | 0.264 | 0.632 | 0.396 | 0.073 | 0.493 | 0.866 | 0.028 | 0.841 |

Evaluado en filas con clase de la semana actual. ROC-AUC macro no aplica a la persistencia ni al árbol de la línea base (vacío).
