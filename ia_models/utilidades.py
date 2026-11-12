"""Funciones propias que usan los modelos guardados de BIODIVERSITY-GUARD-Predict.

Los modelos de `ia_models/*.joblib` guardan la referencia a este modulo (`ia_models.utilidades`) y no a `__main__`,
asi que se pueden cargar desde cualquier proceso con `import ia_models.utilidades` antes de `joblib.load`.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def inv_log1p_no_negativo(x):
    # inversa de log1p con minimo 0 (los focos no pueden ser negativos)
    return np.clip(np.expm1(x), 0, None)


def con_log(regresor):
    # el modelo se entrena con log1p(y) y predice en focos reales
    return TransformedTargetRegressor(regressor=regresor, func=np.log1p, inverse_func=inv_log1p_no_negativo, check_inverse=False)


class XGBClasificadorBalanceado(XGBClassifier):
    # XGBoost con pesos de clase 'balanced' calculados con las etiquetas de entrenamiento
    def fit(self, X, y, **kwargs):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kwargs)
