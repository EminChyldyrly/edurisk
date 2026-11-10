"""Модели риска: правило, логистическая регрессия (elastic net), бустинг, калибровка и стекинг.

Все модели имеют fit(X, y, sample_weight=None) / predict_proba(X) -> вектор вероятности провала.
Калибратор и стекинг обучаются на ВАЛИДАЦИИ (студенты, не входившие в обучение базовых моделей).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import MODEL_FEATURES


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class RuleModel:
    """Базовый уровень: «2 проваленные контрольные подряд» или «средний балл прошлых семестров < 3».
    Вероятность в каждой из двух групп — доля провалов этой группы в обучении (так правило честно сравнимо
    по Brier и ECE); ранжирование грубое, из двух значений."""
    def fit(self, X, y, sample_weight=None):
        r = self._rule(X)
        y = np.asarray(y)
        self.p_ = (float(y[r].mean()) if r.any() else 0.5, float(y[~r].mean()) if (~r).any() else 0.5)
        return self

    @staticmethod
    def _rule(X):
        gpa = X["prior_gpa"].to_numpy(float)
        return (X["cp_max_fail_streak"].to_numpy() >= 2) | (np.nan_to_num(gpa, nan=5.0) < 3.0)

    def predict_proba(self, X):
        return np.where(self._rule(X), self.p_[0], self.p_[1])


class LogitModel:
    def __init__(self, C: float = 0.5, l1_ratio: float = 0.5, features=None):
        self.features = list(features) if features is not None else MODEL_FEATURES
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=3000))

    def fit(self, X, y, sample_weight=None):
        kw = {} if sample_weight is None else {"logisticregression__sample_weight": sample_weight}
        self.cols_ = [c for c in self.features if X[c].notna().any()]
        self.pipe.fit(X[self.cols_], y, **kw)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[self.cols_])[:, 1]


class BoostModel:
    def __init__(self, seed: int = 0, max_iter: int = 200, lr: float = 0.06, features=None):
        self.features = list(features) if features is not None else MODEL_FEATURES
        self.m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=seed)

    def fit(self, X, y, sample_weight=None):
        self.cols_ = [c for c in self.features if X[c].notna().any()]
        self.m.fit(X[self.cols_], y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[self.cols_])[:, 1]


class Calibrated:
    """Изотоническая калибровка прогноза уже обученной модели; обучается на валидации."""
    def __init__(self, base):
        self.base = base

    def fit_calibration(self, X_val, y_val):
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            self.base.predict_proba(X_val), y_val)
        return self

    def predict_proba(self, X):
        return self.iso_.predict(self.base.predict_proba(X))


class Stack:
    """Стекинг: логистическая регрессия над логитами базовых моделей; веса подбираются только на валидации."""
    def __init__(self, models):
        self.models = models

    def _Z(self, X):
        return np.column_stack([_logit(m.predict_proba(X)) for m in self.models])

    def fit_stack(self, X_val, y_val):
        self.meta_ = LogisticRegression(C=1.0, max_iter=1000).fit(self._Z(X_val), y_val)
        self.weights_ = self.meta_.coef_[0]
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._Z(X))[:, 1]
