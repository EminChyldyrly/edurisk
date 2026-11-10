"""Оценка 2–5 как порядковая цель (T2): каскад Фрэнка–Холла, многоклассовые базовые модели и решения по
матрице стоимости с ЗЕТ-весом.

Каскад: три бинарные модели P(оценка ≥ 3), P(≥ 4), P(= 5); вероятности классов получаются разностями после
принудительной монотонности (P≥3 ≥ P≥4 ≥ P=5), поэтому классы всегда неотрицательны и суммируются в 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import MODEL_FEATURES

GRADES = np.array([2, 3, 4, 5])


def cost_matrix(asym: float = 3.0) -> np.ndarray:
    """M[i, j] = стоимость решения «оценка GRADES[j]» при истинной GRADES[i]: |разница| ступеней, а
    завышение прогноза (предсказали лучше, чем будет: студент не получит помощь) в asym раз дороже занижения."""
    if asym <= 0:
        raise ValueError("asym должна быть положительной")
    d = GRADES[None, :] - GRADES[:, None]                      # >0: предсказали выше истины
    return np.abs(d) * np.where(d > 0, asym, 1.0)


def decide_min_cost(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Оценка с минимальной ожидаемой стоимостью: argmin_j Σ_i P[i] M[i, j]."""
    return GRADES[np.argmin(P @ M, axis=1)]


def decide_argmax(P: np.ndarray) -> np.ndarray:
    return GRADES[np.argmax(P, axis=1)]


def expected_cost(y_grade, pred, w, M) -> float:
    """Средняя по объектам стоимость w · M[y, pred]."""
    iy, ip = np.searchsorted(GRADES, y_grade), np.searchsorted(GRADES, pred)
    return float(np.mean(np.asarray(w, float) * M[iy, ip]))


def class_metrics(y_grade, pred) -> dict:
    y_grade, pred = np.asarray(y_grade), np.asarray(pred)
    return {"qwk": float(cohen_kappa_score(y_grade, pred, weights="quadratic")),
            "mae": float(np.mean(np.abs(y_grade - pred))),
            "macro_f1": float(f1_score(y_grade, pred, average="macro", labels=GRADES, zero_division=0)),
            "recall_2": float(np.mean(pred[y_grade == 2] == 2)) if (y_grade == 2).any() else float("nan"),
            "accuracy": float(np.mean(y_grade == pred))}


def probs_from_cumulative(c3, c4, c5) -> np.ndarray:
    """P(≥3), P(≥4), P(=5) → вероятности четырёх классов с принудительной монотонностью."""
    C = np.minimum.accumulate(np.clip(np.column_stack([c3, c4, c5]), 0.0, 1.0), axis=1)
    return np.column_stack([1 - C[:, 0], C[:, 0] - C[:, 1], C[:, 1] - C[:, 2], C[:, 2]])


class FrankHall:
    """Каскад из трёх бинарных моделей. `factory()` создаёт модель с fit(X, y, sample_weight)/predict_proba."""
    CUTS = (3, 4, 5)

    def __init__(self, factory, calibrate: bool = True):
        self.factory, self.calibrate = factory, calibrate

    def fit(self, X, grade, sample_weight=None):
        g = np.asarray(grade)
        self.models_ = [self.factory().fit(X, (g >= c).astype(int), sample_weight) for c in self.CUTS]
        return self

    def fit_calibration(self, X_val, grade_val):
        g = np.asarray(grade_val)
        self.iso_ = [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            m.predict_proba(X_val), (g >= c).astype(int)) for m, c in zip(self.models_, self.CUTS)]
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = [m.predict_proba(X) for m in self.models_]
        if self.calibrate:
            raw = [i.predict(r) for i, r in zip(self.iso_, raw)]
        return probs_from_cumulative(*raw)


class Multinomial:
    """Многоклассовая модель без учёта порядка (базовый уровень): logit или бустинг."""
    def __init__(self, kind: str = "logit", seed: int = 0):
        if kind == "logit":
            self.m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                   LogisticRegression(C=0.5, max_iter=2000))
        elif kind == "boost":
            self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=seed)
        else:
            raise ValueError("kind: 'logit' или 'boost'")
        self.kind = kind

    def fit(self, X, grade, sample_weight=None):
        self.cols_ = [c for c in MODEL_FEATURES if X[c].notna().any()]
        if self.kind == "logit":
            kw = {} if sample_weight is None else {"logisticregression__sample_weight": sample_weight}
            self.m.fit(X[self.cols_], grade, **kw)
        else:
            self.m.fit(X[self.cols_], grade, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        P = self.m.predict_proba(X[self.cols_])
        out = np.zeros((len(X), 4))
        out[:, np.searchsorted(GRADES, self.m.classes_)] = P
        return out


class PriorModel:
    """Базовый уровень: доли оценок в обучении."""
    def fit(self, X, grade, sample_weight=None):
        g = np.asarray(grade)
        self.p_ = np.array([(g == c).mean() for c in GRADES])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.tile(self.p_, (len(X), 1))
