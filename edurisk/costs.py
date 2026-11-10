"""Стоимость ошибок с учётом З.Е. и решение по порогу.

Для бинарной задачи «риск» матрица M = [[0, c_fa], [k, 0]] (ложная тревога стоит c_fa, пропущенный
провал k), а вес дисциплины w(d) = (З.Е.(d) / средние З.Е.)^gamma умножает ОБЕ ошибки. Поэтому порог решения
p* = c_fa / (c_fa + k) от З.Е. не зависит. ЗЕТ-вес меняет (1) итоговую оценку стоимости, (2) вес объектов при
обучении, (3) порядок в ограниченном списке внимания (приоритет = p · w · k).
"""
from __future__ import annotations

import numpy as np


def ze_weights(ze, gamma: float = 2.0, mean_ze: float | None = None) -> np.ndarray:
    ze = np.asarray(ze, float)
    if (ze <= 0).any():
        raise ValueError("З.Е. должны быть положительными")
    m = ze.mean() if mean_ze is None else mean_ze
    return (ze / m) ** gamma


def bayes_threshold(k: float, c_fa: float = 1.0) -> float:
    if k <= 0 or c_fa <= 0:
        raise ValueError("стоимости должны быть положительными")
    return c_fa / (c_fa + k)


def mean_cost(y, flag, w, k: float = 3.0, c_fa: float = 1.0) -> float:
    """Средняя по объектам стоимость: w·(k·[y=1, flag=0] + c_fa·[y=0, flag=1])."""
    y, flag, w = np.asarray(y).astype(bool), np.asarray(flag).astype(bool), np.asarray(w, float)
    return float(np.mean(w * (k * (y & ~flag) + c_fa * (~y & flag))))


def best_threshold(y, p, w, k: float = 3.0, c_fa: float = 1.0) -> float:
    """Порог, минимизирующий эмпирическую стоимость (подбирается на валидации)."""
    p = np.asarray(p, float)
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    costs = [mean_cost(y, p >= g, w, k, c_fa) for g in grid]
    return float(grid[int(np.argmin(costs))])


def weighted_recall_at_n(y, p, w, n: int, k: float = 3.0) -> float:
    """Доля «взвешенных провалов», попавших в верхние n по приоритету p·w·k."""
    y, p, w = np.asarray(y).astype(bool), np.asarray(p, float), np.asarray(w, float)
    if not 0 < n <= len(y):
        raise ValueError("n должно быть от 1 до числа объектов")
    top = np.argsort(-(p * w * k), kind="stable")[:n]
    tot = (w * y).sum()
    return float((w[top] * y[top]).sum() / tot) if tot > 0 else float("nan")


def row_cost(y, flag, w, k: float = 3.0, c_fa: float = 1.0) -> np.ndarray:
    """Потеря каждого объекта; mean_cost — её среднее."""
    y, flag, w = np.asarray(y).astype(bool), np.asarray(flag).astype(bool), np.asarray(w, float)
    return w * (k * (y & ~flag) + c_fa * (~y & flag))
