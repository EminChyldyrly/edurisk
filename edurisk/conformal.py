"""Конформные наборы оценок: набор классов, содержащий истинную с заданной вероятностью (1 − alpha).

Нонконформность s = 1 − P(истинный класс). Порог q — квантиль уровня ceil((n+1)(1−alpha))/n по калибровочной
выборке (СТУДЕНТЫ ВНЕ ОБУЧЕНИЯ БАЗОВОЙ МОДЕЛИ). Набор: все классы c с 1 − P(c) ≤ q.
`classwise=True` — порог отдельно для каждого истинного класса: покрытие гарантируется внутри каждого класса,
в том числе для редкого класса «2» (что важнее для системы раннего предупреждения, чем среднее покрытие).
Гарантия требует обмениваемости калибровочных и новых данных: при сдвиге когорты она нарушается.
"""
from __future__ import annotations

import numpy as np


def _q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))             # номер порядковой статистики
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def calibrate(P_cal: np.ndarray, y_idx: np.ndarray, alpha: float = 0.1, classwise: bool = False):
    """Порог (число) или вектор порогов по истинным классам."""
    if not 0 < alpha < 1:
        raise ValueError("alpha в (0, 1)")
    s = 1.0 - P_cal[np.arange(len(y_idx)), y_idx]
    if not classwise:
        return _q(s, alpha)
    return np.array([_q(s[y_idx == c], alpha) for c in range(P_cal.shape[1])])


def prediction_sets(P: np.ndarray, q) -> np.ndarray:
    """(n, K) bool: класс входит в набор."""
    return (1.0 - P) <= np.broadcast_to(q, (P.shape[1],))[None, :]


def coverage(sets: np.ndarray, y_idx: np.ndarray) -> float:
    return float(sets[np.arange(len(y_idx)), y_idx].mean())


def mean_size(sets: np.ndarray) -> float:
    return float(sets.sum(1).mean())
