"""Недельные признаки. Главный принцип: признаки на неделю t считаются ТОЛЬКО по срезу
`cohort.observed(t)`, в котором будущих данных физически нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import Cohort

STATIC = ["ege", "workload_ze", "has_job"]
RAW = ["logins_total", "logins_last2", "logins_last4", "minutes_mean4", "cp_failed_count", "cp_missed_count",
       "last_score", "mean_score"]
DERIVED = ["activity_trend4", "logins_ewma", "weeks_since_login", "zero_login_weeks", "cp_fail_rate",
           "cp_fail_streak", "cp_max_fail_streak", "score_delta", "activity_vs_cohort"]
FEATURES = STATIC + RAW + DERIVED
AUDIT = ["gender", "origin"]           # только для анализа справедливости, в признаки не входят
EWMA_ALPHA = 0.4      # вес самой свежей недели


def _slope(y: np.ndarray) -> np.ndarray:
    """Наклон МНК по строкам; для одной точки 0."""
    w = y.shape[1]
    if w < 2:
        return np.zeros(y.shape[0])
    x = np.arange(w, dtype=float)
    x -= x.mean()
    return (y * x).sum(1) / (x ** 2).sum()


def _streaks(failed: np.ndarray, sched: np.ndarray):
    """Текущая и максимальная серии подряд проваленных контрольных точек (по назначенным точкам)."""
    n = failed.shape[0]
    cur = np.zeros(n, dtype=int)
    best = np.zeros(n, dtype=int)
    for col in np.flatnonzero(sched):
        cur = np.where(failed[:, col], cur + 1, 0)
        best = np.maximum(best, cur)
    return cur, best


def build_features(cohort: Cohort, week: int) -> pd.DataFrame:
    """Признаки всех студентов на конец недели `week` (индекс — student_id)."""
    c = cohort.observed(week)
    L, M, S = c.logins.astype(float), c.minutes, c.cp_score
    k = week
    failed = c.cp_failed()
    sched = c.cp_scheduled
    n_cp = int(sched.sum())

    out = c.students[STATIC].copy()
    out["logins_total"] = L.sum(1)
    out["logins_last2"] = L[:, -2:].sum(1)
    out["logins_last4"] = L[:, -4:].sum(1)
    out["minutes_mean4"] = M[:, -4:].mean(1)
    out["cp_failed_count"] = failed.sum(1)
    out["cp_missed_count"] = (sched[None, :] & np.isnan(S)).sum(1)

    got = ~np.isnan(S)                                    # сданные контрольные (есть балл)
    n_got = got.sum(1)
    with np.errstate(invalid="ignore"):
        mean_score = np.where(n_got > 0, np.nansum(S, 1) / np.maximum(n_got, 1), np.nan)
    last_col = np.flatnonzero(sched)[-1] if n_cp else None
    out["last_score"] = S[:, last_col] if n_cp else np.nan
    out["mean_score"] = mean_score

    out["activity_trend4"] = _slope(L[:, -4:])
    ew = L[:, 0].copy()
    for t in range(1, k):
        ew = EWMA_ALPHA * L[:, t] + (1 - EWMA_ALPHA) * ew
    out["logins_ewma"] = ew
    has = L > 0
    last_login = np.where(has.any(1), k - 1 - np.argmax(has[:, ::-1], axis=1), -1)
    out["weeks_since_login"] = k - 1 - last_login
    out["zero_login_weeks"] = (~has).sum(1)
    out["cp_fail_rate"] = out["cp_failed_count"] / n_cp if n_cp else 0.0
    cur, best = _streaks(failed, sched)
    out["cp_fail_streak"], out["cp_max_fail_streak"] = cur, best

    # изменение балла: последние два сданных
    delta = np.full(len(out), np.nan)
    for i in range(len(out)):
        idx = np.flatnonzero(got[i])
        if len(idx) >= 2:
            delta[i] = S[i, idx[-1]] - S[i, idx[-2]]
    out["score_delta"] = delta

    mean4 = out["logins_last4"].mean()
    out["activity_vs_cohort"] = out["logins_last4"] / mean4 if mean4 > 0 else 0.0
    return out[FEATURES]


def build_panel(cohort: Cohort, weeks) -> pd.DataFrame:
    """Длинная таблица «студент × неделя» с признаками, колонками week и target."""
    parts = []
    for w in weeks:
        f = build_features(cohort, w)
        f["week"] = w
        f["target"] = cohort.outcome.astype(int)
        parts.append(f)
    return pd.concat(parts)
