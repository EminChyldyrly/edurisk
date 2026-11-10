"""Анализ справедливости модели риска по группам (gender, origin) и мера по смягчению перекоса.

Группы НЕ входят в признаки модели, но могут влиять через косвенные каналы (например, студенты из других
регионов реже заходят в LMS при той же вовлечённости: сценарий S3). Анализ показывает, нет ли систематически
более частых ложных тревог или пропусков у какой-то группы.

Метрики по группе: доля провалов, средний прогноз, доля помеченных, TPR (доля найденных провалов),
FPR (доля ошибочно помеченных), ROC-AUC, ECE. Разности считаются относительно референсной группы,
доверительные интервалы — бутстрэпом по студентам.
Мера по смягчению: пороги по группам, выравнивающие FPR (постобработка). ВНИМАНИЕ: использование атрибута
группы при принятии решения требует отдельного правового и этического обоснования; по умолчанию не включается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import ece


def rates(y, flag) -> tuple:
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    tpr = float(flag[y].mean()) if y.any() else float("nan")
    fpr = float(flag[~y].mean()) if (~y).any() else float("nan")
    return tpr, fpr


def group_report(y, p, groups, thr: float) -> pd.DataFrame:
    """Таблица метрик по группам при пороге `thr`."""
    y, p, groups = np.asarray(y), np.asarray(p, float), np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        tpr, fpr = rates(y[m], p[m] >= thr)
        rows.append({"group": g, "n": int(m.sum()), "base_rate": float(y[m].mean()), "mean_pred": float(p[m].mean()),
                     "flag_rate": float((p[m] >= thr).mean()), "tpr": tpr, "fpr": fpr,
                     "auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].sum() < m.sum() else float("nan"),
                     "ece": ece(y[m], p[m])})
    return pd.DataFrame(rows).set_index("group")


def disparity_ci(y, p, groups, student_ids, thr: float, ref, other, n_boot: int = 300, seed: int = 0) -> dict:
    """Разности TPR, FPR и доли помеченных (other − ref) с 95% интервалом; бутстрэп по студентам."""
    y, p, groups, sid = map(np.asarray, (y, p, groups, student_ids))
    flag = p >= thr
    uniq, inv = np.unique(sid, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)

    def stat(ix):
        out = []
        for fn in (lambda yy, ff: rates(yy, ff)[0], lambda yy, ff: rates(yy, ff)[1], lambda yy, ff: ff.mean()):
            a = (groups[ix] == other)
            b = (groups[ix] == ref)
            out.append(fn(y[ix][a], flag[ix][a]) - fn(y[ix][b], flag[ix][b]))
        return out

    est = stat(np.arange(len(y)))
    boots = np.array([stat(np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]))
                      for _ in range(n_boot)])
    lo, hi = np.nanquantile(boots, [0.025, 0.975], axis=0)
    return {name: {"diff": float(e), "lo": float(l), "hi": float(h)}
            for name, e, l, h in zip(("tpr", "fpr", "flag_rate"), est, lo, hi)}


def equalize_fpr_thresholds(y, p, groups, target_fpr: float) -> dict:
    """Порог для каждой группы, при котором FPR группы не выше target_fpr (на выборке, по которой подбирают —
    валидации). Возвращает {группа: порог}."""
    y, p, groups = np.asarray(y).astype(bool), np.asarray(p, float), np.asarray(groups)
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr в (0, 1)")
    out = {}
    for g in pd.unique(groups):
        neg = np.sort(p[(groups == g) & ~y])
        if len(neg) == 0:
            out[g] = 1.0
            continue
        k = int(np.floor((1 - target_fpr) * len(neg)))
        k = min(k, len(neg) - 1)
        out[g] = float(np.nextafter(neg[k], 1.0))            # строго выше k-го отрицательного
    return out


def apply_thresholds(p, groups, thr: dict) -> np.ndarray:
    p, groups = np.asarray(p, float), np.asarray(groups)
    t = np.array([thr[g] for g in groups])
    return p >= t
