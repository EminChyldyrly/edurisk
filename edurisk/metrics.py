"""Метрики качества и парный бутстрэп по студентам с поправкой Холма."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    """Ожидаемая ошибка калибровки: взвешенное среднее |доля положительных − средний прогноз| по бинам равной ширины."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def summarize(y, p) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}


def paired_bootstrap(metric, y, pa, pb, groups, n_boot: int = 300, seed: int = 0) -> dict:
    """Разность metric(A) − metric(B) с доверительным интервалом: перевыбираются студенты (группы) целиком.
    p-значение — двустороннее по доле бутстрэп-разностей с противоположным знаком."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    est = metric(y, pa) - metric(y, pb)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([members[g] for g in pick])
        if y[ix].min() == y[ix].max():
            diffs[i] = np.nan
            continue
        diffs[i] = metric(y[ix], pa[ix]) - metric(y[ix], pb[ix])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(est), "lo": float(lo), "hi": float(hi), "p": float(max(p, 1.0 / (len(diffs) + 1)))}


def holm(pvals) -> np.ndarray:
    """Поправка Холма–Бонферрони: скорректированные p-значения в исходном порядке."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_bootstrap_loss(loss_a, loss_b, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Разность средних потерь A − B (отрицательная = A лучше) с бутстрэпом по студентам."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    pick = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    boot = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(max(p, 1.0 / (n_boot + 1)))}
