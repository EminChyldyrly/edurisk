"""Протокол этапа 4: обучение/калибровка/стекинг на разных студентах, оценка на тестовых."""
from __future__ import annotations

import numpy as np

from .costs import bayes_threshold, mean_cost, weighted_recall_at_n, ze_weights
from .dataset import build_dataset, split_students, take
from .metrics import summarize
from .models import BoostModel, Calibrated, LogitModel, RuleModel, Stack
from .world import simulate_world

K, GAMMA, TOP_SHARE = 3.0, 2.0, 0.10
MODELS = ["rule", "logit", "boost", "boost+cal", "boost_cs", "boost_cs+cal", "stack"]


def fit_all(train, val, mean_ze: float, k: float = K, gamma: float = GAMMA, seed: int = 0) -> dict:
    """Обучает все модели на train, калибрует и стекует на val."""
    ytr, yva = train.y.to_numpy(), val.y.to_numpy()
    wz = ze_weights(train.ze, gamma, mean_ze)
    cs = wz * np.where(ytr == 1, k, 1.0)                       # ЗЕТ-вес × вес класса
    rule = RuleModel().fit(train, ytr)
    logit = LogitModel().fit(train, ytr)
    boost = BoostModel(seed).fit(train, ytr)
    boost_cs = BoostModel(seed).fit(train, ytr, sample_weight=cs)
    boost_cal = Calibrated(boost).fit_calibration(val, yva)
    boost_cs_cal = Calibrated(boost_cs).fit_calibration(val, yva)   # возвращает вероятности к реальным
    stack = Stack([logit, boost, boost_cs_cal, rule]).fit_stack(val, yva)
    return {"rule": rule, "logit": logit, "boost": boost, "boost+cal": boost_cal, "boost_cs": boost_cs,
            "boost_cs+cal": boost_cs_cal, "stack": stack}


def evaluate(models: dict, test, mean_ze: float, k: float = K, gamma: float = GAMMA) -> dict:
    y = test.y.to_numpy()
    w = ze_weights(test.ze, gamma, mean_ze)
    thr = bayes_threshold(k)
    n = max(1, int(TOP_SHARE * len(y)))
    out = {}
    for name, m in models.items():
        p = m.predict_proba(test)
        r = summarize(y, p)
        r["cost"] = mean_cost(y, p >= thr, w, k)
        r["wrecall_top10"] = weighted_recall_at_n(y, p, w, n, k)
        out[name] = r
    out["_baseline_cost_none"] = {"cost": mean_cost(y, np.zeros(len(y), bool), w, k)}
    out["_baseline_cost_all"] = {"cost": mean_cost(y, np.ones(len(y), bool), w, k)}
    return out


def run_seed(seed: int, n_students: int = 2000, weeks=(4, 8), scenario: str = "S0", k: float = K, gamma: float = GAMMA):
    """Один мир → для каждой недели: словарь моделей, тест и результаты."""
    w = simulate_world(n_students, seed=seed, scenario=scenario)
    mean_ze = float(w.curriculum.ze.mean())
    res = {}
    for t in weeks:
        d = build_dataset(w, weeks=(t,))
        parts = split_students(d.student_id.unique(), seed=seed)
        tr, va, te = (take(d, parts[s]) for s in ("train", "val", "test"))
        models = fit_all(tr, va, mean_ze, k, gamma, seed)
        res[t] = {"models": models, "test": te, "metrics": evaluate(models, te, mean_ze, k, gamma)}
    return res, mean_ze
