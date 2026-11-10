"""Протокол этапа 5: T2 (оценка 2–5 для дифзачётов, экзаменов и курсовых) и T3 (допуск)."""
from __future__ import annotations

import numpy as np

from . import conformal as cf
from .costs import ze_weights
from .dataset import add_targets, build_dataset, split_students, take
from .metrics import summarize
from .models import BoostModel, Calibrated, LogitModel
from .ordinal import (GRADES, FrankHall, Multinomial, PriorModel, class_metrics, cost_matrix, decide_argmax,
                      decide_min_cost, expected_cost)
from .world import simulate_world

ASYM, GAMMA, ALPHA = 3.0, 2.0, 0.10
T2_MODELS = ["prior", "multi_logit", "multi_boost", "fh_logit", "fh_boost", "fh_boost_ze"]


def make_t2(d):
    """Строки оцениваемых форм (дифзачёт, экзамен, курсовая): у зачётов оценки нет."""
    return d[d.grade.notna()].assign(grade=lambda x: x.grade.astype(int)).reset_index(drop=True)


def fit_t2(train, val, mean_ze, gamma=GAMMA, seed=0) -> dict:
    g, gv = train.grade.to_numpy(), val.grade.to_numpy()
    wz = ze_weights(train.ze, gamma, mean_ze)
    fh_l = FrankHall(lambda: LogitModel()).fit(train, g).fit_calibration(val, gv)
    fh_b = FrankHall(lambda: BoostModel(seed)).fit(train, g).fit_calibration(val, gv)
    fh_bz = FrankHall(lambda: BoostModel(seed)).fit(train, g, sample_weight=wz).fit_calibration(val, gv)
    return {"prior": PriorModel().fit(train, g), "multi_logit": Multinomial("logit").fit(train, g),
            "multi_boost": Multinomial("boost", seed).fit(train, g),
            "fh_logit": fh_l, "fh_boost": fh_b, "fh_boost_ze": fh_bz}


def eval_t2(models, test, mean_ze, asym=ASYM, gamma=GAMMA) -> dict:
    y = test.grade.to_numpy()
    w = ze_weights(test.ze, gamma, mean_ze)
    M = cost_matrix(asym)
    out = {}
    for name, m in models.items():
        P = m.predict_proba(test)
        r = {"nll": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), np.searchsorted(GRADES, y)], 1e-6, 1)))),
             **{f"argmax_{k}": v for k, v in class_metrics(y, decide_argmax(P)).items()},
             "cost_argmax": expected_cost(y, decide_argmax(P), w, M),
             "cost_mincost": expected_cost(y, decide_min_cost(P, M), w, M),
             "recall_2_mincost": float(np.mean(decide_min_cost(P, M)[y == 2] == 2)),
             "qwk_mincost": class_metrics(y, decide_min_cost(P, M))["qwk"]}
        r["auc_2"] = summarize((y == 2).astype(int), P[:, 0])["roc_auc"]
        out[name] = r
    return out


def conformal_eval(model, val, test, alpha=ALPHA) -> dict:
    """Конформные наборы: порог по val, покрытие и размер на test; общий и по классам."""
    Pv, Pt = model.predict_proba(val), model.predict_proba(test)
    yv, yt = np.searchsorted(GRADES, val.grade.to_numpy()), np.searchsorted(GRADES, test.grade.to_numpy())
    out = {}
    for name, cw in (("marginal", False), ("classwise", True)):
        S = cf.prediction_sets(Pt, cf.calibrate(Pv, yv, alpha, cw))
        out[name] = {"coverage": cf.coverage(S, yt), "size": cf.mean_size(S),
                     "coverage_by_class": [float(S[yt == c, c].mean()) for c in range(4)]}
    return out


def fit_t3(train, val, seed=0) -> dict:
    ytr, yva = 1 - train.admitted.to_numpy(), 1 - val.admitted.to_numpy()      # 1 = недопуск
    logit = LogitModel().fit(train, ytr)
    boost = BoostModel(seed).fit(train, ytr)
    return {"logit": logit, "boost+cal": Calibrated(boost).fit_calibration(val, yva)}


def eval_t3(models, test) -> dict:
    y = 1 - test.admitted.to_numpy()
    return {n: summarize(y, m.predict_proba(test)) for n, m in models.items()}


def run_seed5(seed: int, n_students: int = 2000, weeks=(4, 8, 12), scenario="S0", asym=ASYM, gamma=GAMMA):
    w = simulate_world(n_students, seed=seed, scenario=scenario)
    mean_ze = float(w.curriculum.ze.mean())
    res = {}
    for t in weeks:
        d = add_targets(build_dataset(w, weeks=(t,)), w)
        parts = split_students(d.student_id.unique(), seed=seed)
        r = {}
        # T3 на всех формах
        tr, va, te = (take(d, parts[s]) for s in ("train", "val", "test"))
        m3 = fit_t3(tr, va, seed)
        r["t3"] = {"models": m3, "test": te, "metrics": eval_t3(m3, te)}
        # T2 только на оцениваемых формах
        g = make_t2(d)
        tr, va, te = (take(g, parts[s]) for s in ("train", "val", "test"))
        m2 = fit_t2(tr, va, mean_ze, gamma, seed)
        r["t2"] = {"models": m2, "val": va, "test": te, "metrics": eval_t2(m2, te, mean_ze, asym, gamma),
                   "conformal": conformal_eval(m2["fh_boost"], va, te)}
        res[t] = r
    return res, mean_ze
