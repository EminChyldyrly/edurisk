"""Объяснимость для преподавателя (без внешних библиотек).

* Группы признаков по смыслу: активность, контрольные, история, исходные данные, дисциплина.
* `group_shapley`: ТОЧНЫЙ интервенционный Шепли по группам (5 игроков, 32 коалиции) для любой модели:
  значение коалиции = средний прогноз при подстановке группы значениями фоновых студентов. Выполняется
  свойство эффективности: сумма вкладов = прогноз − средний прогноз по фону (проверяется тестом).
* `linear_contributions`: точные вклады отдельных признаков для логистической модели (в лог-шансах).
* Что-если: замена группы типичными значениями фона («если бы активность была обычной»).
* Тексты: нейтральные формулировки («зона внимания»), без ярлыков про студента.
Объяснения описывают МОДЕЛЬ, а не причины в мире: связь с причинами проверяется только на симуляторе.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .dataset import MODEL_FEATURES
from .features import DERIVED, RAW, STATIC

GROUPS = {
    "activity": ["logins_total", "logins_last2", "logins_last4", "minutes_mean4", "activity_trend4", "logins_ewma",
                 "weeks_since_login", "zero_login_weeks", "activity_vs_cohort"],
    "assessments": ["cp_failed_count", "cp_missed_count", "last_score", "mean_score", "cp_fail_rate",
                    "cp_fail_streak", "cp_max_fail_streak", "score_delta"],
    "history": ["prior_gpa", "prior_fails"],
    "background": ["ege", "workload_ze", "has_job"],
    "course": ["ze", "form_zachet", "form_difzachet", "form_exam", "form_coursework"],
}
GROUP_RU = {"activity": "активность в LMS", "assessments": "контрольные точки", "history": "история прошлых семестров",
            "background": "исходные данные", "course": "особенности дисциплины"}


def _check_groups():
    flat = [f for g in GROUPS.values() for f in g]
    assert sorted(flat) == sorted(MODEL_FEATURES), "группы должны покрывать признаки модели ровно по одному разу"


_check_groups()


def _subset_predict(predict, row: pd.DataFrame, background: pd.DataFrame, cols: list) -> float:
    """Средний прогноз при подстановке столбцов `cols` фона вместо значений строки."""
    X = pd.concat([row] * len(background), ignore_index=True)
    X[cols] = background[cols].to_numpy()
    return float(np.mean(predict(X)))


def group_shapley(predict, row: pd.DataFrame, background: pd.DataFrame, groups: dict = GROUPS) -> dict:
    """Вклады групп в прогноз для одной строки `row` (DataFrame из 1 строки). Точный Шепли по коалициям.
    predict(DataFrame) -> вероятности. Возвращает {группа: вклад}; сумма = f(row) − f(фон)."""
    if len(row) != 1:
        raise ValueError("нужна ровно одна строка")
    names = list(groups)
    n = len(names)
    # v(S): группы из S берём из строки, остальные из фона => подставляем фон в дополнение к S
    cache = {}

    def v(S: frozenset) -> float:
        if S not in cache:
            swap = [c for g in names if g not in S for c in groups[g]]
            cache[S] = _subset_predict(predict, row, background, swap) if swap else float(predict(row)[0])
        return cache[S]

    out = {}
    for g in names:
        others = [x for x in names if x != g]
        tot = 0.0
        for k in range(n):
            w = 1.0 / (n * comb(n - 1, k))
            for S in combinations(others, k):
                S = frozenset(S)
                tot += w * (v(S | {g}) - v(S))
        out[g] = tot
    out["_base"] = v(frozenset())
    out["_pred"] = v(frozenset(names))
    return out


def linear_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Вклады признаков в лог-шанс для LogitModel: coef_j · стандартизованное значение (после заполнения
    медианой), без константы. Индикаторы пропусков относятся к своему признаку."""
    imp, sc, lr = model.pipe.steps[0][1], model.pipe.steps[1][1], model.pipe.steps[2][1]
    Z = sc.transform(imp.transform(X[model.cols_]))
    C = Z * lr.coef_[0][None, :]
    n_base = len(model.cols_)
    out = pd.DataFrame(C[:, :n_base], columns=model.cols_, index=X.index)
    for k, j in enumerate(imp.indicator_.features_):
        out.iloc[:, j] += C[:, n_base + k]
    return out


def what_if(predict, row: pd.DataFrame, background: pd.DataFrame, group: str, groups: dict = GROUPS) -> float:
    """Прогноз, если значения группы заменить медианой фона. Это сценарий о МОДЕЛИ, не обещание эффекта вмешательства."""
    r = row.copy()
    for c in groups[group]:
        r[c] = background[c].median()
    return float(predict(r)[0])


def group_importance(predict, X: pd.DataFrame, y, metric, groups: dict = GROUPS, seed: int = 0) -> dict:
    """Групповая перестановочная важность: падение метрики при перемешивании группы столбцов (вместе)."""
    rng = np.random.default_rng(seed)
    base = metric(y, predict(X))
    out = {}
    for g, cols in groups.items():
        Xp = X.copy()
        Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
        out[g] = float(base - metric(y, predict(Xp)))
    return out


def explain_text(row: pd.Series, contrib: dict, top: int = 2) -> str:
    """Короткое объяснение для преподавателя. Нейтрально: «зона внимания», факты, без оценок личности."""
    p, base = contrib["_pred"], contrib["_base"]
    ranked = sorted(((v, g) for g, v in contrib.items() if not g.startswith("_")), reverse=True)
    facts = []
    for v, g in ranked[:top]:
        if v <= 0.01:
            continue
        facts.append(f"{GROUP_RU[g]}: {_fact(row, g)}")
    level = "высокая" if p >= 0.5 else "повышенная" if p >= 0.25 else "обычная"
    head = f"Вероятность не пройти аттестацию с первой попытки по модели {min(p, 0.99):.0%} (в среднем по курсу {base:.0%}): зона внимания {level}."
    if not facts:
        return head + " Выраженных факторов повышения нет."
    return head + " Что повышает оценку риска: " + "; ".join(facts) + ". Это оценка модели, решение остаётся за преподавателем."


def _fact(r: pd.Series, g: str) -> str:
    if g == "assessments":
        bits = []
        if r.get("cp_max_fail_streak", 0) >= 2:
            bits.append(f"не сданы подряд контрольные: {int(r.cp_max_fail_streak)}")
        elif r.get("cp_failed_count", 0) >= 1:
            bits.append(f"не сдано контрольных: {int(r.cp_failed_count)}")
        if r.get("cp_missed_count", 0) >= 1:
            bits.append(f"пропущено: {int(r.cp_missed_count)}")
        return ", ".join(bits) or "результаты ниже типичных"
    if g == "activity":
        bits = []
        if pd.notna(r.get("activity_vs_cohort")) and r.activity_vs_cohort < 0.7:
            bits.append(f"активность за 4 недели {r.activity_vs_cohort:.0%} от средней по курсу")
        if r.get("weeks_since_login", 0) >= 2:
            bits.append(f"нет входов {int(r.weeks_since_login)} нед.")
        return ", ".join(bits) or "активность ниже типичной"
    if g == "history":
        return (f"средний балл прошлых семестров {r.prior_gpa:.1f}" if pd.notna(r.get("prior_gpa"))
                else "истории прошлых семестров нет") + (f", неудачных попыток: {int(r.prior_fails)}" if r.get("prior_fails", 0) >= 1 else "")
    if g == "background":
        return "исходные данные (например, балл ЕГЭ) ниже типичных" if pd.notna(r.get("ege")) else "исходные данные"
    return f"дисциплина с трудоёмкостью {int(r.get('ze', 0))} З.Е."
