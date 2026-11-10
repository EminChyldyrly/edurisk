"""Адаптер открытого набора OULAD (Open University Learning Analytics Dataset) к формату `World`.

ВАЖНО: написан по описанию набора (имена файлов и колонок ниже), но на настоящих файлах НЕ запускался —
их нет в среде разработки. Тесты идут на маленьком синтетическом наборе в том же формате. Перед
использованием прогоните `python scripts/load_oulad.py <папка>` у себя и сверьте сводку с ожидаемой.
Лицензию набора (насколько мне известно, CC BY 4.0) проверьте на странице набора и укажите в работе.

Соответствие понятий:
  модуль + семестр презентации -> дисциплина (form='zachet': итог «сдал / не сдал», оценки нет);
  Withdrawn, Fail -> провал (fail), Pass, Distinction -> успех; Withdrawn -> dropped;
  клики в VLE -> logins = число дней активности за неделю, minutes = сумма кликов (это КЛИКИ, не минуты);
  контрольные точки = ТМА и СМА со сроком в первые n_weeks недель (экзамены исключены: итог);
  балл засчитывается, если сдано не позже срока и не перенесено из прошлой попытки (is_banked);
  студенческие поля: studied_credits -> workload_ze; ege и has_job неизвестны (NaN и 0).
Неделя = date // 7 + 1, где date — день от начала презентации; даты до начала отбрасываются.
Проходной балл по заданию в OULAD равен 40.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .world import World

FILES = ["courses.csv", "studentInfo.csv", "studentVle.csv", "assessments.csv", "studentAssessment.csv"]
PASS_SCORE = 40.0


def _read(d: pathlib.Path, name: str, cols: list[str]) -> pd.DataFrame:
    p = d / name
    if not p.exists():
        raise FileNotFoundError(f"нет файла {name} в {d}")
    df = pd.read_csv(p)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"в {name} нет колонок {missing}")
    return df


def load_oulad(folder: str, n_weeks: int = 16) -> World:
    d = pathlib.Path(folder)
    courses = _read(d, "courses.csv", ["code_module", "code_presentation"])
    info = _read(d, "studentInfo.csv", ["code_module", "code_presentation", "id_student", "gender", "region",
                                        "studied_credits", "final_result"])
    vle = _read(d, "studentVle.csv", ["code_module", "code_presentation", "id_student", "date", "sum_click"])
    asm = _read(d, "assessments.csv", ["code_module", "code_presentation", "id_assessment", "assessment_type", "date"])
    sa = _read(d, "studentAssessment.csv", ["id_assessment", "id_student", "date_submitted", "is_banked", "score"])

    pres = sorted(courses.code_presentation.unique())
    keys = sorted(set(zip(courses.code_module, courses.code_presentation)), key=lambda k: (pres.index(k[1]), k[0]))
    cidx = {k: i for i, k in enumerate(keys)}
    C, W = len(keys), n_weeks

    ids = np.sort(info.id_student.unique())
    row = {int(i): k for k, i in enumerate(ids)}
    n = len(ids)

    first = info.drop_duplicates("id_student").set_index("id_student").loc[ids]
    students = pd.DataFrame({
        "ege": np.nan, "workload_ze": first.studied_credits.astype(float).to_numpy(), "has_job": 0,
        "gender": first.gender.map({"M": "m", "F": "f"}).fillna("?").to_numpy(),
        "origin": first.region.fillna("?").to_numpy()}, index=pd.Index(ids, name="student_id"))
    curriculum = pd.DataFrame({
        "name": [k[0] for k in keys], "semester": [pres.index(k[1]) for k in keys], "form": "zachet",
        "ze": 6, "difficulty": 0.0, "prereq": -1}, index=pd.RangeIndex(C, name="course"))

    enrolled = np.zeros((n, C), bool)
    dropped, admitted, passed = (np.zeros((n, C), bool) for _ in range(3))
    info = info[[(m, p) in cidx for m, p in zip(info.code_module, info.code_presentation)]]
    r = info.id_student.map(row).to_numpy()
    c = np.array([cidx[(m, p)] for m, p in zip(info.code_module, info.code_presentation)], dtype=int)
    enrolled[r, c] = True
    wd = (info.final_result == "Withdrawn").to_numpy()
    dropped[r, c] = wd
    admitted[r, c] = ~wd
    passed[r, c] = info.final_result.isin(["Pass", "Distinction"]).to_numpy()

    logins = np.full((n, C, W), np.nan, np.float32)
    minutes = np.full((n, C, W), np.nan, np.float32)
    logins[enrolled] = 0.0
    minutes[enrolled] = 0.0
    v = vle[(vle.date >= 0) & (vle.date < 7 * W)].copy()
    v["week"] = v.date // 7
    v["c"] = [cidx.get((m, p), -1) for m, p in zip(v.code_module, v.code_presentation)]
    v = v[v.c >= 0]
    v["r"] = v.id_student.map(row)
    v = v[v.r.notna()]
    days = v.groupby(["r", "c", "week"]).date.nunique()
    clicks = v.groupby(["r", "c", "week"]).sum_click.sum()
    for (ri, ci, wk), val in days.items():
        if enrolled[int(ri), int(ci)]:
            logins[int(ri), int(ci), int(wk)] = val
            minutes[int(ri), int(ci), int(wk)] = clicks[(ri, ci, wk)]

    sched = np.zeros((C, W), bool)
    score = np.full((n, C, W), np.nan, np.float32)
    a = asm[asm.assessment_type.isin(["TMA", "CMA"]) & asm.date.notna()].copy()
    a = a[(a.date >= 0) & (a.date < 7 * W)]
    a["c"] = [cidx.get((m, p), -1) for m, p in zip(a.code_module, a.code_presentation)]
    a = a[a.c >= 0]
    a["week"] = (a.date // 7).astype(int)
    for ac, aw in zip(a.c, a.week):
        sched[int(ac), int(aw)] = True
    s = sa.merge(a[["id_assessment", "c", "week", "date"]], on="id_assessment")
    s = s[(s.is_banked == 0) & s.score.notna() & (s.date_submitted <= s.date)]
    s["r"] = s.id_student.map(row)
    s = s[s.r.notna()]
    for (ri, ci, wk), val in s.groupby(["r", "c", "week"]).score.mean().items():
        if enrolled[int(ri), int(ci)]:
            score[int(ri), int(ci), int(wk)] = val
    # назначенная точка без оценки у студента — пропуск (NaN) — уже так; вне зачисления — не назначена
    score[~enrolled] = np.nan

    grade = np.full((n, C), np.nan)
    retake = np.full((n, C), np.nan)
    return World(students, curriculum, logins, minutes, score, sched, enrolled, dropped, admitted, passed, grade,
                 retake, PASS_SCORE, "OULAD", None)


def summary(w: World) -> str:
    """Сводка для ручной сверки с описанием набора."""
    L = w.labels()
    return (f"студентов {w.n_students}, дисциплин (модуль×презентация) {w.n_courses}, зачислений {len(L)}, "
            f"провалов {L.fail.mean():.1%}, отчислений {L.dropped.mean():.1%}, "
            f"доля пустых недель активности {np.nanmean(w.logins[w.enrolled] == 0):.1%}, "
            f"контрольных точек в плане {int(w.cp_scheduled.sum())}")
