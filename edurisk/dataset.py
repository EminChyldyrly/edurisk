"""Обучающая выборка «студент × дисциплина × неделя» из World и разбиение по студентам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curriculum import FORMS
from .features import FEATURES, build_features
from .world import World

FORM_COLS = [f"form_{f}" for f in FORMS]
HISTORY = ["prior_gpa", "prior_fails"]
COURSE = ["ze"] + FORM_COLS
MODEL_FEATURES = FEATURES + HISTORY + COURSE


def build_dataset(w: World, weeks=(4, 8), courses=None) -> pd.DataFrame:
    """Строка = студент в дисциплине на конец недели t. Признаки считаются по `course_view(c).observed(t)`,
    история — только по итогам прошлых семестров (`history_before`), метка = провал с первой попытки.
    Метка не входит в признаки; в `ze` хранится трудоёмкость дисциплины (известна заранее из плана)."""
    cur = w.curriculum
    hist = {s: w.history_before(s) for s in sorted(cur.semester.unique())}
    out = []
    for c in (range(w.n_courses) if courses is None else courses):
        coh = w.course_view(c)
        sem, form, ze = int(cur.semester[c]), cur.form[c], int(cur.ze[c])
        for t in weeks:
            X = build_features(coh, t)
            h = hist[sem].loc[X.index, HISTORY]
            X = X.join(h)
            X["ze"] = ze
            for f in FORMS:
                X[f"form_{f}"] = int(form == f)
            X["course"], X["semester"], X["week"] = c, sem, t
            X["y"] = coh.outcome.astype(int)
            out.append(X.rename_axis("student_id").reset_index())
    return pd.concat(out, ignore_index=True)


def split_students(student_ids, seed: int = 0, fractions=(0.6, 0.2, 0.2)) -> dict:
    """Случайное разбиение СТУДЕНТОВ на train/val/test: один студент целиком в одной части."""
    if abs(sum(fractions) - 1) > 1e-9 or min(fractions) <= 0:
        raise ValueError("доли должны быть положительными и давать в сумме 1")
    ids = np.unique(np.asarray(student_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    a = int(round(fractions[0] * len(ids)))
    b = a + int(round(fractions[1] * len(ids)))
    return {"train": set(ids[:a].tolist()), "val": set(ids[a:b].tolist()), "test": set(ids[b:].tolist())}


def take(df: pd.DataFrame, ids: set) -> pd.DataFrame:
    return df[df.student_id.isin(ids)].reset_index(drop=True)


def add_targets(d: pd.DataFrame, w: World) -> pd.DataFrame:
    """Добавляет целевые T2 (grade: 2–5, NaN у зачётов) и T3 (admitted). Цели не входят в MODEL_FEATURES."""
    lab = w.labels()[["student_id", "course", "grade", "admitted"]]
    out = d.merge(lab, on=["student_id", "course"], how="left", validate="many_to_one")
    out["admitted"] = out["admitted"].astype(int)
    return out
