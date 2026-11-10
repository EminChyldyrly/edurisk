"""Учебный план: дисциплины, формы контроля, З.Е., расписание контрольных точек, пререквизиты.

Формы контроля в РФ — отдельные категории зачётки, поэтому и целевые переменные разные:
  zachet     — зачёт: результат «зачёт / незачёт» (оценки нет);
  difzachet  — дифференцированный зачёт: оценка 2–5;
  exam       — экзамен: оценка 2–5, нужен допуск;
  coursework — курсовая работа: оценка 2–5, вместо недельной активности — вехи (этапы сдачи).

Учебный план синтетический (названия обобщённые), структура задаётся генератором.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FORMS = ("zachet", "difzachet", "exam", "coursework")
GRADED = ("difzachet", "exam", "coursework")          # у этих форм итог — оценка 2–5

# недели (с 1) контрольных точек по форме контроля; в семестре 16 недель
CHECKPOINTS = {
    "zachet": (3, 6, 9, 12, 15),        # лабораторные / задания
    "difzachet": (4, 8, 12, 16),
    "exam": (5, 10, 14),                # рубежные контроли
    "coursework": (4, 8, 12, 15),       # вехи курсовой
}
ZE_RANGE = {"zachet": (2, 3), "difzachet": (3, 4), "exam": (4, 6), "coursework": (3, 3)}

_NAMES = ["Математический анализ", "Линейная алгебра", "Программирование", "Физика", "Дискретная математика",
          "Теория вероятностей", "Базы данных", "Алгоритмы и структуры данных", "Операционные системы",
          "Компьютерные сети", "Численные методы", "Электротехника", "Инженерная графика", "Экономика",
          "Иностранный язык", "Философия", "История", "Безопасность жизнедеятельности", "Теория управления",
          "Машинное обучение", "Метрология", "Цифровая обработка сигналов", "Проектирование ПО",
          "Теория информации", "Физическая культура"]


@dataclass(frozen=True)
class CurriculumConfig:
    n_semesters: int = 4
    prereq_prob: float = 0.5


def build_curriculum(cfg: CurriculumConfig = CurriculumConfig(), seed: int = 0) -> pd.DataFrame:
    """Таблица дисциплин. Индекс — номер курса c (0..C-1), порядок: семестр, затем позиция в семестре.

    В каждом семестре: 2 экзамена, 1 дифзачёт, 2 зачёта; со 2-го семестра ещё курсовая вместо одного зачёта.
    Колонки: name, semester, form, ze, difficulty, prereq (индекс курса или -1)."""
    if cfg.n_semesters < 1:
        raise ValueError("нужен хотя бы один семестр")
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(cfg.n_semesters):
        forms = ["exam", "exam", "difzachet", "zachet", "coursework" if s >= 1 else "zachet"]
        names = rng.choice(len(_NAMES), size=len(forms), replace=False)
        for form, nm in zip(forms, names):
            lo, hi = ZE_RANGE[form]
            rows.append(dict(name=_NAMES[nm], semester=s, form=form, ze=int(rng.integers(lo, hi + 1)),
                             difficulty=float(np.clip(rng.normal(0, 0.8), -1.6, 1.6))))
    cur = pd.DataFrame(rows)
    cur.index.name = "course"
    prereq = np.full(len(cur), -1)
    for c in range(len(cur)):
        s = cur.semester[c]
        if s > 0 and rng.random() < cfg.prereq_prob:
            prev = np.flatnonzero((cur.semester == s - 1).to_numpy() & (cur.form != "coursework").to_numpy())
            prereq[c] = int(rng.choice(prev))
    cur["prereq"] = prereq
    return cur


def checkpoint_mask(form: str, n_weeks: int = 16) -> np.ndarray:
    """(n_weeks,) bool: в какие недели у формы есть контрольная точка."""
    if form not in CHECKPOINTS:
        raise ValueError(f"неизвестная форма контроля: {form}")
    m = np.zeros(n_weeks, dtype=bool)
    for w in CHECKPOINTS[form]:
        if not 1 <= w <= n_weeks:
            raise ValueError(f"контрольная неделя {w} вне 1..{n_weeks}")
        m[w - 1] = True
    return m
