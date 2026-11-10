"""Симулятор учебной когорты с известной причинной моделью.

Зачем он нужен: на реальных данных нельзя проверить, находит ли модель именно ту
закономерность, которая есть. Здесь причина провала задана явно, поэтому видно,
что модель и объяснения должны найти. Это не замена реальным данным: результаты на
симуляторе доказывают работоспособность конвейера, а не качество на живых студентах.

Модель мира (все параметры в SimConfig):
  * способности a ~ N(0,1); балл ЕГЭ растёт с a, но с большим шумом;
  * мотивация m ~ N(0,1);
  * вовлечённость e(t) = m + шум AR(1) - спад после момента «потери интереса»
    (у части студентов момент и скорость спада случайны);
  * логины и минуты в LMS зависят от e(t); контрольные баллы зависят от a и e(t);
  * студент может бросить учёбу: недельный риск растёт при низкой e(t);
  * исход (провал курса) зависит от a и от вовлечённости ВО ВСЕЙ второй половине
    семестра, то есть частично определяется «будущим» относительно ранних недель.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimConfig:
    n_weeks: int = 16
    checkpoint_weeks: tuple = (2, 4, 6, 8, 10, 12, 14, 16)   # номера недель, с 1
    pass_score: float = 60.0
    disengage_share: float = 0.35      # доля студентов с «потерей интереса»
    onset_range: tuple = (4, 12)       # в какие недели она может начаться (включительно)
    decay_range: tuple = (0.35, 0.8)   # скорость спада вовлечённости за неделю
    ar_coef: float = 0.6
    ar_sigma: float = 0.35
    cp_base: float = 73.0              # средний балл контрольной при a = 0, e = 0
    dropout_base: float = -6.3         # логит недельного риска ухода при e = 0
    dropout_slope: float = -1.0
    fail_threshold: float = -2.0       # порог итоговой «успеваемости»


@dataclass
class Cohort:
    """Наблюдаемые данные когорты. Скрытые причины лежат отдельно в `truth`."""
    students: pd.DataFrame                # индекс student_id; статические признаки
    logins: np.ndarray                    # (n, W) целые
    minutes: np.ndarray                   # (n, W)
    cp_score: np.ndarray                  # (n, W): балл или NaN (нет контрольной / пропущена)
    cp_scheduled: np.ndarray              # (W,) bool: в эту неделю есть контрольная точка
    pass_score: float
    outcome: np.ndarray                   # (n,) bool: True = провалил / ушёл
    truth: pd.DataFrame = field(default=None, repr=False)   # скрытые переменные; в признаки НЕ попадают

    @property
    def n_students(self) -> int:
        return self.logins.shape[0]

    @property
    def n_weeks(self) -> int:
        return self.logins.shape[1]

    def cp_failed(self) -> np.ndarray:
        """(n, W) bool: контрольная была назначена и не сдана на проходной балл (пропуск = провал)."""
        sched = self.cp_scheduled[None, :]
        passed = np.nan_to_num(self.cp_score, nan=-1.0) >= self.pass_score
        return sched & ~passed

    def observed(self, week: int) -> "Cohort":
        """Копия, в которой остались только недели 1..week. Всё, что считается по такому
        срезу, физически не может использовать будущее."""
        if not 1 <= week <= self.n_weeks:
            raise ValueError(f"week должна быть от 1 до {self.n_weeks}, получено {week}")
        return Cohort(
            students=self.students, logins=self.logins[:, :week].copy(),
            minutes=self.minutes[:, :week].copy(), cp_score=self.cp_score[:, :week].copy(),
            cp_scheduled=self.cp_scheduled[:week].copy(), pass_score=self.pass_score,
            outcome=self.outcome, truth=None)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_students: int = 1000, seed: int = 0, cfg: SimConfig = SimConfig()) -> Cohort:
    if n_students < 2:
        raise ValueError("нужно хотя бы 2 студента")
    rng = np.random.default_rng(seed)
    n, W = n_students, cfg.n_weeks

    ability = rng.normal(size=n)
    motivation = rng.normal(size=n)
    ege = np.clip(np.round(62 + 9 * ability + rng.normal(0, 9, n)), 30, 100)
    workload = rng.choice([24, 27, 30, 33], size=n, p=[0.2, 0.3, 0.35, 0.15])
    has_job = rng.random(n) < (0.18 + 0.10 * (motivation < -0.5))
    gender = rng.choice(["f", "m"], size=n)
    origin = rng.choice(["local", "other_region"], size=n, p=[0.6, 0.4])

    # --- вовлечённость
    disengaging = rng.random(n) < cfg.disengage_share
    onset = rng.integers(cfg.onset_range[0], cfg.onset_range[1] + 1, size=n)
    decay = rng.uniform(*cfg.decay_range, size=n)
    weeks = np.arange(1, W + 1)[None, :]
    drop_part = np.where(disengaging[:, None], decay[:, None] * np.clip(weeks - onset[:, None], 0, None), 0.0)
    noise = np.zeros((n, W))
    for t in range(W):
        prev = noise[:, t - 1] if t else 0.0
        noise[:, t] = cfg.ar_coef * prev + rng.normal(0, cfg.ar_sigma, n)
    e = motivation[:, None] - drop_part + noise

    # --- уход из курса
    hazard = _sigmoid(cfg.dropout_base + cfg.dropout_slope * e)
    hazard[:, :3] = 0.0
    leaves = rng.random((n, W)) < hazard
    left_at = np.where(leaves.any(1), leaves.argmax(1), W)        # индекс первой недели без активности
    active = np.arange(W)[None, :] < left_at[:, None]
    dropped = left_at < W

    # --- наблюдаемая активность
    logins = rng.poisson(np.exp(1.2 + 0.6 * e)) * active
    minutes = logins * rng.lognormal(np.log(20) + 0.2 * e, 0.4)   # после ухода logins = 0
    minutes = np.round(minutes, 1)

    # --- контрольные точки
    sched = np.zeros(W, dtype=bool)
    for w in cfg.checkpoint_weeks:
        if not 1 <= w <= W:
            raise ValueError(f"контрольная неделя {w} вне 1..{W}")
        sched[w - 1] = True
    raw = np.clip(cfg.cp_base + 10 * ability[:, None] + 9 * e + rng.normal(0, 8, (n, W)), 0, 100)
    cp_score = np.where(sched[None, :] & active, np.round(raw, 1), np.nan)

    # --- исход
    half = W // 2
    perf = (0.5 * ability + 0.8 * e[:, half:].mean(1) + 0.3 * e[:, :half].mean(1)
            + rng.normal(0, 0.5, n))
    outcome = dropped | (perf < cfg.fail_threshold)

    students = pd.DataFrame({"ege": ege, "workload_ze": workload, "has_job": has_job.astype(int),
                             "gender": gender, "origin": origin},
                            index=pd.RangeIndex(n, name="student_id"))
    truth = pd.DataFrame({"ability": ability, "motivation": motivation, "disengaging": disengaging,
                          "onset": onset, "dropped": dropped, "left_at": left_at},
                         index=students.index)
    return Cohort(students, logins.astype(int), minutes, cp_score, sched, cfg.pass_score,
                  outcome.astype(bool), truth)
