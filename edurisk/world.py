"""Многодисциплинарный симулятор (этап 2): студенты проходят семестры подряд, в каждом несколько
дисциплин с разными формами контроля. Причинная модель известна и лежит в `truth`.

Что нового по сравнению с одной дисциплиной (simulate.py):
  * у студента скрытое состояние переносится между семестрами (мотивация дрейфует);
  * вовлечённость в семестре общая для дисциплин + собственный «интерес» к каждой дисциплине;
  * ОБРАТНАЯ СВЯЗЬ: проваленные контрольные точки снижают вовлечённость дальше (сценарий S1 —
    порогово: после двух подряд провалов, «выученная беспомощность»);
  * допуск к промежуточной аттестации (не менее половины контрольных точек сдано), недопуск = «2»;
  * отчисление: студент уходит из ВСЕХ следующих дисциплин;
  * пререквизиты: успех в предшествующей дисциплине помогает в следующей;
  * итог первой попытки (это и есть метка) + отдельно пересдача (в метку не попадает).

Сценарии (`scenario`):
  S0 гладкий базовый; S1 пороговая обратная связь; S2 сдвиг когорты (cohort_year=1: другая связь
  активности с успехом); S3 смещение измерения по группе (студенты из других регионов реже заходят в LMS
  при той же вовлечённости, а на успеваемость это не влияет — проверка справедливости);
  S4 потери данных (часть недель логинов и часть оценок контрольных не записана).

Метка не утекает в признаки: метка — результат первой попытки в сессии, которая идёт после семестра.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .curriculum import CurriculumConfig, GRADED, build_curriculum, checkpoint_mask
from .simulate import Cohort

SCENARIOS = ("S0", "S1", "S2", "S3", "S4")


@dataclass(frozen=True)
class WorldConfig:
    n_weeks: int = 16
    pass_score: float = 60.0
    disengage_share: float = 0.20
    onset_range: tuple = (4, 12)
    decay_range: tuple = (0.35, 0.8)
    ar_coef: float = 0.6
    ar_sigma: float = 0.35
    course_ar: float = 0.5
    course_sigma: float = 0.25
    motivation_carry: float = 0.8          # доля мотивации, переходящая в следующий семестр
    cp_base: float = 80.0
    dropout_base: float = -9.3
    dropout_slope: float = -1.0
    dropout_from_week: int = 4
    feedback_linear: float = 0.10          # S0: падение вовлечённости за каждую проваленную КТ
    feedback_jump: float = 1.0             # S1: скачок при серии из двух провалов подряд
    feedback_floor: float = -2.5
    admit_share: float = 0.4               # допуск: доля сданных КТ не меньше
    grade_cuts: tuple = (-3.1, -1.06, 0.67)  # границы скрытой успеваемости для оценок 2|3|4|5
    zachet_cut: float = -3.3
    retake_bonus: float = 0.5
    retake_sigma: float = 0.5
    s3_login_factor: float = 0.75          # S3: во сколько раз реже заходят студенты из других регионов
    s2_cp_shift: float = -4.0              # S2: сдвиг базового балла в когорте 1
    s2_perf_shift: float = -0.5            # S2: когорта 1 сдаёт итог тяжелее (строже требования)
    s2_activity_loading: float = 0.3       # S2: связь активности с вовлечённостью в когорте 1 (было 0.6)
    s4_login_loss: float = 0.15
    s4_score_loss: float = 0.10


@dataclass
class World:
    """Данные нескольких семестров. Оси: n студентов, C курсов (всего в плане), W недель."""
    students: pd.DataFrame
    curriculum: pd.DataFrame
    logins: np.ndarray             # (n, C, W) float, NaN = запись потеряна (S4) или студент не учится
    minutes: np.ndarray            # (n, C, W)
    cp_score: np.ndarray           # (n, C, W): балл; NaN = не сдана / нет записи
    cp_scheduled: np.ndarray       # (C, W) bool
    enrolled: np.ndarray           # (n, C) bool: студент учился на момент начала семестра дисциплины
    dropped: np.ndarray            # (n, C) bool: отчислен в ходе семестра этой дисциплины
    admitted: np.ndarray           # (n, C) bool: допущен (для зачёта — выполнил условия)
    passed: np.ndarray             # (n, C) bool: итог первой попытки (False для не учившихся — не используется)
    grade: np.ndarray              # (n, C) float: 2..5 для оцениваемых форм, иначе NaN
    retake_passed: np.ndarray      # (n, C) float: 1/0 после пересдачи, NaN если пересдачи не было
    pass_score: float
    scenario: str
    truth: dict = field(default=None, repr=False)

    @property
    def n_students(self) -> int:
        return self.logins.shape[0]

    @property
    def n_courses(self) -> int:
        return self.logins.shape[1]

    @property
    def n_weeks(self) -> int:
        return self.logins.shape[2]

    def course_view(self, c: int, lost: str = "zero") -> Cohort:
        """Дисциплина c как одноклассная когорта (формат этапа 1): только учившиеся студенты.
        Метка `outcome` = «не прошёл с первой попытки» (незачёт, 2, недопуск, отчисление).
        `lost`: как заполнить потерянные записи сценария S4: 'zero' (нули) — признаки этапа 1
        не отличают потерю от неактивности, это учтено на этапе 4."""
        if lost != "zero":
            raise ValueError("пока поддерживается только lost='zero'")
        m = self.enrolled[:, c]
        L = np.nan_to_num(self.logins[m, c, :], nan=0.0)
        M = np.nan_to_num(self.minutes[m, c, :], nan=0.0)
        return Cohort(students=self.students.loc[m], logins=L.astype(int), minutes=M,
                      cp_score=self.cp_score[m, c, :].copy(), cp_scheduled=self.cp_scheduled[c].copy(),
                      pass_score=self.pass_score, outcome=~self.passed[m, c],
                      truth=None if self.truth is None else pd.DataFrame(
                          {k: v[m] for k, v in self.truth["student"].items()}, index=self.students.index[m]))

    def history_before(self, semester: int) -> pd.DataFrame:
        """Что известно о студенте к началу семестра: только итоги ПРЕДЫДУЩИХ семестров.
        Колонки: prior_gpa (средняя оценка по оцениваемым, взвешенная З.Е.; NaN если нет),
        prior_fails (число недопройденных с первой попытки), prior_ze (З.Е. пройденных курсов)."""
        cur = self.curriculum
        idx = np.flatnonzero(cur.semester.to_numpy() < semester)
        out = pd.DataFrame(index=self.students.index)
        if len(idx) == 0:
            out["prior_gpa"] = np.nan
            out["prior_fails"] = 0
            out["prior_ze"] = 0
            return out
        ze = cur.ze.to_numpy()[idx][None, :].astype(float)
        enr = self.enrolled[:, idx]
        g = self.grade[:, idx]
        w = np.where(enr & ~np.isnan(g), ze, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["prior_gpa"] = np.where(w.sum(1) > 0, (np.nan_to_num(g) * w).sum(1) / w.sum(1), np.nan)
        out["prior_fails"] = (enr & ~self.passed[:, idx]).sum(1)
        out["prior_ze"] = np.where(enr & self.passed[:, idx], ze, 0).sum(1).astype(int)
        return out

    def labels(self) -> pd.DataFrame:
        """Длинная таблица «студент × дисциплина» с целевыми: fail (T1), grade (T2), admitted (T3)."""
        n, C = self.enrolled.shape
        s, c = np.nonzero(self.enrolled)
        cur = self.curriculum
        return pd.DataFrame({
            "student_id": self.students.index.to_numpy()[s], "course": c,
            "semester": cur.semester.to_numpy()[c], "form": cur.form.to_numpy()[c],
            "ze": cur.ze.to_numpy()[c],
            "fail": ~self.passed[s, c], "grade": self.grade[s, c],
            "admitted": self.admitted[s, c], "dropped": self.dropped[s, c],
            "retake_passed": self.retake_passed[s, c]})


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _grade_from_perf(p: np.ndarray, cuts) -> np.ndarray:
    """Скрытая успеваемость → оценка 2..5 по трём порогам."""
    return 2 + (p >= cuts[0]).astype(int) + (p >= cuts[1]) + (p >= cuts[2])


def simulate_world(n_students: int = 1000, seed: int = 0, curriculum: pd.DataFrame | None = None,
                   cfg: WorldConfig = WorldConfig(), scenario: str = "S0", cohort_year: int = 0) -> World:
    if scenario not in SCENARIOS:
        raise ValueError(f"сценарий должен быть одним из {SCENARIOS}, получено {scenario!r}")
    if n_students < 2:
        raise ValueError("нужно хотя бы 2 студента")
    cur = curriculum if curriculum is not None else build_curriculum(CurriculumConfig(), seed=0)
    rng = np.random.default_rng(seed)
    n, W, C = n_students, cfg.n_weeks, len(cur)
    n_sem = int(cur.semester.max()) + 1

    # --- параметры, зависящие от сценария
    cp_base, act_load, perf_shift = cfg.cp_base, 0.6, 0.0
    if scenario == "S2" and cohort_year == 1:
        cp_base += cfg.s2_cp_shift
        act_load = cfg.s2_activity_loading
        perf_shift = cfg.s2_perf_shift

    # --- студенты
    ability = rng.normal(size=n)
    mot = rng.normal(size=n)
    ege = np.clip(np.round(62 + 9 * ability + rng.normal(0, 9, n)), 30, 100)
    workload = rng.choice([24, 27, 30, 33], size=n, p=[0.2, 0.3, 0.35, 0.15])
    has_job = rng.random(n) < (0.18 + 0.10 * (mot < -0.5))
    gender = rng.choice(["f", "m"], size=n)
    origin = rng.choice(["local", "other_region"], size=n, p=[0.6, 0.4])
    login_factor = np.where((scenario == "S3") & (origin == "other_region"), cfg.s3_login_factor, 1.0)

    logins = np.full((n, C, W), np.nan, dtype=np.float32)
    minutes = np.full((n, C, W), np.nan, dtype=np.float32)
    cp_score = np.full((n, C, W), np.nan, dtype=np.float32)
    sched_all = np.stack([checkpoint_mask(f, W) for f in cur.form])
    enrolled = np.zeros((n, C), bool)
    dropped_in = np.zeros((n, C), bool)
    admitted = np.zeros((n, C), bool)
    passed = np.zeros((n, C), bool)
    grade = np.full((n, C), np.nan)
    retake = np.full((n, C), np.nan)
    perf_all = np.zeros((n, C))
    alive = np.ones(n, bool)
    left_sem = np.full(n, -1)
    left_week = np.full(n, -1)
    disengaging_s = np.zeros((n, n_sem), bool)
    onset_s = np.zeros((n, n_sem), int)
    mot_s = np.zeros((n, n_sem))
    fb_hist = np.zeros((n, n_sem, W))

    weeks1 = np.arange(1, W + 1)
    for s in range(n_sem):
        cs = np.flatnonzero(cur.semester.to_numpy() == s)
        K = len(cs)
        if s > 0:
            mot = cfg.motivation_carry * mot + np.sqrt(1 - cfg.motivation_carry ** 2) * rng.normal(size=n)
        mot_s[:, s] = mot
        dis = rng.random(n) < cfg.disengage_share
        onset = rng.integers(cfg.onset_range[0], cfg.onset_range[1] + 1, size=n)
        decay = rng.uniform(*cfg.decay_range, size=n)
        disengaging_s[:, s], onset_s[:, s] = dis, onset
        interest = rng.normal(size=(n, K))
        diff = cur.difficulty.to_numpy()[cs]
        sched = sched_all[cs]                                   # (K, W)
        enrolled[:, cs] = alive[:, None]
        start_alive = alive.copy()

        f = np.zeros(n)                                         # накопленный эффект обратной связи
        noise = np.zeros(n)
        cn = np.zeros((n, K))
        fail_cnt = np.zeros((n, K))
        streak = np.zeros((n, K), int)
        e_hist = np.zeros((n, K, W))
        for t in range(W):
            noise = cfg.ar_coef * noise + rng.normal(0, cfg.ar_sigma, n)
            drop_part = np.where(dis, decay * np.clip(t + 1 - onset, 0, None), 0.0)
            e_stu = mot - drop_part + noise + f - 0.25 * has_job
            cn = cfg.course_ar * cn + rng.normal(0, cfg.course_sigma, (n, K))
            e_c = e_stu[:, None] + cn + 0.3 * interest
            e_hist[:, :, t] = e_c

            # отчисление: уход виден со следующей активности (неделя ухода уже без активности)
            if t + 1 >= cfg.dropout_from_week:
                leave = alive & (rng.random(n) < _sigmoid(cfg.dropout_base + cfg.dropout_slope * e_stu))
            else:
                rng.random(n)                                   # поток случайных чисел не зависит от условия
                leave = np.zeros(n, bool)
            alive = alive & ~leave
            newly = leave & start_alive
            left_sem[newly & (left_sem < 0)] = s
            left_week[newly & (left_week < 0)] = t
            act = alive[:, None]

            lam = np.exp(1.2 + act_load * e_c) * login_factor[:, None]
            lg = rng.poisson(lam) * act
            mn = np.round(lg * rng.lognormal(np.log(20) + 0.2 * e_c, 0.4), 1)
            logins[:, cs, t] = np.where(start_alive[:, None], lg, np.nan)
            minutes[:, cs, t] = np.where(start_alive[:, None], mn, np.nan)

            # контрольные точки
            raw = np.clip(cp_base + 10 * ability[:, None] - 4 * diff[None, :] + 9 * e_c
                          + rng.normal(0, 8, (n, K)), 0, 100)
            is_cp = sched[:, t][None, :] & act
            cp_score[:, cs, t] = np.where(is_cp, np.round(raw, 1), np.nan)
            failed = is_cp & (raw < cfg.pass_score)
            new_streak = np.where(failed, streak + 1, np.where(is_cp, 0, streak))
            hit2 = failed & (new_streak == 2)
            streak = new_streak
            fail_cnt += failed
            if scenario == "S1":
                f = f - cfg.feedback_jump * hit2.any(1) - 0.03 * failed.sum(1)
            else:
                f = f - cfg.feedback_linear * failed.sum(1)
            f = np.maximum(f, cfg.feedback_floor)
            fb_hist[:, s, t] = f

        # --- итоги семестра
        n_cp = sched.sum(1)
        # допуск считается по настоящим баллам (потери записей S4 накладываются после цикла)
        true_cp = cp_score[:, cs, :]
        passed_cp = ((true_cp >= cfg.pass_score) & sched[None, :, :]).sum(2)
        share = passed_cp / n_cp[None, :]
        dropped_here = start_alive & ~alive
        dropped_in[:, cs] = dropped_here[:, None] & enrolled[:, cs]
        adm = (share >= cfg.admit_share) & start_alive[:, None] & ~dropped_here[:, None]

        e1 = e_hist[:, :, : W // 2].mean(2)
        e2 = e_hist[:, :, W // 2:].mean(2)
        prereq = cur.prereq.to_numpy()[cs]
        pre = np.zeros((n, K))
        for k, p in enumerate(prereq):
            if p >= 0:
                pre[:, k] = perf_all[:, p]
        perf = (0.45 * ability[:, None] + 0.75 * e2 + 0.3 * e1 - 0.35 * diff[None, :] + 0.15 * pre
                + perf_shift + rng.normal(0, 0.5, (n, K)))
        perf_all[:, cs] = np.where(start_alive[:, None], perf, 0.0)

        forms = cur.form.to_numpy()[cs]
        is_graded = np.isin(forms, GRADED)[None, :]
        g_first = _grade_from_perf(perf, cfg.grade_cuts)
        ok_graded = adm & (g_first >= 3)
        ok_zachet = adm & (perf >= cfg.zachet_cut)
        ok = np.where(is_graded, ok_graded, ok_zachet)
        gfin = np.where(adm, g_first, 2)                       # недопуск и отчисление = «2»
        grade[:, cs] = np.where(is_graded & start_alive[:, None], gfin, np.nan)
        passed[:, cs] = np.where(start_alive[:, None], ok, False)
        admitted[:, cs] = adm

        # пересдача для не отчисленных, не прошедших с первой попытки
        need = start_alive[:, None] & ~ok & ~dropped_here[:, None]
        p2 = perf + cfg.retake_bonus + rng.normal(0, cfg.retake_sigma, (n, K))
        ok2 = np.where(is_graded, _grade_from_perf(p2, cfg.grade_cuts) >= 3, p2 >= cfg.zachet_cut)
        retake[:, cs] = np.where(need, ok2.astype(float), np.nan)

    # --- потери данных (S4): теряются ЗАПИСИ, исход остаётся настоящим
    if scenario == "S4":
        lost_l = rng.random(logins.shape) < cfg.s4_login_loss
        lost_s = rng.random(cp_score.shape) < cfg.s4_score_loss
        logins[lost_l] = np.nan
        minutes[lost_l] = np.nan
        cp_score[lost_s & ~np.isnan(cp_score)] = np.nan

    students = pd.DataFrame({"ege": ege, "workload_ze": workload, "has_job": has_job.astype(int),
                             "gender": gender, "origin": origin},
                            index=pd.RangeIndex(n, name="student_id"))
    truth = {"student": {"ability": ability, "motivation0": mot_s[:, 0], "left_semester": left_sem,
                         "left_week": left_week},
             "motivation": mot_s, "disengaging": disengaging_s, "onset": onset_s, "perf": perf_all,
             "feedback": fb_hist}
    return World(students, cur, logins, minutes, cp_score, sched_all, enrolled, dropped_in, admitted, passed,
                 grade, retake, cfg.pass_score, scenario, truth)
