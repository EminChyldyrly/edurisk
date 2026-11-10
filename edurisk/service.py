"""Сервис риска для интерфейса преподавателя: список по дисциплине, карточка студента, «не согласен», журнал.
Логика отделена от веб-слоя (app.py), чтобы тестироваться без сервера."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import numpy as np
import pandas as pd

from .costs import ze_weights
from .dataset import build_dataset, split_students, take
from .explain import GROUP_RU, GROUPS, explain_text, group_shapley, what_if
from .models import BoostModel, Calibrated
from .privacy import can, pseudonym
from .store import log_action
from .world import World, simulate_world

LEVELS = ((0.5, "высокая"), (0.25, "повышенная"), (0.0, "обычная"))


def level_of(p: float) -> str:
    return next(name for thr, name in LEVELS if p >= thr)


# --- подписанные токены (замена OAuth2/JWT на этапе прототипа; в продакшене — внешний провайдер входа)
def make_token(user: str, role: str, secret: bytes, ttl: int = 3600, now: float | None = None) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"u": user, "r": role, "exp": int((now or time.time()) + ttl)}).encode()).decode()
    sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def read_token(token: str, secret: bytes, now: float | None = None):
    """(user, role) или None, если подпись неверна или срок истёк."""
    try:
        payload, sig = token.rsplit(".", 1)
        good = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        d = json.loads(base64.urlsafe_b64decode(payload.encode()))
        if d["exp"] < (now or time.time()):
            return None
        return d["u"], d["r"]
    except Exception:
        return None


class RiskService:
    def __init__(self, world: World, conn, model, background: pd.DataFrame, week: int, teachers: dict,
                 secret: bytes, k: float = 3.0, gamma: float = 2.0):
        self.world, self.conn, self.model, self.bg, self.week = world, conn, model, background, week
        self.teachers, self.secret, self.k = teachers, secret, k
        d = build_dataset(world, weeks=(week,))
        d["p"] = model.predict_proba(d)
        d["w"] = ze_weights(d.ze, gamma, float(world.curriculum.ze.mean()))
        d["priority"] = d.p * d.w * k
        d["pseudo"] = [pseudonym(i, secret) for i in d.student_id]
        self.data = d
        self._by_pseudo = {pseudonym(i, secret): int(i) for i in world.students.index}

    # --- доступ
    def allowed_courses(self, user: str) -> list:
        return self.teachers.get(user, [])

    def course_name(self, c: int) -> str:
        r = self.world.curriculum.loc[c]
        return f"{r['name']} ({r.form}, {r.ze} З.Е.)"

    # --- данные
    def risk_list(self, course: int, top: int | None = None) -> pd.DataFrame:
        d = self.data[self.data.course == course].sort_values("priority", ascending=False)
        d = d.assign(level=[level_of(p) for p in d.p])
        cols = ["pseudo", "p", "level", "priority", "cp_max_fail_streak", "weeks_since_login", "prior_gpa"]
        return d[cols].head(top) if top else d[cols]

    def card(self, course: int, pseudo: str) -> dict:
        sid = self._by_pseudo.get(pseudo)
        if sid is None:
            raise KeyError("студент не найден")
        d = self.data
        row = d[(d.student_id == sid) & (d.course == course)]
        if row.empty:
            raise KeyError("студент не учится в этой дисциплине")
        contrib = group_shapley(self.model.predict_proba, row, self.bg)
        w = self.world
        sched = np.flatnonzero(w.cp_scheduled[course, :self.week])
        return {
            "pseudo": pseudo, "course": course, "course_name": self.course_name(course), "p": float(row.p.iloc[0]),
            "level": level_of(float(row.p.iloc[0])), "contrib": {g: contrib[g] for g in GROUPS},
            "text": explain_text(row.iloc[0], contrib),
            "what_if": {g: what_if(self.model.predict_proba, row, self.bg, g) for g in ("activity", "assessments")},
            "logins": [float(x) for x in np.nan_to_num(w.logins[sid, course, :self.week])],
            "cp": [(int(t) + 1, None if np.isnan(w.cp_score[sid, course, t]) else float(w.cp_score[sid, course, t]))
                   for t in sched],
            "pass_score": w.pass_score, "week": self.week}

    # --- журнал и несогласие
    def log_view(self, user: str, course: int, pseudo: str):
        log_action(self.conn, user, "view_card", self._by_pseudo.get(pseudo), f"course={course};week={self.week}")

    def disagree(self, user: str, course: int, pseudo: str, comment: str):
        if pseudo not in self._by_pseudo:
            raise KeyError("студент не найден")
        log_action(self.conn, user, "disagree", self._by_pseudo[pseudo],
                   json.dumps({"course": course, "week": self.week, "comment": comment[:500]}, ensure_ascii=False))

    def audit(self, limit: int = 200) -> list:
        return self.conn.execute(
            "SELECT id, ts, actor, action, student_id, details FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def build_demo_service(seed: int = 1, n_train: int = 1500, n_serve: int = 400, week: int = 8) -> RiskService:
    """Обучает модель на одной когорте и обслуживает другую (студенты не пересекаются)."""
    from .store import connect, init_db, save_world
    wt = simulate_world(n_train, seed=seed)
    d = build_dataset(wt, weeks=(week,))
    parts = split_students(d.student_id.unique(), seed=seed)
    tr, va = take(d, parts["train"]), take(d, parts["val"])
    model = Calibrated(BoostModel(seed).fit(tr, tr.y.to_numpy())).fit_calibration(va, va.y.to_numpy())
    ws = simulate_world(n_serve, seed=seed + 1000)
    conn = connect()
    init_db(conn)
    save_world(conn, ws)
    teachers = {"teacher1": [0, 1, 2], "teacher2": [3, 4]}
    return RiskService(ws, conn, model, tr.sample(100, random_state=0), week, teachers, secret=b"demo-secret-change-me")
