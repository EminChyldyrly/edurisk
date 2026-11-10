"""Хранилище (этап 3): схема БД в 3НФ, загрузка/выгрузка World, запросы со срезом по неделе.

Работает через DB-API (sqlite3 из стандартной библиотеки — на нём идут тесты). Для PostgreSQL
DDL генерируется тем же кодом: `python scripts/make_schema.py pg > db/schema_pg.sql`. Запуск на
PostgreSQL в среде разработки НЕ проверялся (нет сети и сервера).

Скрытая «истина» симулятора (`World.truth`) в БД не хранится: БД моделирует то, что видит вуз.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

import numpy as np
import pandas as pd

from .curriculum import FORMS
from .world import World

_TABLES = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE student (
    student_id  INTEGER PRIMARY KEY,
    ege         DOUBLE PRECISION,
    workload_ze DOUBLE PRECISION,
    has_job     INTEGER CHECK (has_job IN (0, 1)),
    gender      TEXT,
    origin      TEXT
);
CREATE TABLE course (
    course_id        INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    semester         INTEGER NOT NULL,
    form             TEXT NOT NULL CHECK (form IN ({forms})),
    ze               INTEGER NOT NULL CHECK (ze > 0),
    difficulty       DOUBLE PRECISION NOT NULL,
    prereq_course_id INTEGER REFERENCES course (course_id)
);
CREATE TABLE checkpoint_schedule (
    course_id INTEGER NOT NULL REFERENCES course (course_id),
    week      INTEGER NOT NULL CHECK (week >= 1),
    PRIMARY KEY (course_id, week)
);
CREATE TABLE enrollment (
    student_id    INTEGER NOT NULL REFERENCES student (student_id),
    course_id     INTEGER NOT NULL REFERENCES course (course_id),
    dropped       INTEGER NOT NULL CHECK (dropped IN (0, 1)),
    admitted      INTEGER NOT NULL CHECK (admitted IN (0, 1)),
    passed        INTEGER NOT NULL CHECK (passed IN (0, 1)),
    grade         INTEGER CHECK (grade BETWEEN 2 AND 5),
    retake_passed INTEGER CHECK (retake_passed IN (0, 1)),
    PRIMARY KEY (student_id, course_id)
);
CREATE TABLE weekly_activity (
    student_id INTEGER NOT NULL,
    course_id  INTEGER NOT NULL,
    week       INTEGER NOT NULL CHECK (week >= 1),
    logins     DOUBLE PRECISION,
    minutes    DOUBLE PRECISION,
    PRIMARY KEY (student_id, course_id, week),
    FOREIGN KEY (student_id, course_id) REFERENCES enrollment (student_id, course_id)
);
CREATE TABLE checkpoint (
    student_id INTEGER NOT NULL,
    course_id  INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    score      DOUBLE PRECISION CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    PRIMARY KEY (student_id, course_id, week),
    FOREIGN KEY (student_id, course_id) REFERENCES enrollment (student_id, course_id),
    FOREIGN KEY (course_id, week) REFERENCES checkpoint_schedule (course_id, week)
);
CREATE TABLE audit_log (
    id         {autoid},
    ts         TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    student_id INTEGER,
    details    TEXT
);
CREATE INDEX idx_activity_course_week ON weekly_activity (course_id, week);
CREATE INDEX idx_audit_student ON audit_log (student_id);
"""

TABLE_NAMES = ["meta", "student", "course", "checkpoint_schedule", "enrollment", "weekly_activity",
               "checkpoint", "audit_log"]


def schema_sql(dialect: str = "sqlite") -> str:
    """DDL для sqlite или pg (PostgreSQL)."""
    if dialect == "sqlite":
        autoid = "INTEGER PRIMARY KEY AUTOINCREMENT"
    elif dialect == "pg":
        autoid = "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
    else:
        raise ValueError("dialect: 'sqlite' или 'pg'")
    return _TABLES.format(forms=", ".join(f"'{f}'" for f in FORMS), autoid=autoid).strip() + "\n"


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(schema_sql("sqlite"))


def _n(x):
    """numpy-скаляр → python; NaN → None (NULL)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def save_world(conn: sqlite3.Connection, w: World) -> None:
    n, C, W = w.logins.shape
    ids = [int(i) for i in w.students.index]
    cur = w.curriculum
    with conn:
        conn.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("scenario", w.scenario), ("pass_score", repr(float(w.pass_score))), ("n_weeks", str(W))])
        conn.executemany("INSERT INTO student VALUES (?, ?, ?, ?, ?, ?)", [
            (i, _n(r.ege), _n(r.workload_ze), int(r.has_job), r.gender, r.origin)
            for i, r in zip(ids, w.students.itertuples())])
        conn.executemany("INSERT INTO course VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (int(c), r.name, int(r.semester), r.form, int(r.ze), float(r.difficulty),
             None if r.prereq < 0 else int(r.prereq)) for c, r in zip(cur.index, cur.itertuples())])
        conn.executemany("INSERT INTO checkpoint_schedule VALUES (?, ?)", [
            (int(cur.index[c]), int(t) + 1) for c in range(C) for t in np.flatnonzero(w.cp_scheduled[c])])
        si, ci = np.nonzero(w.enrolled)
        conn.executemany("INSERT INTO enrollment VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (ids[s], int(cur.index[c]), int(w.dropped[s, c]), int(w.admitted[s, c]), int(w.passed[s, c]),
             None if np.isnan(w.grade[s, c]) else int(w.grade[s, c]),
             None if np.isnan(w.retake_passed[s, c]) else int(w.retake_passed[s, c])) for s, c in zip(si, ci)])
        conn.executemany("INSERT INTO weekly_activity VALUES (?, ?, ?, ?, ?)", (
            (ids[s], int(cur.index[c]), t + 1, _n(w.logins[s, c, t]), _n(w.minutes[s, c, t]))
            for s, c in zip(si, ci) for t in range(W)))
        conn.executemany("INSERT INTO checkpoint VALUES (?, ?, ?, ?)", (
            (ids[s], int(cur.index[c]), int(t) + 1, _n(w.cp_score[s, c, t]))
            for s, c in zip(si, ci) for t in np.flatnonzero(w.cp_scheduled[c])))


def load_world(conn: sqlite3.Connection) -> World:
    """Обратная операция. `truth` в БД нет, поэтому у загруженного World truth=None."""
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    W = int(meta["n_weeks"])
    st = pd.read_sql_query("SELECT * FROM student ORDER BY student_id", conn).set_index("student_id")
    co = pd.read_sql_query("SELECT * FROM course ORDER BY course_id", conn).set_index("course_id")
    cur = pd.DataFrame({"name": co.name, "semester": co.semester, "form": co.form, "ze": co.ze,
                        "difficulty": co.difficulty,
                        "prereq": co.prereq_course_id.fillna(-1).astype(int)})
    cur.index = pd.Index(range(len(co)), name="course")
    if list(co.index) != list(range(len(co))):
        raise ValueError("course_id должны идти подряд с 0")
    n, C = len(st), len(co)
    row = {int(i): k for k, i in enumerate(st.index)}
    rows = lambda s: s.map(row).to_numpy()

    sched = np.zeros((C, W), bool)
    cs = pd.read_sql_query("SELECT course_id, week FROM checkpoint_schedule", conn)
    sched[cs.course_id.to_numpy(), cs.week.to_numpy() - 1] = True

    enrolled = np.zeros((n, C), bool)
    dropped, admitted, passed = (np.zeros((n, C), bool) for _ in range(3))
    grade, retake = np.full((n, C), np.nan), np.full((n, C), np.nan)
    en = pd.read_sql_query("SELECT * FROM enrollment", conn)
    r, c = rows(en.student_id), en.course_id.to_numpy()
    enrolled[r, c] = True
    dropped[r, c], admitted[r, c], passed[r, c] = en.dropped == 1, en.admitted == 1, en.passed == 1
    grade[r, c] = en.grade.astype(float)
    retake[r, c] = en.retake_passed.astype(float)

    logins = np.full((n, C, W), np.nan, np.float32)
    minutes = np.full((n, C, W), np.nan, np.float32)
    act = pd.read_sql_query("SELECT * FROM weekly_activity", conn)
    r, c, t = rows(act.student_id), act.course_id.to_numpy(), act.week.to_numpy() - 1
    logins[r, c, t] = act.logins.astype(float)
    minutes[r, c, t] = act.minutes.astype(float)
    score = np.full((n, C, W), np.nan, np.float32)
    cp = pd.read_sql_query("SELECT * FROM checkpoint", conn)
    score[rows(cp.student_id), cp.course_id.to_numpy(), cp.week.to_numpy() - 1] = cp.score.astype(float)

    students = st.copy()
    students["has_job"] = students.has_job.astype(int)
    return World(students, cur, logins, minutes, score, sched, enrolled, dropped, admitted, passed, grade,
                 retake, float(meta["pass_score"]), meta["scenario"], None)


def activity_until(conn: sqlite3.Connection, course_id: int, week: int) -> pd.DataFrame:
    """Активность по дисциплине за недели 1..week — срез на уровне SQL (будущего в выдаче нет)."""
    if week < 1:
        raise ValueError("week должна быть >= 1")
    return pd.read_sql_query(
        "SELECT student_id, week, logins, minutes FROM weekly_activity "
        "WHERE course_id = ? AND week <= ? ORDER BY student_id, week", conn, params=(course_id, week))


def log_action(conn: sqlite3.Connection, actor: str, action: str, student_id: int | None = None,
               details: str | None = None) -> None:
    """Запись в журнал действий (кто, что и о ком смотрел). Журнал только дополняется."""
    with conn:
        conn.execute("INSERT INTO audit_log (ts, actor, action, student_id, details) VALUES (?, ?, ?, ?, ?)",
                     (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), actor, action,
                      student_id, details))
