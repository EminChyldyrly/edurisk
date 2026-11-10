import pathlib
import sqlite3
import unittest

import numpy as np

from edurisk.features import build_features
from edurisk.store import (TABLE_NAMES, activity_until, connect, init_db, load_world, log_action, schema_sql,
                           save_world)
from edurisk.world import simulate_world

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _eq(a, b):
    np.testing.assert_array_equal(a, b)


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = simulate_world(120, seed=4)
        cls.conn = connect()
        init_db(cls.conn)
        save_world(cls.conn, cls.w)
        cls.back = load_world(cls.conn)

    def test_round_trip_is_exact(self):
        a, b = self.w, self.back
        for name in ("logins", "minutes", "cp_score", "grade", "retake_passed"):
            _eq(getattr(a, name), getattr(b, name))
        for name in ("cp_scheduled", "enrolled", "dropped", "admitted", "passed"):
            _eq(getattr(a, name), getattr(b, name))
        self.assertEqual((a.scenario, a.pass_score), (b.scenario, b.pass_score))
        self.assertTrue(a.curriculum.reset_index(drop=True).equals(b.curriculum.reset_index(drop=True)))
        import pandas.testing as pdt
        pdt.assert_frame_equal(a.students, b.students, check_dtype=False, check_index_type=False)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_round_trip_keeps_lost_records_s4(self):
        w = simulate_world(80, seed=4, scenario="S4")
        conn = connect()
        init_db(conn)
        save_world(conn, w)
        _eq(load_world(conn).logins, w.logins)
        _eq(load_world(conn).cp_score, w.cp_score)

    def test_loaded_world_gives_same_features(self):
        for c in (0, 7):
            X1 = build_features(self.w.course_view(c), 8)
            X2 = build_features(self.back.course_view(c), 8)
            np.testing.assert_allclose(X1.to_numpy(float), X2.to_numpy(float), equal_nan=True)

    def test_row_counts_match_world(self):
        q = lambda s: self.conn.execute(s).fetchone()[0]
        W = self.w.n_weeks
        self.assertEqual(q("SELECT COUNT(*) FROM enrollment"), self.w.enrolled.sum())
        self.assertEqual(q("SELECT COUNT(*) FROM weekly_activity"), self.w.enrolled.sum() * W)
        sched = self.w.cp_scheduled.sum(1)
        self.assertEqual(q("SELECT COUNT(*) FROM checkpoint"), int((self.w.enrolled * sched[None, :]).sum()))
        self.assertEqual(q("SELECT COUNT(*) FROM course"), self.w.n_courses)

    def test_sql_cutoff_matches_observed(self):
        c, t = 3, 6
        df = activity_until(self.conn, c, t)
        self.assertEqual(df.week.max(), t)
        coh = self.w.course_view(c).observed(t)
        piv = df.pivot(index="student_id", columns="week", values="logins")
        _eq(piv.to_numpy(), self.w.logins[self.w.enrolled[:, c], c, :t])
        self.assertEqual(piv.shape, coh.logins.shape)
        with self.assertRaises(ValueError):
            activity_until(self.conn, c, 0)

    def test_constraints_reject_bad_rows(self):
        conn = connect()
        init_db(conn)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO course VALUES (0, 'x', 0, 'lab', 3, 0.0, NULL)")      # неизвестная форма
        conn.execute("INSERT INTO course VALUES (0, 'x', 0, 'exam', 3, 0.0, NULL)")
        conn.execute("INSERT INTO student VALUES (1, 70, 30, 0, 'f', 'local')")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO enrollment VALUES (1, 0, 0, 1, 1, 7, NULL)")            # оценка 7
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO enrollment VALUES (2, 0, 0, 1, 1, 4, NULL)")            # нет студента 2
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO weekly_activity VALUES (1, 0, 1, 3, 10)")               # нет зачисления

    def test_audit_log_appends(self):
        conn = connect()
        init_db(conn)
        log_action(conn, "teacher:1", "view_card", 42, "week=6")
        log_action(conn, "teacher:1", "disagree", 42)
        rows = conn.execute("SELECT actor, action, student_id, details FROM audit_log ORDER BY id").fetchall()
        self.assertEqual(rows, [("teacher:1", "view_card", 42, "week=6"), ("teacher:1", "disagree", 42, None)])

    def test_schema_files_and_dialects(self):
        self.assertEqual((ROOT / "db" / "schema_pg.sql").read_text(encoding="utf-8"), schema_sql("pg"))
        self.assertIn("GENERATED ALWAYS AS IDENTITY", schema_sql("pg"))
        self.assertNotIn("AUTOINCREMENT", schema_sql("pg"))
        for t in TABLE_NAMES:
            self.assertIn(f"CREATE TABLE {t} ", schema_sql("sqlite"))
        with self.assertRaises(ValueError):
            schema_sql("mysql")


if __name__ == "__main__":
    unittest.main()
