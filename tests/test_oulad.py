import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from edurisk.features import build_features
from edurisk.oulad import load_oulad, summary
from edurisk.store import connect, init_db, load_world, save_world


def make_fixture(d: pathlib.Path):
    pd.DataFrame({"code_module": ["AAA", "AAA"], "code_presentation": ["2013J", "2014J"],
                  "module_presentation_length": [268, 269]}).to_csv(d / "courses.csv", index=False)
    pd.DataFrame({
        "code_module": ["AAA"] * 4, "code_presentation": ["2013J", "2013J", "2013J", "2014J"],
        "id_student": [11, 12, 13, 11], "gender": ["M", "F", "M", "M"], "region": ["Wales", "Scotland", "Wales", "Wales"],
        "studied_credits": [60, 120, 60, 60], "final_result": ["Pass", "Withdrawn", "Fail", "Distinction"]
    }).to_csv(d / "studentInfo.csv", index=False)
    # студент 11: клики в 2013J: день 0 (3), день 3 (2), день 3 повторно (1) -> 2 дня, 6 кликов в нед.1;
    # день 7 (4) -> нед.2; день -5 отбрасывается; день 112 = неделя 17 при n_weeks=16 отбрасывается
    rows = [("AAA", "2013J", 11, 0, 3), ("AAA", "2013J", 11, 3, 2), ("AAA", "2013J", 11, 3, 1),
            ("AAA", "2013J", 11, 7, 4), ("AAA", "2013J", 11, -5, 9), ("AAA", "2013J", 11, 112, 7),
            ("AAA", "2013J", 12, 6, 5), ("AAA", "2014J", 11, 20, 8)]
    pd.DataFrame(rows, columns=["code_module", "code_presentation", "id_student", "date", "sum_click"]
                 ).to_csv(d / "studentVle.csv", index=False)
    pd.DataFrame({
        "code_module": ["AAA"] * 4, "code_presentation": ["2013J"] * 4, "id_assessment": [1, 2, 3, 4],
        "assessment_type": ["TMA", "CMA", "Exam", "TMA"], "date": [20, 20, 30, 200], "weight": [10, 10, 70, 10]
    }).to_csv(d / "assessments.csv", index=False)
    # TMA(1) и CMA(2) сроки в день 20 -> неделя 3 (индекс 2); Exam и поздний TMA(4) не точки
    pd.DataFrame({
        "id_assessment": [1, 2, 1, 2, 1, 3], "id_student": [11, 11, 12, 12, 13, 11],
        "date_submitted": [19, 20, 25, 20, 20, 30], "is_banked": [0, 0, 0, 1, 0, 0],
        "score": [80.0, 60.0, 90.0, 50.0, np.nan, 70.0]}).to_csv(d / "studentAssessment.csv", index=False)


class OuladTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)
        make_fixture(cls.d)
        cls.w = load_oulad(str(cls.d))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_structure(self):
        w = self.w
        self.assertEqual((w.n_students, w.n_courses, w.n_weeks), (3, 2, 16))
        self.assertEqual(list(w.curriculum.name), ["AAA", "AAA"])
        self.assertEqual(list(w.curriculum.semester), [0, 1])
        # студент 11 учится в обеих презентациях, остальные в одной
        np.testing.assert_array_equal(w.enrolled, [[True, True], [True, False], [True, False]])
        self.assertEqual(w.pass_score, 40.0)

    def test_outcomes(self):
        w = self.w
        np.testing.assert_array_equal(w.passed[:, 0], [True, False, False])
        np.testing.assert_array_equal(w.dropped[:, 0], [False, True, False])
        np.testing.assert_array_equal(w.admitted[:, 0], [True, False, True])
        self.assertTrue(w.passed[0, 1])                                    # Distinction = успех
        self.assertEqual(list(w.students.workload_ze), [60, 120, 60])
        self.assertEqual(list(w.students.gender), ["m", "f", "m"])

    def test_week_binning_distinct_days_and_clicks(self):
        w = self.w
        self.assertEqual(w.logins[0, 0, 0], 2)         # дни 0 и 3 (день 3 дважды -> один день)
        self.assertEqual(w.minutes[0, 0, 0], 6)        # 3 + 2 + 1
        self.assertEqual(w.logins[0, 0, 1], 1)         # день 7 -> неделя 2
        self.assertEqual(w.minutes[0, 0, 1], 4)
        self.assertEqual(w.logins[1, 0, 0], 1)         # день 6 -> всё ещё неделя 1
        self.assertEqual(w.minutes[1, 0, 0], 5)
        self.assertEqual(np.nansum(w.logins[0, 0]), 3)  # -5 и 112 не попали
        self.assertEqual(w.logins[0, 1, 2], 1)         # день 20 второй презентации -> неделя 3
        self.assertTrue(np.isnan(w.logins[1, 1]).all())  # не зачислен -> NaN
        self.assertEqual(w.logins[2, 0].sum(), 0)      # зачислен, но кликов нет -> нули

    def test_checkpoints(self):
        w = self.w
        self.assertEqual(list(np.flatnonzero(w.cp_scheduled[0])), [2])      # только неделя 3; экзамен и день 200 нет
        self.assertEqual(w.cp_scheduled[1].sum(), 0)
        # студент 11: TMA 80 (в срок) и CMA 60 (в срок) -> среднее 70
        self.assertEqual(w.cp_score[0, 0, 2], 70)
        # студент 12: TMA сдан с опозданием (25 > 20) не считается, CMA перенесён (banked) не считается -> NaN
        self.assertTrue(np.isnan(w.cp_score[1, 0, 2]))
        self.assertTrue(np.isnan(w.cp_score[2, 0, 2]))                      # балл не указан
        self.assertEqual(np.isnan(w.cp_score[:, 0, [0, 1, 3]]).all(), True)

    def test_works_with_stage1_features_store_and_summary(self):
        coh = self.w.course_view(0)
        X = build_features(coh, 8)
        self.assertEqual(len(X), 3)
        np.testing.assert_array_equal(coh.outcome, [False, True, True])
        conn = connect()
        init_db(conn)
        save_world(conn, self.w)
        back = load_world(conn)
        np.testing.assert_array_equal(back.logins, self.w.logins)
        self.assertEqual(back.scenario, "OULAD")
        self.assertIn("студентов 3", summary(self.w))

    def test_errors(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                load_oulad(t)
            make_fixture(pathlib.Path(t))
            pd.read_csv(pathlib.Path(t) / "studentInfo.csv").drop(columns=["region"]).to_csv(
                pathlib.Path(t) / "studentInfo.csv", index=False)
            with self.assertRaises(ValueError):
                load_oulad(t)


if __name__ == "__main__":
    unittest.main()
