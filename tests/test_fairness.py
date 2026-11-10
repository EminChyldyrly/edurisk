import unittest

import numpy as np
import pandas as pd

from edurisk.fairness import (apply_thresholds, disparity_ci, equalize_fpr_thresholds, group_report, rates)
from edurisk.privacy import ROLES, can, k_anonymity, pseudonym, pseudonymize_frame


class RatesTests(unittest.TestCase):
    def test_rates_by_hand(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0])
        f = np.array([1, 1, 0, 1, 0, 0, 0])
        self.assertEqual(rates(y, f), (2 / 3, 1 / 4))
        tpr, fpr = rates(np.zeros(3), np.ones(3))
        self.assertTrue(np.isnan(tpr))
        self.assertEqual(fpr, 1.0)

    def test_group_report_by_hand(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
        p = np.array([0.9, 0.1, 0.2, 0.6, 0.8, 0.3, 0.9, 0.4, 0.5, 0.1])
        g = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"])
        r = group_report(y, p, g, thr=0.5)
        self.assertEqual(r.loc["a", "n"], 3)
        self.assertAlmostEqual(r.loc["a", "base_rate"], 2 / 3)
        self.assertAlmostEqual(r.loc["a", "tpr"], 0.5)
        self.assertAlmostEqual(r.loc["a", "fpr"], 0.0)
        self.assertAlmostEqual(r.loc["b", "flag_rate"], 2 / 3)
        self.assertAlmostEqual(r.loc["b", "fpr"], 0.5)
        self.assertAlmostEqual(r.loc["b", "mean_pred"], (0.6 + 0.8 + 0.3) / 3)
        self.assertAlmostEqual(r.loc["a", "auc"], 1.0)
        self.assertAlmostEqual(r.loc["c", "auc"], 0.75)      # 3 из 4 пар упорядочены верно


class DisparityTests(unittest.TestCase):
    def test_point_estimate_and_detection(self):
        rng = np.random.default_rng(0)
        n = 4000
        sid = np.arange(n)
        g = np.where(rng.random(n) < 0.5, "ref", "oth")
        y = (rng.random(n) < 0.2).astype(int)
        p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, n) + 0.15 * (g == "oth"), 0, 1)   # другая группа получает завышенный прогноз
        r = disparity_ci(y, p, g, sid, thr=0.5, ref="ref", other="oth", n_boot=100, seed=1)
        flag = p >= 0.5
        m = g == "oth"
        self.assertAlmostEqual(r["fpr"]["diff"], flag[m & (y == 0)].mean() - flag[~m & (y == 0)].mean())
        self.assertGreater(r["fpr"]["lo"], 0)                     # перекос по ложным тревогам виден
        self.assertGreater(r["flag_rate"]["lo"], 0)
        fair = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, n), 0, 1)
        r2 = disparity_ci(y, fair, g, sid, thr=0.5, ref="ref", other="oth", n_boot=100, seed=1)
        self.assertLess(abs(r2["fpr"]["diff"]), 0.02)            # без смещения разность около нуля
        self.assertLess(abs(r2["flag_rate"]["diff"]), 0.03)


class MitigationTests(unittest.TestCase):
    def test_equalized_fpr_exact_small_case(self):
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.95])
        y = np.array([0] * 10 + [1])
        g = np.array(["a"] * 11)
        thr = equalize_fpr_thresholds(y, p, g, target_fpr=0.25)["a"]
        flag = apply_thresholds(p, g, {"a": thr})
        self.assertEqual(flag[:10].sum(), 2)                  # FPR = 0.2 ≤ 0.25 (консервативно), не 0.1
        self.assertTrue(flag[9] and flag[10] and not flag[7])

    def test_equalized_fpr(self):
        rng = np.random.default_rng(1)
        n = 6000
        g = np.where(rng.random(n) < 0.5, "a", "b")
        y = (rng.random(n) < 0.25).astype(int)
        p = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.15, n) + 0.12 * (g == "b"), 0, 1)
        thr = equalize_fpr_thresholds(y, p, g, target_fpr=0.1)
        self.assertGreater(thr["b"], thr["a"])                    # группе с завышенным прогнозом порог выше
        flag = apply_thresholds(p, g, thr)
        for grp in ("a", "b"):
            m = (g == grp) & (y == 0)
            self.assertLessEqual(flag[m].mean(), 0.1 + 1e-9)
            self.assertGreater(flag[m].mean(), 0.08)              # и не сильно ниже цели
        with self.assertRaises(ValueError):
            equalize_fpr_thresholds(y, p, g, 1.5)
        self.assertEqual(equalize_fpr_thresholds(np.ones(3), np.array([.1, .2, .3]), np.array(["a"] * 3), 0.1), {"a": 1.0})


class PrivacyTests(unittest.TestCase):
    def test_pseudonym(self):
        a = pseudonym(42, b"secret")
        self.assertEqual(a, pseudonym(42, b"secret"))
        self.assertNotEqual(a, pseudonym(43, b"secret"))
        self.assertNotEqual(a, pseudonym(42, b"other"))
        self.assertNotIn("42", a)
        self.assertEqual(len(a), 12)
        with self.assertRaises(ValueError):
            pseudonym(1, b"")

    def test_pseudonymize_frame(self):
        df = pd.DataFrame({"student_id": [1, 2], "gender": ["f", "m"], "origin": ["x", "y"], "ege": [70, 80]})
        out = pseudonymize_frame(df, b"s")
        self.assertEqual(list(out.columns), ["student_id", "ege"])
        self.assertEqual(list(out.ege), [70, 80])
        self.assertNotIn(1, list(out.student_id))
        self.assertEqual(list(df.columns), ["student_id", "gender", "origin", "ege"])    # исходник не тронут

    def test_k_anonymity(self):
        df = pd.DataFrame({"g": ["f", "f", "f", "m", "m"], "o": ["a", "a", "a", "b", "b"], "v": range(5)})
        self.assertEqual(k_anonymity(df, ["g", "o"]), 2)
        self.assertEqual(k_anonymity(df, ["g"]), 2)
        self.assertEqual(k_anonymity(df.iloc[:0], ["g"]), 0)
        self.assertEqual(k_anonymity(df.assign(id=range(5)), ["id"]), 1)

    def test_policy(self):
        self.assertTrue(can("teacher", "view_student_card"))
        self.assertTrue(can("teacher", "mark_disagree"))
        self.assertFalse(can("teacher", "view_audit"))
        self.assertFalse(can("methodist", "view_student_card"))
        self.assertTrue(can("admin", "view_audit"))
        self.assertFalse(can("admin", "view_student_card"))
        self.assertFalse(can("guest", "view_aggregates"))
        self.assertFalse(can("teacher", "delete_everything"))
        self.assertEqual(set(ROLES), {"teacher", "methodist", "admin"})


if __name__ == "__main__":
    unittest.main()
