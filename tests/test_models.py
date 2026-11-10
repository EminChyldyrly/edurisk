import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from edurisk.costs import (bayes_threshold, best_threshold, mean_cost, row_cost, weighted_recall_at_n,
                           ze_weights)
from edurisk.dataset import MODEL_FEATURES, build_dataset, split_students, take
from edurisk.experiment import MODELS, fit_all, run_seed
from edurisk.metrics import ece, holm, paired_bootstrap, paired_bootstrap_loss
from edurisk.models import BoostModel, Calibrated, RuleModel, Stack
from edurisk.world import simulate_world


class CostTests(unittest.TestCase):
    def test_ze_weights(self):
        w = ze_weights([2, 4, 6], gamma=1.0)
        np.testing.assert_allclose(w, [0.5, 1.0, 1.5])
        np.testing.assert_allclose(ze_weights([2, 4, 6], gamma=0.0), [1, 1, 1])
        np.testing.assert_allclose(ze_weights([2, 4, 6], gamma=2.0), [0.25, 1.0, 2.25])
        np.testing.assert_allclose(ze_weights([3, 6], gamma=1.0, mean_ze=3.0), [1.0, 2.0])
        with self.assertRaises(ValueError):
            ze_weights([0, 3])

    def test_threshold(self):
        self.assertAlmostEqual(bayes_threshold(3.0), 0.25)
        self.assertAlmostEqual(bayes_threshold(1.0), 0.5)
        self.assertAlmostEqual(bayes_threshold(4.0, c_fa=2.0), 1 / 3)
        with self.assertRaises(ValueError):
            bayes_threshold(0)

    def test_mean_cost_by_hand(self):
        y = np.array([1, 1, 0, 0, 1])
        flag = np.array([1, 0, 1, 0, 0])
        w = np.array([1.0, 2.0, 0.5, 1.0, 1.0])
        # пропуски: объекты 1 (w=2) и 4 (w=1) -> k*(2+1); ложная тревога: объект 2 (w=0.5)
        self.assertAlmostEqual(mean_cost(y, flag, w, k=3.0), (3 * 3 + 0.5) / 5)
        np.testing.assert_allclose(row_cost(y, flag, w, 3.0), [0, 6, 0.5, 0, 3])

    def test_decision_threshold_does_not_depend_on_ze_weight(self):
        rng = np.random.default_rng(0)
        p = rng.random(4000)
        y = rng.random(4000) < p
        w1, w2 = np.ones(4000), rng.uniform(0.2, 3, 4000)
        t_bayes = bayes_threshold(3.0)
        grid = np.linspace(0.05, 0.95, 91)
        for w in (w1, w2):
            costs = [mean_cost(y, p >= g, w, 3.0) for g in grid]
            self.assertLess(abs(grid[int(np.argmin(costs))] - t_bayes), 0.08)
        self.assertLess(abs(best_threshold(y, p, w2, 3.0) - t_bayes), 0.08)

    def test_weighted_recall_at_n_and_priority_order(self):
        y = np.array([1, 1, 0, 1])
        p = np.array([0.9, 0.5, 0.8, 0.3])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(weighted_recall_at_n(y, p, w, 2), 1 / 3)        # верх: объекты 0 и 2
        # тяжёлый курс (w=6) у объекта 3 поднимает его приоритет выше всех
        w2 = np.array([1.0, 1.0, 1.0, 6.0])
        self.assertAlmostEqual(weighted_recall_at_n(y, p, w2, 1), 6 / 8)
        self.assertAlmostEqual(weighted_recall_at_n(y, p, w, 4), 1.0)
        with self.assertRaises(ValueError):
            weighted_recall_at_n(y, p, w, 0)


class MetricTests(unittest.TestCase):
    def test_ece_by_hand(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(ece(y, p), 0.1)
        self.assertAlmostEqual(ece(y, np.array([0.0, 0.0, 1.0, 1.0])), 0.0)
        self.assertAlmostEqual(ece(np.array([1, 1]), np.array([0.2, 0.2])), 0.8)
        self.assertAlmostEqual(ece(np.array([0, 1]), np.array([1.0, 0.0])), 1.0)   # p=1.0 в последнем бине

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(holm([0.5]), [0.5])
        self.assertTrue((holm([0.4, 0.9]) <= 1).all())

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(1)
        n = 600
        groups = np.repeat(np.arange(n // 3), 3)
        y = (rng.random(n) < 0.3).astype(int)
        good = y * 0.5 + rng.random(n) * 0.6
        bad = rng.random(n)
        r = paired_bootstrap(roc_auc_score, y, good, bad, groups, n_boot=200, seed=2)
        self.assertGreater(r["diff"], 0.2)
        self.assertGreater(r["lo"], 0)
        self.assertLess(r["p"], 0.05)
        same = paired_bootstrap(roc_auc_score, y, good, good, groups, n_boot=100, seed=2)
        self.assertEqual((same["diff"], same["lo"], same["hi"]), (0.0, 0.0, 0.0))

    def test_paired_bootstrap_loss(self):
        groups = np.repeat(np.arange(50), 4)
        r = paired_bootstrap_loss(np.full(200, 0.5), np.full(200, 0.2), groups, n_boot=200)
        self.assertAlmostEqual(r["diff"], 0.3)
        self.assertAlmostEqual(r["lo"], 0.3)
        self.assertAlmostEqual(r["hi"], 0.3)
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=200), rng.normal(size=200)
        r = paired_bootstrap_loss(a, b, groups, n_boot=500)
        self.assertLess(r["lo"], r["diff"] < 1 and r["hi"])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = simulate_world(400, seed=3)
        cls.d = build_dataset(cls.w, weeks=(4, 8))

    def test_rows_labels_and_columns(self):
        w, d = self.w, self.d
        self.assertEqual(len(d), 2 * w.enrolled.sum())
        for col in MODEL_FEATURES + ["y", "student_id", "course", "week", "semester"]:
            self.assertIn(col, d.columns)
        one = d[(d.course == 5) & (d.week == 8)].set_index("student_id")
        coh = w.course_view(5)
        np.testing.assert_array_equal(one.loc[coh.students.index, "y"].to_numpy(), coh.outcome.astype(int))
        self.assertTrue((d.groupby("form_exam").ze.min() >= 2).all())
        self.assertEqual(set(d[MODEL_FEATURES].columns) & {"y", "gender", "origin"}, set())

    def test_history_is_missing_in_first_semester_and_present_later(self):
        d = self.d
        self.assertTrue(d[d.semester == 0].prior_gpa.isna().all())
        self.assertTrue(d[d.semester == 2].prior_gpa.notna().mean() > 0.8)

    def test_features_do_not_use_the_future(self):
        w2 = simulate_world(400, seed=3)
        rng = np.random.default_rng(0)
        w2.logins[:, :, 8:] = rng.integers(0, 50, w2.logins[:, :, 8:].shape)
        w2.cp_score[:, :, 8:] = np.where(w2.cp_scheduled[:, 8:], rng.uniform(0, 100, w2.cp_score[:, :, 8:].shape), np.nan)
        # итоги семестров 1+ тоже подменяем: история семестра 0 и признаки не должны измениться
        sem = w2.curriculum.semester.to_numpy()
        w2.passed[:, sem >= 1] = ~w2.passed[:, sem >= 1]
        d2 = build_dataset(w2, weeks=(8,), courses=[c for c in range(w2.n_courses) if sem[c] <= 1])
        base = self.d[(self.d.week == 8) & (self.d.semester <= 1)].drop(columns="y").reset_index(drop=True)
        got = d2.drop(columns="y").reset_index(drop=True)
        pd.testing.assert_frame_equal(base[got.columns], got, check_dtype=False)

    def test_split_by_student(self):
        parts = split_students(self.d.student_id, seed=1)
        a, b, c = parts["train"], parts["val"], parts["test"]
        self.assertEqual(a & b, set())
        self.assertEqual(a & c, set())
        self.assertEqual(b & c, set())
        self.assertEqual(a | b | c, set(self.d.student_id.unique()))
        self.assertAlmostEqual(len(a) / 400, 0.6, delta=0.01)
        tr, te = take(self.d, a), take(self.d, c)
        self.assertEqual(set(tr.student_id) & set(te.student_id), set())
        self.assertEqual(parts["train"], split_students(self.d.student_id, seed=1)["train"])
        with self.assertRaises(ValueError):
            split_students([1, 2, 3], fractions=(0.5, 0.5, 0.5))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w = simulate_world(1200, seed=11)
        d = build_dataset(w, weeks=(8,))
        parts = split_students(d.student_id.unique(), seed=1)
        cls.tr, cls.va, cls.te = (take(d, parts[s]) for s in ("train", "val", "test"))

    def test_rule_by_hand(self):
        X = pd.DataFrame({"cp_max_fail_streak": [0, 2, 1, 0], "prior_gpa": [4.5, 4.5, 2.5, np.nan]})
        r = RuleModel().fit(X, np.array([0, 1, 1, 0]))
        np.testing.assert_array_equal(r._rule(X), [False, True, True, False])
        np.testing.assert_allclose(r.predict_proba(X), [0.0, 1.0, 1.0, 0.0])

    def test_boost_learns_and_shuffled_labels_do_not(self):
        m = BoostModel().fit(self.tr, self.tr.y.to_numpy())
        self.assertGreater(roc_auc_score(self.te.y, m.predict_proba(self.te)), 0.85)
        # перемешанные метки: качество около случайного (метки сильно скоррелированы внутри студента,
        # поэтому один прогон шумит на ±0.05, и берём среднее по нескольким перестановкам)
        aucs = [roc_auc_score(self.te.y, BoostModel().fit(
            self.tr, np.random.default_rng(i).permutation(self.tr.y.to_numpy())).predict_proba(self.te))
            for i in range(5)]
        self.assertLess(abs(np.mean(aucs) - 0.5), 0.07)
        self.assertGreater(roc_auc_score(self.te.y, m.predict_proba(self.te)) - np.mean(aucs), 0.25)

    def test_calibration_fixes_weighted_training(self):
        w = np.where(self.tr.y == 1, 5.0, 1.0)
        raw = BoostModel().fit(self.tr, self.tr.y.to_numpy(), sample_weight=w)
        cal = Calibrated(raw).fit_calibration(self.va, self.va.y.to_numpy())
        e_raw = ece(self.te.y, raw.predict_proba(self.te))
        e_cal = ece(self.te.y, cal.predict_proba(self.te))
        self.assertGreater(e_raw, 0.08)
        self.assertLess(e_cal, 0.04)
        grid = np.linspace(0, 1, 50)
        self.assertTrue((np.diff(cal.iso_.predict(grid)) >= -1e-12).all())    # монотонна

    def test_fit_all_uses_class_weights_and_calibrates_on_validation(self):
        ms = fit_all(self.tr, self.va, mean_ze=4.0, k=3.0, gamma=2.0, seed=0)
        self.assertEqual(set(ms), set(MODELS) - set())
        plain = ms["boost"].predict_proba(self.te).mean()
        weighted = ms["boost_cs"].predict_proba(self.te).mean()
        self.assertGreater(weighted, plain + 0.05)           # вес класса k=3 завышает вероятности
        yv = self.va.y.to_numpy()
        for name in ("boost+cal", "boost_cs+cal"):
            pv = ms[name].predict_proba(self.va)
            self.assertLess(ece(yv, pv), 0.005, name)        # изотоника подогнана именно на val
        self.assertLess(ece(self.te.y, ms["boost_cs+cal"].predict_proba(self.te)), 0.04)

    def test_stack_weights_prefer_informative_model(self):
        good = BoostModel().fit(self.tr, self.tr.y.to_numpy())

        class Noise:
            def predict_proba(self, X):
                return np.random.default_rng(len(X)).random(len(X))
        s = Stack([good, Noise()]).fit_stack(self.va, self.va.y.to_numpy())
        self.assertGreater(s.weights_[0], 5 * abs(s.weights_[1]))
        p = s.predict_proba(self.te)
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class ExperimentTests(unittest.TestCase):
    def test_run_seed_smoke_and_separation(self):
        res, mze = run_seed(3, n_students=500, weeks=(8,))
        r = res[8]
        for m in MODELS:
            self.assertTrue(all(np.isfinite(v) for v in r["metrics"][m].values()), m)
        self.assertLess(r["metrics"]["stack"]["cost"], r["metrics"]["_baseline_cost_none"]["cost"])
        self.assertLess(r["metrics"]["rule"]["roc_auc"], r["metrics"]["logit"]["roc_auc"])
        self.assertGreater(mze, 2)


if __name__ == "__main__":
    unittest.main()
