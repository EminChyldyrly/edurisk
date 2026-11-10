import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edurisk import conformal as cf
from edurisk.dataset import add_targets, build_dataset
from edurisk.experiment5 import conformal_eval, fit_t3, make_t2, run_seed5
from edurisk.ordinal import (GRADES, FrankHall, Multinomial, PriorModel, class_metrics, cost_matrix,
                             decide_argmax, decide_min_cost, expected_cost, probs_from_cumulative)
from edurisk.world import simulate_world


class OrdinalTests(unittest.TestCase):
    def test_cost_matrix(self):
        M = cost_matrix(3.0)
        self.assertEqual(M.shape, (4, 4))
        np.testing.assert_array_equal(np.diag(M), 0)
        self.assertEqual(M[0, 3], 9.0)     # истина 2, предсказали 5: завышение на 3 ступени ×3
        self.assertEqual(M[3, 0], 3.0)     # истина 5, предсказали 2: занижение на 3 ступени
        self.assertEqual(M[1, 2], 3.0)
        self.assertEqual(M[2, 1], 1.0)
        np.testing.assert_array_equal(cost_matrix(1.0), cost_matrix(1.0).T)
        with self.assertRaises(ValueError):
            cost_matrix(0)

    def test_decisions_by_hand(self):
        P = np.array([[0.3, 0.0, 0.35, 0.35]])
        self.assertEqual(decide_argmax(P)[0], 4)                       # первый максимум
        self.assertEqual(decide_min_cost(P, cost_matrix(3.0))[0], 2)   # завышение дорого → осторожная оценка
        self.assertEqual(decide_min_cost(P, cost_matrix(1.0))[0], 4)   # симметрично → медиана
        P2 = np.array([[0.5, 0.0, 0.0, 0.5]])
        self.assertEqual(decide_min_cost(P2, cost_matrix(3.0))[0], 2)
        # масштаб положительного веса дисциплины на решение не влияет
        M = cost_matrix(3.0)
        np.testing.assert_array_equal(decide_min_cost(P, M), decide_min_cost(P, 5.0 * M))

    def test_expected_cost_by_hand(self):
        M = cost_matrix(3.0)
        self.assertAlmostEqual(expected_cost(np.array([2, 5]), np.array([5, 2]), np.array([1.0, 2.0]), M),
                               (9 * 1 + 3 * 2) / 2)
        self.assertEqual(expected_cost(np.array([3, 4]), np.array([3, 4]), np.ones(2), M), 0.0)

    def test_class_metrics(self):
        y = np.array([2, 3, 4, 5, 2, 4])
        m = class_metrics(y, y)
        self.assertEqual((m["qwk"], m["mae"], m["recall_2"], m["accuracy"], m["macro_f1"]), (1.0, 0.0, 1.0, 1.0, 1.0))
        c = class_metrics(y, np.full(6, 3))
        self.assertEqual(c["recall_2"], 0.0)
        self.assertAlmostEqual(c["qwk"], 0.0)
        self.assertAlmostEqual(c["mae"], np.mean(np.abs(y - 3)))
        self.assertAlmostEqual(class_metrics(y, np.array([3, 3, 4, 5, 2, 4]))["recall_2"], 0.5)

    def test_probs_from_cumulative_is_valid(self):
        P = probs_from_cumulative(np.array([0.5]), np.array([0.7]), np.array([0.9]))
        np.testing.assert_allclose(P, [[0.5, 0.0, 0.0, 0.5]])           # монотонность принудительно
        P = probs_from_cumulative(np.array([0.8]), np.array([0.5]), np.array([0.1]))
        np.testing.assert_allclose(P, [[0.2, 0.3, 0.4, 0.1]])
        rng = np.random.default_rng(0)
        P = probs_from_cumulative(*rng.random((3, 500)))
        self.assertTrue((P >= -1e-12).all())
        np.testing.assert_allclose(P.sum(1), 1.0)
        P = probs_from_cumulative(np.array([1.4]), np.array([-0.2]), np.array([0.3]))   # вне [0,1] обрезается
        self.assertTrue((P >= 0).all())


class _OneCol:
    def fit(self, X, y, sample_weight=None):
        self.m = LogisticRegression().fit(X[["x"]], y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[["x"]])[:, 1]


class FrankHallTests(unittest.TestCase):
    def test_cascade_on_toy_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=6000)
        g = 2 + (x + rng.normal(0, 0.5, 6000) > -1) + (x + rng.normal(0, 0.5, 6000) > 0.2) + (x + rng.normal(0, 0.5, 6000) > 1.3)
        X = pd.DataFrame({"x": x})
        fh = FrankHall(lambda: _OneCol()).fit(X.iloc[:3000], g[:3000]).fit_calibration(X.iloc[3000:4500], g[3000:4500])
        P = fh.predict_proba(X.iloc[4500:])
        np.testing.assert_allclose(P.sum(1), 1.0)
        self.assertTrue((P >= 0).all())
        exp = P @ GRADES
        order = np.argsort(x[4500:])
        self.assertGreater(np.corrcoef(exp, x[4500:])[0, 1], 0.9)
        self.assertLess(exp[order[:50]].mean(), exp[order[-50:]].mean() - 1.5)
        # средние вероятности классов близки к реальным долям
        for c, share in zip(GRADES, [(g[4500:] == c).mean() for c in GRADES]):
            self.assertAlmostEqual(P[:, c - 2].mean(), share, delta=0.03)


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = simulate_world(500, seed=21)
        cls.d = add_targets(build_dataset(cls.w, weeks=(8,)), cls.w)

    def test_targets_are_consistent_with_world(self):
        d, w = self.d, self.w
        self.assertEqual(len(d), w.enrolled.sum())
        forms = np.where(d.form_zachet == 1, "zachet", "other")
        self.assertTrue(d[forms == "zachet"].grade.isna().all())
        self.assertTrue(d[forms == "other"].grade.between(2, 5).all())
        g = make_t2(d)
        self.assertTrue(set(g.grade.unique()) <= {2, 3, 4, 5})
        self.assertEqual(len(g), (d.form_zachet == 0).sum())
        for _, r in d.sample(50, random_state=0).iterrows():
            self.assertEqual(int(r.admitted), int(w.admitted[int(r.student_id), int(r.course)]))
        # недопущенные имеют двойку, y согласован: двойка ⇒ y = 1
        self.assertTrue((g[g.admitted == 0].grade == 2).all())
        self.assertTrue((g[g.grade == 2].y == 1).all())
        self.assertFalse({"grade", "admitted"} & set(__import__("edurisk.dataset", fromlist=["x"]).MODEL_FEATURES))

    def test_multinomial_aligns_missing_class(self):
        g = make_t2(self.d)
        tr = g[g.grade != 4]
        for kind in ("logit", "boost"):
            m = Multinomial(kind).fit(tr, tr.grade.to_numpy())
            P = m.predict_proba(g)
            self.assertEqual(P.shape, (len(g), 4))
            np.testing.assert_allclose(P[:, 2], 0.0)
            np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
        with self.assertRaises(ValueError):
            Multinomial("svm")

    def test_prior_model(self):
        p = PriorModel().fit(None, np.array([2, 2, 3, 5]))
        np.testing.assert_allclose(p.predict_proba(np.zeros(3))[0], [0.5, 0.25, 0.0, 0.25])


class ConformalTests(unittest.TestCase):
    def test_quantile_by_hand(self):
        P = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
        y = np.zeros(4, int)                       # s = 0.1, 0.2, 0.3, 0.4
        self.assertAlmostEqual(cf.calibrate(P, y, alpha=0.5), 0.3)    # k = ceil(5*0.5) = 3 → третья
        self.assertAlmostEqual(cf.calibrate(P, y, alpha=0.25), 0.4)   # k = ceil(5*0.75) = 4
        self.assertEqual(cf.calibrate(P, y, alpha=0.1), np.inf)       # k = 5 > n → набор из всех классов
        with self.assertRaises(ValueError):
            cf.calibrate(P, y, alpha=1.0)

    def test_sets_and_helpers(self):
        P = np.array([[0.7, 0.2, 0.1, 0.0], [0.4, 0.4, 0.1, 0.1]])
        S = cf.prediction_sets(P, 0.95)
        np.testing.assert_array_equal(S, [[1, 1, 1, 0], [1, 1, 1, 1]])
        np.testing.assert_array_equal(S.sum(1), [3, 4])
        self.assertEqual(cf.mean_size(S), 3.5)
        self.assertEqual(cf.coverage(S, np.array([3, 2])), 0.5)
        np.testing.assert_array_equal(cf.prediction_sets(P, 0.35), [[1, 0, 0, 0], [0, 0, 0, 0]])
        S = cf.prediction_sets(P, np.array([0.0, 1.0, 0.0, 0.0]))          # пороги по классам
        np.testing.assert_array_equal(S, [[0, 1, 0, 0], [0, 1, 0, 0]])

    def test_set_boundary_is_inclusive(self):
        P = np.array([[0.75, 0.25]])                   # 1 − P = [0.25, 0.75], точно представимо
        np.testing.assert_array_equal(cf.prediction_sets(P, 0.25), [[True, False]])
        np.testing.assert_array_equal(cf.prediction_sets(P, 0.75), [[True, True]])
        # точка, давшая порог, сама остаётся в своём наборе
        Pc = np.array([[0.75, 0.25], [0.5, 0.5], [0.875, 0.125]])
        y = np.zeros(3, int)
        q = cf.calibrate(Pc, y, alpha=0.4)             # k = ceil(4·0.6) = 3 → наибольшая нонконформность 0.5
        self.assertEqual(q, 0.5)
        self.assertTrue(cf.prediction_sets(Pc, q)[1, 0])

    def test_coverage_guarantee_on_exchangeable_data(self):
        # гарантия верна «в среднем» по калибровочным выборкам: усредняем по 30 повторам
        K, n = 4, 4000
        cov = {0.1: [], 0.2: []}
        cw = {0.1: [], 0.2: []}
        for seed in range(30):
            rng = np.random.default_rng(seed)
            P = np.exp(rng.normal(size=(n, K)) * 1.5)
            P /= P.sum(1, keepdims=True)
            y = (P.cumsum(1) < rng.random(n)[:, None]).sum(1)
            Pc, yc, Pt, yt = P[:n // 2], y[:n // 2], P[n // 2:], y[n // 2:]
            for alpha in cov:
                cov[alpha].append(cf.coverage(cf.prediction_sets(Pt, cf.calibrate(Pc, yc, alpha)), yt))
                Sc = cf.prediction_sets(Pt, cf.calibrate(Pc, yc, alpha, classwise=True))
                cw[alpha].append([Sc[yt == c, c].mean() for c in range(K)])
        for alpha in cov:
            self.assertGreaterEqual(np.mean(cov[alpha]), 1 - alpha - 0.005)
            self.assertLessEqual(np.mean(cov[alpha]), 1 - alpha + 0.02)
            self.assertTrue((np.mean(cw[alpha], axis=0) >= 1 - alpha - 0.01).all())


class ExperimentTests(unittest.TestCase):
    def test_conformal_eval_calibrates_on_validation_only(self):
        res, _ = run_seed5(6, n_students=500, weeks=(8,))
        r = res[8]["t2"]
        m, va, te = r["models"]["fh_boost"], r["val"], r["test"]
        out = conformal_eval(m, va, te)["classwise"]
        Pv, Pt = m.predict_proba(va), m.predict_proba(te)
        yv, yt = np.searchsorted(GRADES, va.grade), np.searchsorted(GRADES, te.grade)
        S = cf.prediction_sets(Pt, cf.calibrate(Pv, yv, 0.1, True))
        self.assertAlmostEqual(out["coverage"], cf.coverage(S, yt))
        self.assertAlmostEqual(out["size"], cf.mean_size(S))
        # подмена тестовых меток меняет покрытие, но не порог: набор определяется только валидацией и P(test)
        te2 = te.copy()
        te2["grade"] = 5
        out2 = conformal_eval(m, va, te2)["classwise"]
        self.assertAlmostEqual(out2["size"], out["size"])

    def test_cascade_calibrated_on_validation(self):
        res, _ = run_seed5(6, n_students=500, weeks=(8,))
        r = res[8]["t2"]
        va = r["val"]
        for name in ("fh_logit", "fh_boost", "fh_boost_ze"):
            P = r["models"][name].predict_proba(va)
            # изотоническая регрессия сохраняет среднее: на валидации P(оценка ≥ 3) совпадает с долей
            self.assertAlmostEqual(float(np.mean(1 - P[:, 0])), float(np.mean(va.grade >= 3)), places=6, msg=name)

    def test_run_seed5_smoke(self):
        res, mze = run_seed5(5, n_students=600, weeks=(8,))
        r = res[8]
        m = r["t2"]["metrics"]
        self.assertGreater(m["fh_boost"]["argmax_qwk"], 0.5)
        self.assertLess(m["prior"]["argmax_qwk"], 0.05)
        self.assertLess(m["fh_boost"]["cost_mincost"], m["fh_boost"]["cost_argmax"])    # решение по стоимости помогает
        self.assertGreater(m["fh_boost"]["recall_2_mincost"], m["fh_boost"]["argmax_recall_2"])
        self.assertGreater(r["t3"]["metrics"]["logit"]["roc_auc"], 0.8)
        cw = r["t2"]["conformal"]["classwise"]
        self.assertTrue(all(c >= 0.8 for c in cw["coverage_by_class"]), cw)
        # студенты в test не встречаются в val
        self.assertEqual(set(r["t2"]["val"].student_id) & set(r["t2"]["test"].student_id), set())


if __name__ == "__main__":
    unittest.main()
