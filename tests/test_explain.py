import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from edurisk.dataset import MODEL_FEATURES, build_dataset, split_students, take
from edurisk.explain import (GROUPS, GROUP_RU, explain_text, group_importance, group_shapley,
                             linear_contributions, what_if)
from edurisk.models import BoostModel, LogitModel
from edurisk.world import simulate_world


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))), columns=MODEL_FEATURES)


class ToyAdditive:
    """f(x) = a·logins_total + b·prior_gpa: вклады групп известны аналитически."""
    def __call__(self, X):
        return 2.0 * X["logins_total"].to_numpy() + 0.5 * X["prior_gpa"].to_numpy()


class GroupShapleyTests(unittest.TestCase):
    def test_groups_cover_features_exactly_once(self):
        flat = [f for g in GROUPS.values() for f in g]
        self.assertEqual(sorted(flat), sorted(MODEL_FEATURES))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(GROUPS), set(GROUP_RU))

    def test_additive_model_matches_analytic_values(self):
        bg, row = _frame(40, 1), _frame(1, 2)
        c = group_shapley(ToyAdditive(), row, bg)
        self.assertAlmostEqual(c["activity"], 2.0 * (row.logins_total[0] - bg.logins_total.mean()))
        self.assertAlmostEqual(c["history"], 0.5 * (row.prior_gpa[0] - bg.prior_gpa.mean()))
        for g in ("assessments", "background", "course"):
            self.assertAlmostEqual(c[g], 0.0, places=12)

    def test_efficiency_on_nonadditive_model(self):
        bg, row = _frame(30, 3), _frame(1, 4)
        f = lambda X: (1 / (1 + np.exp(-(X["logins_total"] * X["cp_failed_count"] + X["ege"] ** 2 - X["ze"])))).to_numpy()
        c = group_shapley(f, row, bg)
        self.assertAlmostEqual(sum(v for k, v in c.items() if not k.startswith("_")), c["_pred"] - c["_base"], places=10)
        self.assertAlmostEqual(c["_pred"], float(f(row)[0]))
        self.assertAlmostEqual(c["_base"], float(np.mean(f(bg))))

    def test_interaction_is_split_between_groups(self):
        # f = logins_total * cp_failed_count: вклад взаимодействия делится поровну между активностью и контрольными
        bg = pd.DataFrame(0.0, index=range(5), columns=MODEL_FEATURES)
        row = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)
        row["logins_total"], row["cp_failed_count"] = 2.0, 3.0
        c = group_shapley(lambda X: (X["logins_total"] * X["cp_failed_count"]).to_numpy(), row, bg)
        self.assertAlmostEqual(c["activity"], 3.0)
        self.assertAlmostEqual(c["assessments"], 3.0)

    def test_requires_single_row(self):
        with self.assertRaises(ValueError):
            group_shapley(ToyAdditive(), _frame(2), _frame(5))


class LinearAndWhatIfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w = simulate_world(500, seed=8)
        d = build_dataset(w, weeks=(8,))
        parts = split_students(d.student_id.unique(), seed=1)
        cls.tr, cls.te = take(d, parts["train"]), take(d, parts["test"])
        cls.lg = LogitModel().fit(cls.tr, cls.tr.y.to_numpy())
        cls.bs = BoostModel().fit(cls.tr, cls.tr.y.to_numpy())

    def test_linear_contributions_sum_to_logit_minus_intercept(self):
        X = self.te.head(50)
        C = linear_contributions(self.lg, X)
        p = self.lg.predict_proba(X)
        logit = np.log(p / (1 - p))
        np.testing.assert_allclose(C.sum(axis=1).to_numpy() + self.lg.pipe.steps[2][1].intercept_[0], logit, atol=1e-6)
        self.assertEqual(list(C.columns), self.lg.cols_)

    def test_linear_contribution_sign_matches_effect(self):
        C = linear_contributions(self.lg, self.te)
        # у студентов с длинной серией провалов вклад cp_max_fail_streak в риск в среднем положителен
        hi = self.te.cp_max_fail_streak >= 2
        self.assertGreater(C.loc[hi, "cp_max_fail_streak"].mean(), 0)
        self.assertGreater(C.loc[hi, "cp_max_fail_streak"].mean(), C.loc[~hi, "cp_max_fail_streak"].mean())

    def test_what_if_lowers_risk_for_high_risk_student(self):
        bg = self.tr.sample(100, random_state=0)
        row = self.te[self.te.y == 1].iloc[[0]]
        p0 = float(self.bs.predict_proba(row)[0])
        p1 = min(what_if(self.bs.predict_proba, row, bg, g) for g in ("activity", "assessments"))
        self.assertLess(p1, p0)
        # исходная строка не изменена
        self.assertEqual(float(self.bs.predict_proba(row)[0]), p0)

    def test_what_if_substitutes_median_of_background(self):
        bg = _frame(41, 7)
        row = _frame(1, 8)
        f = lambda X: X["logins_total"].to_numpy() + 10 * X["cp_failed_count"].to_numpy()
        self.assertAlmostEqual(what_if(f, row, bg, "activity"),
                               bg.logins_total.median() + 10 * row.cp_failed_count[0] + 0 * 0 + 0)
        # остальные столбцы группы тоже заменены, остальные группы нет
        self.assertAlmostEqual(what_if(lambda X: X["ege"].to_numpy(), row, bg, "activity"), row.ege[0])
        self.assertAlmostEqual(what_if(lambda X: X["logins_ewma"].to_numpy(), row, bg, "activity"), bg.logins_ewma.median())

    def test_group_importance_detects_used_and_ignored_groups(self):
        class OnlyActivity:
            def __call__(self, X):
                return 1 / (1 + np.exp(-X["logins_ewma"].to_numpy()))
        rng = np.random.default_rng(0)
        X = _frame(3000, 5)
        y = (rng.random(3000) < 1 / (1 + np.exp(-X.logins_ewma.to_numpy()))).astype(int)
        imp = group_importance(OnlyActivity(), X, y, roc_auc_score)
        self.assertGreater(imp["activity"], 0.1)
        for g in ("assessments", "history", "background", "course"):
            self.assertAlmostEqual(imp[g], 0.0, places=12)

    def test_boosting_explanation_points_to_behavior_for_low_activity_student(self):
        bg = self.tr.sample(100, random_state=0)
        te = self.te
        low = te[(te.activity_vs_cohort < 0.3) & (te.cp_max_fail_streak >= 1) & (te.y == 1)].iloc[[0]]
        c = group_shapley(self.bs.predict_proba, low, bg)
        top = max((v, g) for g, v in c.items() if not g.startswith("_"))[1]
        self.assertIn(top, ("activity", "assessments"))


class TextTests(unittest.TestCase):
    def _row(self, **kw):
        base = dict(cp_max_fail_streak=0, cp_failed_count=0, cp_missed_count=0, activity_vs_cohort=1.0,
                    weeks_since_login=0, prior_gpa=4.2, prior_fails=0, ege=70.0, ze=4)
        base.update(kw)
        return pd.Series(base)

    def test_text_mentions_facts_and_is_neutral(self):
        contrib = {"activity": 0.2, "assessments": 0.3, "history": 0.0, "background": 0.0, "course": 0.0,
                   "_base": 0.15, "_pred": 0.65}
        t = explain_text(self._row(cp_max_fail_streak=3, cp_failed_count=3, activity_vs_cohort=0.4, weeks_since_login=2), contrib)
        self.assertIn("65%", t)
        self.assertIn("15%", t)
        self.assertIn("подряд контрольные: 3", t)
        self.assertIn("40%", t)
        self.assertIn("высокая", t)
        self.assertIn("решение остаётся за преподавателем", t)
        for bad in ("двоечник", "ленив", "безнадёж", "слаб"):
            self.assertNotIn(bad, t.lower())
        # порядок: контрольные (0.3) раньше активности (0.2)
        self.assertLess(t.index("контрольные точки"), t.index("активность в LMS"))

    def test_no_positive_factors(self):
        contrib = {"activity": -0.1, "assessments": -0.05, "history": 0.0, "background": 0.0, "course": 0.005,
                   "_base": 0.2, "_pred": 0.05}
        t = explain_text(self._row(), contrib)
        self.assertIn("обычная", t)
        self.assertIn("Выраженных факторов повышения нет", t)

    def test_history_fact_without_history(self):
        contrib = {"activity": 0.0, "assessments": 0.0, "history": 0.2, "background": 0.0, "course": 0.0,
                   "_base": 0.2, "_pred": 0.4}
        t = explain_text(self._row(prior_gpa=np.nan), contrib)
        self.assertIn("истории прошлых семестров нет", t)
        t = explain_text(self._row(prior_gpa=2.8, prior_fails=2), contrib)
        self.assertIn("2.8", t)
        self.assertIn("неудачных попыток: 2", t)
        self.assertIn("повышенная", t)


if __name__ == "__main__":
    unittest.main()
