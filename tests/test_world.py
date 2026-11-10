import unittest

import numpy as np

from edurisk.curriculum import CHECKPOINTS, CurriculumConfig, GRADED, build_curriculum, checkpoint_mask
from edurisk.features import FEATURES, build_features
from edurisk.world import WorldConfig, simulate_world


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


class CurriculumTests(unittest.TestCase):
    def test_structure(self):
        cur = build_curriculum(CurriculumConfig(n_semesters=6), seed=3)
        self.assertEqual(len(cur), 30)
        self.assertEqual(set(cur.form), {"exam", "difzachet", "zachet", "coursework"})
        self.assertEqual((cur[cur.semester == 0].form == "coursework").sum(), 0)
        self.assertTrue(((cur.form == "coursework").groupby(cur.semester).sum()[1:] == 1).all())
        self.assertTrue((cur.prereq < np.arange(len(cur))).all())
        for c, p in cur.prereq[cur.prereq >= 0].items():
            self.assertEqual(cur.semester[p], cur.semester[c] - 1)
            self.assertNotEqual(cur.form[p], "coursework")
        for _, r in cur.iterrows():
            self.assertGreaterEqual(r.ze, 2)
            self.assertLessEqual(r.ze, 6)

    def test_deterministic_and_masks(self):
        self.assertTrue(build_curriculum(seed=1).equals(build_curriculum(seed=1)))
        for form, weeks in CHECKPOINTS.items():
            m = checkpoint_mask(form)
            self.assertEqual(list(np.flatnonzero(m) + 1), list(weeks))
        with self.assertRaises(ValueError):
            checkpoint_mask("lab")


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = simulate_world(3000, seed=5)
        cls.cfg = WorldConfig()

    def test_deterministic(self):
        a, b = simulate_world(200, seed=2), simulate_world(200, seed=2)
        np.testing.assert_array_equal(a.logins, b.logins)
        np.testing.assert_array_equal(a.grade, b.grade)
        np.testing.assert_array_equal(a.passed, b.passed)
        self.assertFalse(np.array_equal(a.logins, simulate_world(200, seed=3).logins, equal_nan=True))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            simulate_world(100, scenario="S9")
        with self.assertRaises(ValueError):
            simulate_world(1)

    def test_enrollment_and_dropout_consistency(self):
        w = self.w
        sem = w.curriculum.semester.to_numpy()
        left = w.truth["student"]["left_semester"]
        for s in range(sem.max() + 1):
            cs = np.flatnonzero(sem == s)
            should = (left < 0) | (left >= s)               # учился, пока не ушёл в более раннем семестре
            np.testing.assert_array_equal(w.enrolled[:, cs], np.repeat(should[:, None], len(cs), 1))
            np.testing.assert_array_equal(w.dropped[:, cs], np.repeat((left == s)[:, None], len(cs), 1))
        self.assertTrue((w.dropped <= w.enrolled).all())

    def test_no_activity_after_leaving_and_nan_when_not_enrolled(self):
        w = self.w
        sem = w.curriculum.semester.to_numpy()
        left = w.truth["student"]["left_semester"]
        lw = w.truth["student"]["left_week"]
        for i in np.flatnonzero(left >= 0)[:200]:
            cs = np.flatnonzero(sem == left[i])
            self.assertTrue((w.logins[i, cs, lw[i]:] == 0).all())
            self.assertTrue(np.isnan(w.cp_score[i, cs, lw[i]:]).all())
            later = np.flatnonzero(sem > left[i])
            self.assertTrue(np.isnan(w.logins[i, later]).all())
        un = ~w.enrolled
        self.assertTrue(np.isnan(w.logins[un]).all() and np.isnan(w.cp_score[un]).all())
        self.assertFalse(w.passed[un].any() or w.admitted[un].any())
        self.assertTrue(np.isnan(w.grade[un]).all())

    def test_scores_only_on_scheduled_weeks(self):
        w = self.w
        off = ~w.cp_scheduled
        for c in range(w.n_courses):
            self.assertTrue(np.isnan(w.cp_score[:, c, off[c]]).all())
        sc = w.cp_score[~np.isnan(w.cp_score)]
        self.assertTrue(((sc >= 0) & (sc <= 100)).all())

    def test_grades_and_results_are_consistent(self):
        w = self.w
        forms = w.curriculum.form.to_numpy()
        graded = np.isin(forms, GRADED)[None, :] & w.enrolled
        zach = (forms == "zachet")[None, :] & w.enrolled
        self.assertTrue(np.isin(w.grade[graded], [2, 3, 4, 5]).all())
        self.assertTrue(np.isnan(w.grade[zach]).all())
        np.testing.assert_array_equal(w.passed[graded], w.grade[graded] >= 3)
        self.assertTrue((~w.passed[w.enrolled & ~w.admitted]).all())     # недопуск/отчисление = провал
        self.assertTrue((w.grade[w.enrolled & ~w.admitted & graded] == 2).all())
        self.assertTrue((w.grade[w.dropped & graded] == 2).all())
        self.assertFalse(w.admitted[w.dropped].any())

    def test_admission_rule_against_naive_loop(self):
        w = self.w
        rng = np.random.default_rng(0)
        for _ in range(400):
            i, c = rng.integers(w.n_students), rng.integers(w.n_courses)
            if not w.enrolled[i, c] or w.dropped[i, c]:
                continue
            weeks = np.flatnonzero(w.cp_scheduled[c])
            ok = sum(1 for t in weeks if w.cp_score[i, c, t] >= w.pass_score)
            self.assertEqual(bool(w.admitted[i, c]), ok / len(weeks) >= self.cfg.admit_share)

    def test_retakes_only_after_first_attempt_failure(self):
        w = self.w
        has = ~np.isnan(w.retake_passed)
        self.assertTrue(has.sum() > 100)
        self.assertTrue((w.enrolled[has] & ~w.passed[has] & ~w.dropped[has]).all())
        np.testing.assert_array_equal(has, w.enrolled & ~w.passed & ~w.dropped)
        self.assertGreater(np.nanmean(w.retake_passed), 0.4)   # пересдают лучше, но не все

    def test_outcome_shares_are_realistic(self):
        L = self.w.labels()
        self.assertEqual(len(L), self.w.enrolled.sum())
        self.assertTrue(0.14 <= L.fail.mean() <= 0.26, L.fail.mean())
        for form, g in L.groupby("form"):
            self.assertTrue(0.08 <= g.fail.mean() <= 0.30, (form, g.fail.mean()))
        adm = L[~L.dropped].admitted.mean()
        self.assertTrue(0.82 <= adm <= 0.95, adm)
        dropped_students = (self.w.truth["student"]["left_semester"] >= 0).mean()
        self.assertTrue(0.08 <= dropped_students <= 0.22, dropped_students)
        g = L[L.grade.notna() & L.admitted].grade
        self.assertGreater((g == 4).mean(), (g == 3).mean())
        self.assertLess((g == 2).mean(), 0.06)

    def test_causal_signs(self):
        w = self.w
        ab = w.truth["student"]["ability"]
        fail = ~w.passed & w.enrolled
        mean_fail_low = fail[ab < -1].sum() / w.enrolled[ab < -1].sum()
        mean_fail_high = fail[ab > 1].sum() / w.enrolled[ab > 1].sum()
        self.assertGreater(mean_fail_low, 2 * mean_fail_high)
        # пререквизит: успеваемость в предшествующем курсе коррелирует с последующим
        cur, perf = w.curriculum, w.truth["perf"]
        cs = [c for c in range(len(cur)) if cur.prereq[c] >= 0]
        r = []
        for c in cs:
            m = w.enrolled[:, c]
            r.append(np.corrcoef(perf[m, c], perf[m, cur.prereq[c]])[0, 1])
        self.assertGreater(np.mean(r), 0.3)
        # трудные курсы валят чаще
        L = w.labels()
        rate = L.groupby("course").fail.mean()
        self.assertGreater(np.corrcoef(rate.to_numpy(), cur.difficulty.to_numpy()[rate.index])[0, 1], 0.3)

    def test_zachet_result_rule(self):
        w = self.w
        z = np.flatnonzero((w.curriculum.form == "zachet").to_numpy())
        for c in z:
            m = w.enrolled[:, c]
            expect = w.admitted[m, c] & (w.truth["perf"][m, c] >= self.cfg.zachet_cut)
            np.testing.assert_array_equal(w.passed[m, c], expect)
        L = w.labels()
        zz = L[(L.form == "zachet") & L.admitted]
        self.assertGreater(zz.fail.mean(), 0.005)        # допущенные тоже иногда не получают зачёт

    def test_prerequisite_effect_is_recovered(self):
        w, cur = self.w, self.w.curriculum
        sem_of = cur.semester.to_numpy()
        rows, ys = [], []
        for c in range(len(cur)):
            p = cur.prereq[c]
            if p < 0:
                continue
            m = w.enrolled[:, c]
            s = sem_of[c]
            mot = w.truth["motivation"]
            X = np.column_stack([np.ones(m.sum()), w.truth["student"]["ability"][m], mot[m, s], mot[m, s - 1],
                                 np.full(m.sum(), cur.difficulty[c]), w.truth["perf"][m, p]])
            rows.append(X)
            ys.append(w.truth["perf"][m, c])
        X, y = np.vstack(rows), np.concatenate(ys)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        self.assertTrue(0.08 < coef[-1] < 0.22, coef[-1])     # заложено 0.15

    def test_difficulty_lowers_latent_performance(self):
        w, cur = self.w, self.w.curriculum
        mean_perf = np.array([w.truth["perf"][w.enrolled[:, c], c].mean() for c in range(len(cur))])
        self.assertLess(np.corrcoef(mean_perf, cur.difficulty.to_numpy())[0, 1], -0.5)

    def test_history_for_a_student_who_left(self):
        w = self.w
        sem = w.curriculum.semester.to_numpy()
        i = int(np.flatnonzero(w.truth["student"]["left_semester"] == 1)[0])
        h = w.history_before(3)
        fails = sum(int(not w.passed[i, c]) for c in np.flatnonzero(sem < 3) if w.enrolled[i, c])
        self.assertEqual(h.prior_fails[i], fails)
        ze = sum(w.curriculum.ze[c] for c in np.flatnonzero(sem < 3) if w.enrolled[i, c] and w.passed[i, c])
        self.assertEqual(h.prior_ze[i], ze)

    def test_course_view_is_stage1_compatible_and_leak_free(self):
        w = self.w
        c = 2
        coh = w.course_view(c)
        self.assertEqual(coh.n_students, w.enrolled[:, c].sum())
        np.testing.assert_array_equal(coh.outcome, ~w.passed[w.enrolled[:, c], c])
        X = build_features(coh, 8)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertFalse(X.isna().all().any() and False)
        # перемешиваем данные после недели 8 — признаки недели 8 не меняются
        w2 = simulate_world(3000, seed=5)
        rng = np.random.default_rng(1)
        w2.logins[:, c, 8:] = rng.integers(0, 40, w2.logins[:, c, 8:].shape)
        w2.cp_score[:, c, 8:] = np.where(w2.cp_scheduled[c, 8:], rng.uniform(0, 100, w2.cp_score[:, c, 8:].shape), np.nan)
        X2 = build_features(w2.course_view(c), 8)
        np.testing.assert_allclose(X.to_numpy(float), X2.to_numpy(float), equal_nan=True)

    def test_history_before_uses_only_earlier_semesters(self):
        w = self.w
        h0 = w.history_before(0)
        self.assertTrue(h0.prior_gpa.isna().all() and (h0.prior_fails == 0).all())
        sem = w.curriculum.semester.to_numpy()
        h2 = w.history_before(2)
        # вручную для одного студента
        i = int(np.flatnonzero(w.enrolled[:, sem == 1].all(1))[0])
        cs = np.flatnonzero(sem < 2)
        num = den = 0.0
        fails = 0
        for c in cs:
            if w.enrolled[i, c]:
                fails += int(not w.passed[i, c])
                if not np.isnan(w.grade[i, c]):
                    num += w.grade[i, c] * w.curriculum.ze[c]
                    den += w.curriculum.ze[c]
        self.assertAlmostEqual(h2.prior_gpa[i], num / den)
        self.assertEqual(h2.prior_fails[i], fails)
        # подмена итогов семестров 2+ не меняет историю к семестру 2
        w3 = simulate_world(3000, seed=5)
        late = np.flatnonzero(sem >= 2)
        w3.grade[:, late] = 2.0
        w3.passed[:, late] = False
        h2b = w3.history_before(2)
        np.testing.assert_array_equal(h2.to_numpy(float), h2b.to_numpy(float))
        self.assertFalse(np.array_equal(h2.to_numpy(float), w3.history_before(3).to_numpy(float), equal_nan=True))


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = simulate_world(4000, seed=7)

    def test_s0_equals_s2_cohort0(self):
        w = simulate_world(4000, seed=7, scenario="S2", cohort_year=0)
        np.testing.assert_array_equal(w.passed, self.base.passed)
        np.testing.assert_array_equal(w.logins, self.base.logins)

    def test_s1_has_threshold_feedback_jumps(self):
        s1 = simulate_world(4000, seed=7, scenario="S1")
        d0 = np.diff(self.base.truth["feedback"], axis=2).min()
        d1 = np.diff(s1.truth["feedback"], axis=2).min()
        K = 5
        self.assertGreaterEqual(d0, -0.1 * K - 1e-9)             # в S0 падение пропорционально числу провалов
        self.assertLess(d1, -0.95)                               # в S1 есть скачок
        # скачки случаются только после серии из двух провалов подряд -> заметно меньше «мелких» шагов
        small = lambda w: (np.diff(w.truth["feedback"], axis=2) < -1e-9).mean()
        self.assertLess(small(s1), small(self.base))

    def test_s2_shift_makes_cohort0_model_miscalibrated(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        c = 0
        a = simulate_world(4000, seed=8, scenario="S2", cohort_year=0).course_view(c)
        b = simulate_world(4000, seed=8, scenario="S2", cohort_year=1).course_view(c)
        cols = ["logins_total", "logins_last4", "cp_failed_count", "mean_score"]
        Xa, Xb = build_features(a, 12)[cols].fillna(0), build_features(b, 12)[cols].fillna(0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xa, a.outcome)
        gap0 = model.predict_proba(Xa)[:, 1].mean() - a.outcome.mean()
        gap1 = model.predict_proba(Xb)[:, 1].mean() - b.outcome.mean()
        self.assertLess(abs(gap0), 0.01)
        self.assertLess(gap1, -0.02)                              # модель недооценивает риск новой когорты
        self.assertGreater(b.outcome.mean(), a.outcome.mean() + 0.03)
        self.assertLess(np.nanmean(b.cp_score), np.nanmean(a.cp_score) - 2)

    def test_s3_measurement_bias_without_outcome_bias(self):
        s3 = simulate_world(4000, seed=7, scenario="S3")
        o = (s3.students.origin == "other_region").to_numpy()
        act = s3.enrolled & ~s3.dropped
        mean_l = lambda w, m: np.nanmean(w.logins[m][act[m]])
        ratio3 = mean_l(s3, o) / mean_l(s3, ~o)
        ratio0 = mean_l(self.base, o) / mean_l(self.base, ~o)
        self.assertTrue(0.70 <= ratio3 <= 0.85, ratio3)
        self.assertTrue(0.95 <= ratio0 <= 1.05, ratio0)
        fail = lambda w, m: (~w.passed[m] & w.enrolled[m]).sum() / w.enrolled[m].sum()
        self.assertLess(abs(fail(s3, o) - fail(s3, ~o)), 0.02)       # успеваемость группы та же

    def test_s4_loses_records_but_not_outcomes(self):
        s4 = simulate_world(4000, seed=7, scenario="S4")
        np.testing.assert_array_equal(s4.passed, self.base.passed)
        np.testing.assert_array_equal(s4.grade, self.base.grade)
        m = self.base.enrolled & ~self.base.dropped
        lost = np.isnan(s4.logins[m]).mean()
        self.assertTrue(0.12 <= lost <= 0.18, lost)
        had = ~np.isnan(self.base.cp_score)
        lost_s = (np.isnan(s4.cp_score) & had).sum() / had.sum()
        self.assertTrue(0.07 <= lost_s <= 0.13, lost_s)
        self.assertTrue(np.isnan(self.base.logins[~self.base.enrolled]).all())


if __name__ == "__main__":
    unittest.main()
