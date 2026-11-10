import unittest

import numpy as np
import pandas as pd

from edurisk.features import AUDIT, EWMA_ALPHA, FEATURES, build_features, build_panel
from edurisk.simulate import Cohort, simulate


def reference_features(c: Cohort, week: int) -> pd.DataFrame:
    """Независимая реализация «в лоб»: по одному студенту, списки и циклы, без векторизации."""
    rows = []
    sched = [bool(x) for x in c.cp_scheduled[:week]]
    cps = [w for w in range(week) if sched[w]]
    for i in range(c.n_students):
        L = [int(x) for x in c.logins[i, :week]]
        M = [float(x) for x in c.minutes[i, :week]]
        S = [float(x) for x in c.cp_score[i, :week]]
        row = {k: float(c.students.iloc[i][k]) for k in ("ege", "workload_ze", "has_job")}
        row["logins_total"] = sum(L)
        row["logins_last2"] = sum(L[-2:])
        row["logins_last4"] = sum(L[-4:])
        row["minutes_mean4"] = sum(M[-4:]) / len(M[-4:])
        failed = []
        for w in cps:
            s = S[w]
            failed.append(True if np.isnan(s) else s < c.pass_score)
        row["cp_failed_count"] = sum(failed)
        row["cp_missed_count"] = sum(1 for w in cps if np.isnan(S[w]))
        row["last_score"] = S[cps[-1]] if cps else np.nan
        got = [S[w] for w in cps if not np.isnan(S[w])]
        got_all = [s for s in S if not np.isnan(s)]
        row["mean_score"] = sum(got_all) / len(got_all) if got_all else np.nan
        last4 = L[-4:]
        row["activity_trend4"] = float(np.polyfit(range(len(last4)), last4, 1)[0]) if len(last4) > 1 else 0.0
        e = L[0]
        for x in L[1:]:
            e = EWMA_ALPHA * x + (1 - EWMA_ALPHA) * e
        row["logins_ewma"] = e
        k = 0
        for x in reversed(L):
            if x > 0:
                break
            k += 1
        row["weeks_since_login"] = k
        row["zero_login_weeks"] = sum(1 for x in L if x == 0)
        row["cp_fail_rate"] = sum(failed) / len(cps) if cps else 0.0
        cur = best = 0
        for f in failed:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        row["cp_fail_streak"], row["cp_max_fail_streak"] = cur, best
        row["score_delta"] = got_all[-1] - got_all[-2] if len(got_all) >= 2 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows, index=c.students.index)
    m4 = df["logins_last4"].mean()
    df["activity_vs_cohort"] = df["logins_last4"] / m4 if m4 > 0 else 0.0
    return df[FEATURES].astype(float)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = simulate(400, seed=3)

    def test_matches_reference_for_every_week(self):
        for week in range(1, 17):
            got = build_features(self.c, week).astype(float)
            want = reference_features(self.c, week)
            pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9, atol=1e-9,
                                          obj=f"week {week}")

    def test_no_leakage_future_data_does_not_change_features(self):
        rng = np.random.default_rng(0)
        for week in (1, 4, 8, 11):
            before = build_features(self.c, week)
            c2 = simulate(400, seed=3)
            c2.logins[:, week:] = rng.integers(0, 50, c2.logins[:, week:].shape)
            c2.minutes[:, week:] = rng.uniform(0, 500, c2.minutes[:, week:].shape)
            c2.cp_score[:, week:] = rng.uniform(0, 100, c2.cp_score[:, week:].shape)
            c2.outcome = ~c2.outcome
            after = build_features(c2, week)
            pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_past_data_does_change_features(self):
        """Тест чувствительности: иначе «нет утечки» ничего бы не значило (признаки были бы константой)."""
        week = 8
        before = build_features(self.c, week)
        c2 = simulate(400, seed=3)
        c2.logins[:, :week] += 7
        after = build_features(c2, week)
        self.assertFalse(before["logins_total"].equals(after["logins_total"]))
        self.assertTrue((after["logins_last4"] - before["logins_last4"] == 28).all())

    def test_features_do_not_include_outcome_or_audit_attributes(self):
        self.assertFalse({"target", "outcome", *AUDIT} & set(FEATURES))
        self.assertEqual(len(FEATURES), len(set(FEATURES)))

    def test_fail_streak_hand_example(self):
        c = simulate(5, seed=1)
        c.cp_score[:] = np.nan
        # контрольные на неделях 2,4,6,8,10,12,14,16; ряд: пройдена, провал, провал, пройдена, провал, провал, провал
        scores = [80, 40, 30, 90, 20, 10, 5]
        for s, w in zip(scores, (2, 4, 6, 8, 10, 12, 14)):
            c.cp_score[0, w - 1] = s
        f = build_features(c, 14)
        self.assertEqual(f.cp_fail_streak.iloc[0], 3)
        self.assertEqual(f.cp_max_fail_streak.iloc[0], 3)
        f = build_features(c, 8)
        self.assertEqual(f.cp_fail_streak.iloc[0], 0)
        self.assertEqual(f.cp_max_fail_streak.iloc[0], 2)

    def test_missed_checkpoint_breaks_nothing_and_counts_as_failure(self):
        c = simulate(5, seed=1)
        c.cp_score[:] = np.nan
        c.cp_score[0, 1] = 90          # неделя 2 пройдена; недели 4, 6 пропущены
        f = build_features(c, 6)
        self.assertEqual(f.cp_fail_streak.iloc[0], 2)
        self.assertEqual(f.cp_missed_count.iloc[0], 2)
        self.assertEqual(f.last_score.iloc[0] != f.last_score.iloc[0], True)   # NaN: последняя пропущена

    def test_no_checkpoint_yet(self):
        f = build_features(self.c, 1)
        self.assertTrue((f.cp_fail_rate == 0).all() and (f.cp_fail_streak == 0).all())
        self.assertTrue(f.last_score.isna().all() and f.score_delta.isna().all())

    def test_weeks_since_login_and_trend(self):
        c = simulate(3, seed=2)
        c.logins[:] = 0
        c.logins[0, :6] = [1, 2, 3, 4, 0, 0]
        c.logins[1, :6] = [4, 3, 2, 1, 5, 5]
        f = build_features(c, 6)
        self.assertEqual(f.weeks_since_login.iloc[0], 2)
        self.assertEqual(f.weeks_since_login.iloc[2], 6)       # не заходил ни разу
        self.assertAlmostEqual(f.activity_trend4.iloc[0], np.polyfit(range(4), [3, 4, 0, 0], 1)[0])
        self.assertEqual(f.zero_login_weeks.iloc[2], 6)

    def test_ewma_hand_example(self):
        c = simulate(3, seed=2)
        c.logins[:] = 0
        c.logins[0, :2] = [10, 0]
        c.logins[1, :3] = [0, 10, 0]
        f = build_features(c, 2)
        self.assertAlmostEqual(f.logins_ewma.iloc[0], 0.4 * 0 + 0.6 * 10)     # 6.0: свежая неделя весит 0.4
        f = build_features(c, 3)
        self.assertAlmostEqual(f.logins_ewma.iloc[1], 0.4 * 0 + 0.6 * (0.4 * 10 + 0.6 * 0))   # 2.4

    def test_panel_shape(self):
        p = build_panel(self.c, [4, 8])
        self.assertEqual(len(p), 800)
        self.assertEqual(set(p.week), {4, 8})
        self.assertEqual(int(p.target.sum()), 2 * int(self.c.outcome.sum()))


if __name__ == "__main__":
    unittest.main()
