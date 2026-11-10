import dataclasses
import unittest

import numpy as np

from edurisk.simulate import SimConfig, simulate


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = simulate(3000, seed=5)

    def test_deterministic_for_seed(self):
        a, b = simulate(300, seed=9), simulate(300, seed=9)
        np.testing.assert_array_equal(a.logins, b.logins)
        np.testing.assert_array_equal(a.cp_score, b.cp_score)
        np.testing.assert_array_equal(a.outcome, b.outcome)
        self.assertFalse(np.array_equal(a.logins, simulate(300, seed=10).logins))

    def test_shapes_and_ranges(self):
        c = self.c
        self.assertEqual(c.logins.shape, (3000, 16))
        self.assertTrue((c.logins >= 0).all() and (c.minutes >= 0).all())
        sc = c.cp_score[~np.isnan(c.cp_score)]
        self.assertTrue(((sc >= 0) & (sc <= 100)).all())
        self.assertTrue(((c.students.ege >= 30) & (c.students.ege <= 100)).all())

    def test_scores_only_in_checkpoint_weeks(self):
        off = ~self.c.cp_scheduled
        self.assertTrue(np.isnan(self.c.cp_score[:, off]).all())

    def test_left_students_have_no_activity_afterwards(self):
        c = self.c
        left = c.truth.left_at.to_numpy()
        for i in np.flatnonzero(left < c.n_weeks)[:300]:
            self.assertEqual(c.logins[i, left[i]:].sum(), 0)
            self.assertEqual(c.minutes[i, left[i]:].sum(), 0)
            self.assertTrue(np.isnan(c.cp_score[i, left[i]:]).all())

    def test_nobody_leaves_in_first_three_weeks(self):
        self.assertTrue((self.c.truth.left_at >= 3).all())

    def test_everyone_who_left_is_marked_failed(self):
        self.assertTrue(self.c.outcome[self.c.truth.dropped.to_numpy()].all())

    def test_realistic_base_rates(self):
        self.assertTrue(0.18 < self.c.outcome.mean() < 0.40, self.c.outcome.mean())
        self.assertTrue(0.08 < self.c.truth.dropped.mean() < 0.30)

    def test_missed_checkpoint_counts_as_failed(self):
        c = self.c
        failed = c.cp_failed()
        missed = c.cp_scheduled[None, :] & np.isnan(c.cp_score)
        self.assertTrue(failed[missed].all())
        self.assertFalse(failed[:, ~c.cp_scheduled].any())

    def test_ege_is_a_noisy_proxy_of_ability(self):
        r = np.corrcoef(self.c.students.ege, self.c.truth.ability)[0, 1]
        self.assertTrue(0.4 < r < 0.8, r)

    def test_disengaging_students_lose_activity(self):
        c, t = self.c, self.c.truth
        early = c.logins[:, :3].mean(1)
        late = c.logins[:, -3:].mean(1)
        drop = (late - early)[t.disengaging.to_numpy()].mean()
        keep = (late - early)[~t.disengaging.to_numpy()].mean()
        self.assertLess(drop, keep - 0.8)

    def test_observed_cuts_the_future(self):
        o = self.c.observed(5)
        self.assertEqual(o.logins.shape[1], 5)
        self.assertEqual(len(o.cp_scheduled), 5)
        self.assertIsNone(o.truth)
        np.testing.assert_array_equal(o.logins, self.c.logins[:, :5])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            simulate(1)
        with self.assertRaises(ValueError):
            self.c.observed(0)
        with self.assertRaises(ValueError):
            self.c.observed(17)
        with self.assertRaises(ValueError):
            simulate(10, cfg=dataclasses.replace(SimConfig(), checkpoint_weeks=(2, 20)))


if __name__ == "__main__":
    unittest.main()
