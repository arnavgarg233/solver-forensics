#!/usr/bin/env python3
"""Unit tests for src/measurement/cross_conformal.py (repeated split-conformal).

No FEM solve and no classifier of the caller's own: every test feeds synthetic
paired per-IC feature arrays to the detector, so the file runs in seconds and
checks the properties the estimator has to own.

  * calibration contract: the alarm threshold is the maximum of 19 calibration
    nominal scores, an exact 1/20 = 5% level, and the fold choice produces the
    documented 29/19/12 (n=60) and 21/19/8 (n=48) splits;
  * strict isolation: train, calibration and test IC sets are pairwise disjoint,
    and the ONE model that scores a fold's calibration and test ICs was fitted
    on none of them.  This is what the earlier cross-fitted design failed: its
    calibration scores came from models trained on the current test ICs;
  * level and power: the false-alarm rate is MEASURED on untouched test ICs
    (never imposed), and a separated population is detected;
  * sampled-grid limits: the limit is a sampled delta, right-censored when no
    sampled delta reaches the target, with an IC-cluster bootstrap that
    preserves each IC's cross-delta pairing.

Run:  /tmp/solver-forensics-py310/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import cross_conformal as CC    # the module under test


def _pair(n=60, p=5, shift=0.0, seed=0, tag=False):
    """Paired nominal/changed arrays; ``tag`` writes the IC index into column 0."""
    rng = np.random.default_rng(seed)
    nominal = rng.standard_normal((n, p))
    changed = rng.standard_normal((n, p)) + shift * np.eye(p)[0]
    if tag:
        nominal[:, 0] = np.arange(n)
        changed[:, 0] = np.arange(n)
    return nominal, changed


def _recording_classifier(log, factory):
    """A pipeline that records the IC indices it is fitted on and scores.

    Column 0 of the tagged features carries the IC index and is identical for an
    IC's nominal and changed row, so it is uninformative about the class.
    """
    inner = factory()
    ids = lambda X: set(np.rint(np.asarray(X)[:, 0]).astype(int).tolist())

    class Recorder:
        def fit(self, X, y):
            log.append({"train": ids(X), "scored": set()})
            inner.fit(X, y)
            return self

        def predict_proba(self, X):
            log[-1]["scored"] |= ids(X)
            return inner.predict_proba(X)

    return Recorder()


class TestCalibrationContract(unittest.TestCase):
    def test_level_is_exactly_five_percent(self):
        self.assertEqual((CC.N_CAL, CC.CAL_RANK), (19, 19))
        self.assertAlmostEqual(CC.calibration_level(), 0.05, places=15)
        self.assertAlmostEqual(CC.calibration_level(39, 38), 0.05, places=15)
        with self.assertRaises(ValueError):        # 2/20 is not 5%
            CC.calibration_level(19, 18)
        with self.assertRaises(ValueError):        # 2/21 is not 5%
            CC.calibration_level(20, 19)

    def test_documented_split_sizes(self):
        for n, folds, sizes in ((60, 5, (29, 19, 12)), (48, 6, (21, 19, 8))):
            self.assertEqual(CC.choose_folds(n), folds, n)
            plan = CC.fold_plan(n, repeats=1, seed=0)[0]
            self.assertEqual(len(plan), folds)
            for cut in plan:
                self.assertEqual((cut["train"].size, cut["cal"].size,
                                  cut["test"].size), sizes, n)

    def test_fold_choice_keeps_a_calibration_set_and_a_training_set(self):
        for n in range(40, 81):
            plan = CC.fold_plan(n, repeats=1, seed=1)[0]
            for cut in plan:
                self.assertEqual(cut["cal"].size, CC.N_CAL, n)
                self.assertGreaterEqual(cut["train"].size, CC.MIN_TRAIN_IC, n)
                self.assertEqual(cut["train"].size + cut["cal"].size
                                 + cut["test"].size, n, n)

    def test_rejects_unequal_populations_and_small_samples(self):
        nominal, changed = _pair(n=60)
        with self.assertRaises(ValueError):        # unequal populations
            CC.split_conformal_detection(nominal, changed[:59])
        with self.assertRaises(ValueError):        # unequal feature counts
            CC.split_conformal_detection(nominal, changed[:, :4])
        with self.assertRaises(ValueError):        # too few ICs to split three ways
            CC.split_conformal_detection(*_pair(n=30))
        with self.assertRaises(ValueError):
            CC.choose_folds(39)
        with self.assertRaises(ValueError):        # no training ICs left
            CC.fold_plan(60, folds=2)


class TestStrictIsolation(unittest.TestCase):
    def test_train_calibration_and_test_are_pairwise_disjoint(self):
        n = 60
        for repeat in CC.fold_plan(n, repeats=4, seed=7):
            covered = []
            for cut in repeat:
                train, cal, test = cut["train"], cut["cal"], cut["test"]
                self.assertEqual(np.intersect1d(train, cal).size, 0)
                self.assertEqual(np.intersect1d(train, test).size, 0)
                self.assertEqual(np.intersect1d(cal, test).size, 0)
                np.testing.assert_array_equal(
                    np.union1d(np.union1d(train, cal), test), np.arange(n))
                covered.append(test)
            np.testing.assert_array_equal(np.sort(np.concatenate(covered)),
                                          np.arange(n))

    def test_no_fitted_model_sees_calibration_or_test_ics(self):
        # the decisive check on the earlier design: the model that supplies a
        # fold's calibration scores must also be the model that scores the test
        # ICs, and must be fitted on neither
        n, log = 60, []
        original = CC._classifier
        CC._classifier = lambda: _recording_classifier(log, original)
        try:
            CC.split_conformal_detection(*_pair(n=n, shift=1.0, tag=True),
                                         seed=2, repeats=2)
        finally:
            CC._classifier = original
        self.assertEqual(len(log), 2 * CC.choose_folds(n))
        for record in log:
            train, scored = record["train"], record["scored"]
            self.assertEqual(train & scored, set())
            self.assertEqual(len(scored), CC.N_CAL + n // CC.choose_folds(n))
            self.assertEqual(len(train), n - len(scored))
            self.assertEqual(train | scored, set(range(n)))

    def test_determinism_of_the_plan(self):
        again = CC.fold_plan(60, repeats=3, seed=7)
        other = CC.fold_plan(60, repeats=3, seed=8)
        for a, b in zip(CC.fold_plan(60, repeats=3, seed=7), again):
            for fa, fb in zip(a, b):
                for key in ("train", "cal", "test"):
                    np.testing.assert_array_equal(fa[key], fb[key])
        self.assertFalse(all(
            np.array_equal(fa["test"], fb["test"])
            for a, b in zip(again, other) for fa, fb in zip(a, b)))


class TestMeasuredLevelAndPower(unittest.TestCase):
    def test_null_pair_measures_a_bounded_false_alarm_rate(self):
        result = CC.split_conformal_detection(*_pair(n=60, shift=0.0, seed=1),
                                              seed=3)
        self.assertEqual((result["n"], result["folds"], result["repeats"]),
                         (60, 5, CC.REPEATS))
        self.assertAlmostEqual(result["fpr_bound"], 0.05, places=15)
        # measured, not imposed: bounded by the conformal level up to the
        # Monte-Carlo noise of 60 independent ICs
        self.assertLessEqual(result["fpr"], 0.09)
        self.assertLessEqual(result["tpr"], 0.20)     # no power under the null
        self.assertEqual(result["detect"].shape, (60,))
        self.assertTrue(np.all((result["alarm"] >= 0.0) & (result["alarm"] <= 1.0)))
        self.assertAlmostEqual(result["tpr"], float(np.mean(result["detect"])), places=12)
        self.assertAlmostEqual(result["fpr"], float(np.mean(result["alarm"])), places=12)
        lo, hi = result["fpr_ci"]
        self.assertLessEqual(lo, result["fpr"])
        self.assertLessEqual(result["fpr"], hi)

    def test_separated_pair_is_detected_at_the_measured_level(self):
        result = CC.split_conformal_detection(*_pair(n=60, shift=8.0, seed=2),
                                              seed=4)
        self.assertGreaterEqual(result["tpr"], 0.99)
        self.assertGreaterEqual(result["tpr_ci"][0], 0.9)
        self.assertLessEqual(result["fpr"], 0.09)

    def test_scalar_and_paired_feature_columns(self):
        rng = np.random.default_rng(5)
        nominal = rng.standard_normal(60)
        changed = rng.standard_normal(60) + 6.0
        scalar = CC.split_conformal_detection(nominal, changed, seed=6)
        self.assertGreaterEqual(scalar["tpr"], 0.95)
        joint = CC.split_conformal_detection(
            np.column_stack([nominal, 3.0 * nominal]),
            np.column_stack([changed, 3.0 * changed]), seed=6)
        self.assertGreaterEqual(joint["tpr"], 0.95)

    def test_determinism_for_a_seed(self):
        pair = _pair(n=48, p=2, shift=1.5, seed=9)
        first = CC.split_conformal_detection(*pair, seed=11)
        again = CC.split_conformal_detection(*pair, seed=11)
        self.assertEqual(first["folds"], 6)
        np.testing.assert_array_equal(first["detect"], again["detect"])
        np.testing.assert_array_equal(first["alarm"], again["alarm"])
        self.assertEqual(first["tpr_ci"], again["tpr_ci"])


class TestClusterBootstrap(unittest.TestCase):
    def test_effective_sample_size_is_the_number_of_ics(self):
        # per-IC indicators only: repeats, folds and noise replicates must not
        # enter the width
        rng = np.random.default_rng(13)
        for n in (48, 60):
            values = (rng.random(n) < 0.5).astype(float)
            lo, hi = CC.cluster_bootstrap_ci(values, seed=2)
            expected = 2.0 * 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)
            self.assertAlmostEqual(hi - lo, expected, delta=0.35 * expected)
            self.assertLessEqual(lo, float(np.mean(values)))
            self.assertLessEqual(float(np.mean(values)), hi)

    def test_shared_resample_across_columns_and_determinism(self):
        values = np.column_stack([np.arange(60.0), np.arange(60.0)])
        ci = CC.cluster_bootstrap_ci(values, seed=1)
        self.assertEqual(ci.shape, (2, 2))
        np.testing.assert_array_equal(ci[0], ci[1])     # identical columns
        np.testing.assert_array_equal(ci, CC.cluster_bootstrap_ci(values, seed=1))


class TestSampledLimit(unittest.TestCase):
    def test_smallest_sampled_delta_reaching_the_target(self):
        deltas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        tprs = np.array([0.30, 0.80, 0.99, 1.00, 1.00])
        self.assertEqual(CC.sampled_limit(deltas, tprs, 0.95), 0.3)
        # no interpolation into the unmeasured gap between 0.2 and 0.3
        self.assertIn(CC.sampled_limit(deltas, tprs, 0.95), set(deltas.tolist()))

    def test_right_censoring_and_the_nominal_delta(self):
        deltas = np.array([0.0, 0.1, 0.2])
        self.assertIsNone(CC.sampled_limit(deltas, np.array([1.0, 0.2, 0.6]), 0.95))
        # a delta=0 entry is a nominal reference, never a detection limit
        self.assertEqual(CC.sampled_limit(deltas, np.array([1.0, 0.96, 1.0]), 0.95), 0.1)

    def test_bootstrap_limit_preserves_cross_delta_pairing(self):
        # 57 of 60 ICs detected at every delta, 3 never: resampling an IC must
        # move its whole curve together, so a replicate limit is 0.3 or censored
        n = 60
        always = np.zeros(n)
        always[:57] = 1.0
        indicators = np.vstack([0.1 * always, 0.6 * always, always])
        deltas = np.array([0.1, 0.2, 0.3])
        out = CC.bootstrap_limit(deltas, indicators, 0.95, seed=17)
        self.assertEqual(out["n_boot"], CC.N_BOOT)
        self.assertGreater(out["censored_fraction"], 0.0)
        self.assertLess(out["censored_fraction"], 1.0)
        for bound in out["limit_ci"]:
            self.assertIn(bound, (None, 0.3))
        again = CC.bootstrap_limit(deltas, indicators, 0.95, seed=17)
        self.assertEqual(out["limit_ci"], again["limit_ci"])
        self.assertEqual(out["censored_fraction"], again["censored_fraction"])

    def test_bootstrap_limit_of_a_saturated_curve(self):
        indicators = np.ones((3, 60))
        out = CC.bootstrap_limit(np.array([0.1, 0.2, 0.3]), indicators, 0.95, seed=1)
        self.assertEqual(out["censored_fraction"], 0.0)
        self.assertEqual(out["limit_ci"], (0.1, 0.1))

    def test_limits_print_at_the_resolution_of_the_alpha_grid(self):
        self.assertEqual(CC.limit_text(None, 0.5), ">0.5")
        self.assertEqual(CC.limit_text(0.2, 0.5), "0.2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
