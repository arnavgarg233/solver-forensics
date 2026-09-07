#!/usr/bin/env python3
"""Fast tests for the confirmatory structured-noise recovery experiment."""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import cross_conformal as CC
import heated_channel as HC
import reviewer_sensitivity as RS
import structured_noise_recovery as SR


class TestFrozenProtocol(unittest.TestCase):
    def test_confirmatory_protocol_is_predeclared(self):
        cfg = SR.protocol()
        self.assertEqual((cfg["pe"], cfg["grid_n"], cfg["library"]),
                         (100.0, 64, "base5"))
        self.assertEqual((cfg["n_ic"], cfg["n_acquisitions"], cfg["sigma"]),
                         (60, 5, 0.01))
        self.assertEqual(cfg["noise_models"],
                         ("white", "lowpass", "gradient", "multiplicative"))
        np.testing.assert_array_equal(cfg["alphas_weak"],
                                      np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
        np.testing.assert_array_equal(cfg["alphas_strong"],
                                      np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]))
        self.assertEqual(cfg["target_tpr"], 0.95)
        self.assertEqual(cfg["target_fpr"], CC.TARGET_FPR)
        self.assertEqual(tuple(SR.GO_CRITERIA),
                         ("white_positive_control", "structured_recovery", "overall_go"))
        self.assertEqual(SR.METHODS, ("decision_average", "field_average"))
        self.assertEqual(SR.STRUCTURED_MODELS,
                         ("lowpass", "gradient", "multiplicative"))


class TestAcquisitionAveraging(unittest.TestCase):
    def test_averaging_is_deterministic_and_precedes_derivatives(self):
        fields = np.arange(5 * 8 * 8, dtype=float).reshape(5, 8, 8)
        expected = np.mean(fields, axis=0)
        np.testing.assert_array_equal(SR.average_acquisitions(fields), expected)
        np.testing.assert_array_equal(SR.average_acquisitions(fields),
                                      SR.average_acquisitions(fields.copy()))
        with self.assertRaises(ValueError):
            SR.average_acquisitions(fields[0])
        with self.assertRaises(ValueError):
            SR.average_acquisitions(np.empty((0, 8, 8)))

    def test_same_seeded_acquisitions_feed_both_arms(self):
        clean = 1.0 + np.arange(64, dtype=float).reshape(8, 8) / 64.0
        first = SR.noisy_acquisitions(clean, "lowpass", 2, 7, sigma=0.01,
                                      n_acquisitions=5, grid_n=64)
        again = SR.noisy_acquisitions(clean, "lowpass", 2, 7, sigma=0.01,
                                      n_acquisitions=5, grid_n=64)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(first.shape, (5, 8, 8))
        self.assertTrue(any(not np.array_equal(first[0], first[j])
                            for j in range(1, 5)))

    def test_signature_construction_never_mixes_cases(self):
        cfg = SR.protocol()
        cfg.update({"grid_n": 8, "n_ic": 2, "n_acquisitions": 3,
                    "noise_models": ("white",),
                    "alphas": np.array([0.9, 1.0])})
        reference = np.stack([np.zeros((8, 8)), np.full((8, 8), 100.0)])
        working = np.empty((2, 2, 8, 8))
        working[0, 0], working[0, 1] = 10.0, 120.0
        working[1, 0], working[1, 1] = 30.0, 140.0

        original_acquisitions = SR.noisy_acquisitions
        original_signature = RS.signature
        SR.noisy_acquisitions = lambda clean, model, alpha_index, ic_index, **kw: np.stack(
            [clean + r for r in range(kw["n_acquisitions"])])
        RS.signature = lambda observed, ref, names, h=None: np.full(
            5, float(np.mean(observed - ref)))
        try:
            arms = SR.signature_arms(reference, working, cfg)
            changed = working.copy()
            changed[0, 0] += 1000.0
            changed_arms = SR.signature_arms(reference, changed, cfg)
        finally:
            SR.noisy_acquisitions = original_acquisitions
            RS.signature = original_signature

        baseline = arms["decision_average"]["white"]
        recovered = arms["field_average"]["white"]
        self.assertEqual(len(baseline[0.9]), 3)
        self.assertEqual(len(recovered[0.9]), 1)
        np.testing.assert_allclose([rep[0, 0] for rep in baseline[0.9]],
                                   [10.0, 11.0, 12.0])
        self.assertEqual(recovered[0.9][0][0, 0], 11.0)
        self.assertEqual(recovered[0.9][0][1, 0], 21.0)
        for method in SR.METHODS:
            for alpha in (0.9, 1.0):
                for replicate, before in enumerate(arms[method]["white"][alpha]):
                    after = changed_arms[method]["white"][alpha][replicate]
                    affected = method == "field_average" or replicate < 3
                    if alpha == 0.9 and affected:
                        self.assertNotEqual(before[0, 0], after[0, 0])
                        np.testing.assert_array_equal(before[1], after[1])
                    else:
                        np.testing.assert_array_equal(before, after)


class TestCacheContract(unittest.TestCase):
    def test_npz_round_trip_and_metadata_rejection(self):
        cfg = SR.protocol()
        cfg.update({"grid_n": 8, "n_ic": 2,
                    "alphas": np.array([0.9, 1.0, 1.1]),
                    "alphas_weak": np.array([1.0, 0.9]),
                    "alphas_strong": np.array([1.0, 1.1])})
        reference = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
        working = np.arange(3 * 2 * 8 * 8, dtype=float).reshape(3, 2, 8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fields.npz")
            SR.save_field_cache(path, reference, working, cfg)
            loaded_ref, loaded_work = SR.load_field_cache(path, cfg)
            np.testing.assert_array_equal(loaded_ref, reference)
            np.testing.assert_array_equal(loaded_work, working)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(SR.CACHE_KEYS))

            wrong = dict(cfg)
            wrong["sigma"] = 0.02
            with self.assertRaises(ValueError):
                SR.load_field_cache(path, wrong)
            missing = os.path.join(tmp, "missing-key.npz")
            np.savez(missing, schema_version=np.array(SR.CACHE_SCHEMA_VERSION))
            with self.assertRaises(ValueError):
                SR.load_field_cache(missing, cfg)

    def test_cache_is_optional_and_owned_by_this_script(self):
        parser = SR.build_parser()
        self.assertIsNone(parser.parse_args([]).cache)
        self.assertEqual(parser.parse_args(["--cache", "x.npz"]).cache, "x.npz")
        self.assertEqual(os.path.basename(SR.DEFAULT_CACHE_PATH),
                         "structured_noise_recovery_fields.npz")
        self.assertEqual(os.path.basename(SR.CSV_PATH),
                         "structured_noise_recovery.csv")


def _curve(method, noise, direction, limit):
    return {"method": method, "noise": noise, "direction": direction,
            "alphas": np.array([0.9]), "deltas": np.array([0.1]),
            "tpr": np.array([0.96]), "tpr_ci": np.array([[0.90, 1.0]]),
            "fpr": np.array([0.04]), "fpr_ci": np.array([[0.0, 0.08]]),
            "n_ic": 60, "n_noise": 5 if method == "decision_average" else 1,
            "limit": limit, "limit_ci": (limit, limit),
            "censored_fraction": 0.0 if limit is not None else 1.0}


def _synthetic_results(recovery_limit=0.3):
    rows = []
    for noise in ("white", "lowpass", "gradient", "multiplicative"):
        for direction in ("weaken", "strengthen"):
            baseline_limit = 0.4 if noise == "white" else None
            rows.append(_curve("decision_average", noise, direction, baseline_limit))
            rows.append(_curve("field_average", noise, direction, recovery_limit))
    return rows


class TestDecisionAndOutputContracts(unittest.TestCase):
    def test_go_rule_preserves_and_compares_the_failed_baseline(self):
        assessment = SR.assess_go(_synthetic_results())
        self.assertTrue(assessment["white_positive_control"]["pass"])
        self.assertTrue(assessment["structured_recovery"]["pass"])
        self.assertTrue(assessment["overall_go"]["pass"])
        failed = _synthetic_results()
        failed[-1]["limit"] = None
        assessment = SR.assess_go(failed)
        self.assertFalse(assessment["structured_recovery"]["pass"])
        self.assertFalse(assessment["overall_go"]["pass"])

    def test_output_schema_and_writer(self):
        cfg = SR.protocol()
        results = _synthetic_results()
        assessment = SR.assess_go(results)
        rows = SR.output_rows(results, assessment, cfg)
        self.assertEqual(set(row["type"] for row in rows),
                         {"curve", "limit", "criterion"})
        self.assertTrue(all(set(row) <= set(SR.CSV_FIELDS) for row in rows))
        self.assertTrue(all(row["control_role"] == "positive_control"
                            for row in rows if row.get("noise") == "white"))
        limits = [row for row in rows if row["type"] == "limit"]
        self.assertEqual(len(limits), 16)
        self.assertIn(">0.5", {row["detection_limit_delta_alpha"]
                               for row in limits})
        criteria = [row for row in rows if row["type"] == "criterion"]
        self.assertEqual([row["criterion"] for row in criteria],
                         list(SR.GO_CRITERIA))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            SR.write_results(rows, path)
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), SR.CSV_FIELDS)
                self.assertEqual(len(list(reader)), len(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
