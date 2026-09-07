#!/usr/bin/env python3
"""Fast protocol tests for the battery coupling domain map; no FEM sweep."""
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "battery"))

import battery_coupling_sensitivity as S


class TestCouplingParameter(unittest.TestCase):
    def test_ladder_is_frozen_ordered_and_contains_the_unchanged_baseline(self):
        ladder = S.coupling_ladder()
        values = np.array([spec.conductivity for spec in ladder])
        self.assertEqual(tuple(values), (5.0, 10.0, 20.0, 40.0, 80.0))
        self.assertTrue(np.all(np.diff(values) > 0.0))
        self.assertEqual(values.tolist().count(S.BASELINE_CONDUCTIVITY), 1)
        self.assertEqual(S.TARGET_ALPHA, 0.5)
        self.assertEqual(S.SIGMA, 0.01)
        self.assertEqual(S.solve_counts(), {"reference": 240, "working": 720, "total": 960})

    def test_parameter_is_restored_even_when_the_body_raises(self):
        old_kb = S.B.KB
        old_region = S.B.K_BY_REGION["cell"]
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with S.cell_conductivity(7.5):
                self.assertEqual(S.B.KB, 7.5)
                self.assertEqual(S.B.K_BY_REGION["cell"], 7.5)
                raise RuntimeError("deliberate")
        self.assertEqual(S.B.KB, old_kb)
        self.assertEqual(S.B.K_BY_REGION["cell"], old_region)

    def test_ladder_reaches_the_cell_side_of_a_conforming_interface(self):
        mesh = S.B.make_module_mesh(
            n_cell_x=4, n_cell_y=4, n_cool=2, n_plate=2, seed=3
        )
        old_kb = S.B.KB
        old_region = S.B.K_BY_REGION["cell"]
        receipt = S.verify_conjugate_transmission(mesh[0], mesh[1])
        expected = np.array(
            [spec.conductivity / S.B.RHOCP_F for spec in S.COUPLING_LADDER]
        )
        np.testing.assert_allclose(receipt["cell_kappa"], expected)
        self.assertGreater(receipt["interface_edges"], 0)
        self.assertEqual(S.B.KB, old_kb)
        self.assertEqual(S.B.K_BY_REGION["cell"], old_region)


class TestPairingAndIsolation(unittest.TestCase):
    def test_exact_six_fold_nineteen_calibration_plan_has_no_leakage(self):
        plan = S.pairing_plan(repeats=2, seed=17)
        self.assertEqual(len(plan), 2)
        for repeat in plan:
            self.assertEqual(len(repeat), 6)
            covered = []
            for fold in repeat:
                train, cal, test = fold["train"], fold["cal"], fold["test"]
                self.assertEqual((train.size, cal.size, test.size), (21, 19, 8))
                self.assertEqual(np.intersect1d(train, cal).size, 0)
                self.assertEqual(np.intersect1d(train, test).size, 0)
                self.assertEqual(np.intersect1d(cal, test).size, 0)
                np.testing.assert_array_equal(
                    np.union1d(np.union1d(train, cal), test), np.arange(S.N_IC)
                )
                covered.append(test)
            np.testing.assert_array_equal(np.sort(np.concatenate(covered)),
                                          np.arange(S.N_IC))

    def test_detection_wrapper_preserves_pair_rows_and_freezes_split_arguments(self):
        nominal = np.column_stack([np.arange(S.N_IC), np.zeros(S.N_IC)])
        changed = np.column_stack([np.arange(S.N_IC), np.ones(S.N_IC)])
        sentinel = {"tpr": 0.25}
        with mock.patch.object(S.CC, "split_conformal_detection", return_value=sentinel) as call:
            self.assertIs(S.paired_detection(nominal, changed, seed=9), sentinel)
        sent_nominal, sent_changed = call.call_args.args[:2]
        np.testing.assert_array_equal(sent_nominal[:, 0], sent_changed[:, 0])
        self.assertEqual(call.call_args.kwargs["folds"], 6)
        self.assertEqual(call.call_args.kwargs["n_cal"], 19)
        self.assertEqual(call.call_args.kwargs["cal_rank"], 19)
        self.assertEqual(call.call_args.kwargs["seed"], 9)
        with self.assertRaises(ValueError):
            S.paired_detection(nominal, changed[:-1], seed=9)


class TestOwnedCache(unittest.TestCase):
    @staticmethod
    def _small_shapes():
        return {
            "reference_grids": (2, 2, 2),
            "working_grids": (2, 2, 2),
            "energy_errors": (2, 2),
            "linear_residuals": (2, 2),
            "cell_kappa": (2,),
        }

    def _small_fields(self):
        fields = {name: np.zeros(shape) for name, shape in self._small_shapes().items()}
        fields["cell_kappa"] = np.array([1.0, 2.0])
        return fields

    def test_metadata_owns_cache_and_records_matched_references(self):
        metadata = S.cache_metadata()
        self.assertEqual(metadata["cache_schema"], S.CACHE_SCHEMA)
        self.assertEqual(metadata["owner"], "battery_coupling_sensitivity.py")
        self.assertEqual(metadata["protocol_hash"], S.protocol_hash())
        self.assertTrue(metadata["matched_nominal_reference_per_ladder_point"])
        self.assertEqual((metadata["n_ic"], metadata["folds"], metadata["n_cal"]),
                         (48, 6, 19))
        self.assertEqual(metadata["target_alpha"], 0.5)
        self.assertEqual(metadata["frozen_baseline_sensitivity"], 0.083333)
        self.assertEqual(metadata["frozen_baseline_decision"], "NO-FAULT")

    def test_cache_round_trip_and_protocol_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "owned.npz")
            with mock.patch.object(S, "_expected_shapes", self._small_shapes):
                expected = S.save_clean_cache(path, self._small_fields())
                self.assertEqual(expected, path)
                loaded = S.load_clean_cache(path)
                for key, value in self._small_fields().items():
                    np.testing.assert_array_equal(loaded[key], value)

                with np.load(path, allow_pickle=False) as archive:
                    payload = {name: np.array(archive[name], copy=True)
                               for name in archive.files}
                metadata = json.loads(str(payload["metadata_json"].item()))
                metadata["target_alpha"] = 0.6
                payload["metadata_json"] = np.array(json.dumps(metadata))
                np.savez_compressed(path, **payload)
                with self.assertRaisesRegex(ValueError, "frozen protocol"):
                    S.load_clean_cache(path)


class TestCsvSchema(unittest.TestCase):
    @staticmethod
    def _detection(tpr, fpr=0.02):
        return {
            "tpr": float(tpr), "tpr_ci": (max(0.0, tpr - 0.05), min(1.0, tpr + 0.05)),
            "fpr": float(fpr), "fpr_ci": (0.0, min(1.0, fpr + 0.05)),
            "detect": np.full(S.N_IC, float(tpr)),
        }

    def _results(self):
        rows = []
        endpoint = (0.10, 0.30, 0.50, 0.70, 0.90)
        for spec, tpr in zip(S.COUPLING_LADDER, endpoint):
            rows.extend([
                {"spec": spec, "arm": "target", "scheme": "supg", "alpha": 0.5,
                 "detection": self._detection(tpr)},
                {"spec": spec, "arm": S.NEGATIVE_ARM, "scheme": "supg", "alpha": 1.0,
                 "detection": self._detection(0.05)},
                {"spec": spec, "arm": "galerkin_positive", "scheme": "galerkin", "alpha": 0.0,
                 "detection": self._detection(0.95)},
            ])
        return rows

    def test_rows_and_written_header_use_one_complete_schema(self):
        health = {"max_energy_error": 0.0, "max_linear_residual": 1e-12,
                  "transmission_ok": True, "pass": True}
        results = self._results()
        assessment = S.assess(results, health)
        self.assertTrue(assessment["systematic_dependence"])
        rows = S.csv_rows(results, assessment, health)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(tuple(row) == S.CSV_FIELDS for row in rows))
        self.assertEqual(rows[-1]["row_type"], "summary")
        self.assertEqual(rows[-1]["frozen_baseline_decision"], "NO-FAULT")

        with tempfile.TemporaryDirectory() as directory:
            path = S.write_csv(os.path.join(directory, "out.csv"), rows)
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                written = list(reader)
            self.assertEqual(tuple(reader.fieldnames), S.CSV_FIELDS)
            self.assertEqual(len(written), len(rows))
            self.assertEqual(set(written[0]), set(S.CSV_FIELDS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
