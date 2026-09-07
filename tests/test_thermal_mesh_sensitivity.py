#!/usr/bin/env python3
"""Fast tests for the direct thermal solver-mesh sensitivity experiment."""
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
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import heated_channel as HC
import reviewer_sensitivity as RS
import cross_conformal as CC
import thermal_mesh_sensitivity as TMS


class TestFrozenMeshProtocol(unittest.TestCase):
    def test_mesh_plan_and_fixed_observer(self):
        plan = TMS.mesh_plan()
        self.assertGreaterEqual(len(plan), 3)
        self.assertEqual(len({mesh.mesh_id for mesh in plan}), len(plan))
        self.assertIn(TMS.ANCHOR_MESH_ID, {mesh.mesh_id for mesh in plan})
        self.assertEqual(next(mesh for mesh in plan
                              if mesh.mesh_id == TMS.ANCHOR_MESH_ID),
                         TMS.MeshSpec("anchor_60x20", 60, 20, 2026))
        self.assertEqual(TMS.REFERENCE_MESH,
                         TMS.MeshSpec("fixed_reference_180x60", 180, 60, 7001))
        self.assertTrue(all(mesh.nx == 3 * mesh.ny for mesh in plan))
        self.assertEqual(TMS.OBSERVATION_GRID_N, 64)
        self.assertEqual(TMS.LIBRARY_NAME, "base5")
        self.assertEqual(TMS.LIBRARY_TERMS, tuple(RS.LIBRARIES["base5"]))
        self.assertTrue(any(mesh.nx != TMS.OBSERVATION_GRID_N for mesh in plan))

    def test_population_alpha_noise_and_direction_contract(self):
        self.assertEqual((TMS.PE, TMS.N_IC, TMS.IC_SEEDS[0], TMS.IC_SEEDS[-1]),
                         (100.0, HC.N_IC, 1000, 1059))
        np.testing.assert_allclose(TMS.ALPHAS, np.arange(0.5, 1.51, 0.1), atol=1e-12)
        np.testing.assert_allclose(TMS.ALPHAS_WEAK, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        np.testing.assert_allclose(TMS.ALPHAS_STRONG, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        self.assertEqual([name for name, _ in TMS.directions()],
                         ["weaken", "strengthen"])
        self.assertEqual((TMS.NOISE_MODEL, TMS.SIGMA), ("white", 0.01))
        self.assertEqual((TMS.TARGET_TPR, TMS.TARGET_FPR), (0.95, CC.TARGET_FPR))
        counts = TMS.solve_counts()
        self.assertEqual(counts, {"reference": 60, "working": 1980, "total": 2040})

    def test_go_and_reporting_rules_are_frozen_in_protocol_hash(self):
        payload = TMS.protocol_payload()
        self.assertEqual(payload["go_criterion"], TMS.GO_CRITERION)
        self.assertEqual(payload["reporting_rule"], TMS.REPORTING_RULE)
        self.assertEqual(payload["max_allowed_degradation"], 0.1)
        self.assertIn("every mesh/direction", TMS.GO_CRITERION)
        self.assertIn("preserve censored and degraded", TMS.REPORTING_RULE)
        self.assertEqual(TMS.protocol_hash(), TMS.protocol_hash())
        self.assertEqual(len(TMS.protocol_hash()), 64)


class TestPairingAndNoLeakage(unittest.TestCase):
    def test_common_noise_is_paired_across_solver_meshes(self):
        seeds = {(alpha, ic): TMS.noise_seed(alpha, ic)
                 for alpha in range(len(TMS.ALPHAS)) for ic in range(TMS.N_IC)}
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertEqual(TMS.noise_seed(3, 17), TMS.noise_seed(3, 17))
        self.assertEqual(len(TMS.noise_seed.__code__.co_varnames[:2]), 2)

    def test_shared_detector_plan_has_no_train_calibration_test_leakage(self):
        plan = CC.fold_plan(TMS.N_IC, repeats=2, seed=TMS.cell_seed(0, 0, 0))
        all_ids = set(range(TMS.N_IC))
        for repeat in plan:
            seen_test = []
            for fold in repeat:
                train, cal, test = map(set, (fold["train"], fold["cal"], fold["test"]))
                self.assertFalse(train & cal)
                self.assertFalse(train & test)
                self.assertFalse(cal & test)
                self.assertEqual(train | cal | test, all_ids)
                seen_test.extend(test)
            self.assertEqual(sorted(seen_test), list(range(TMS.N_IC)))

    def test_paired_detection_preserves_ic_rows_and_rejects_mismatch(self):
        nominal = np.arange(TMS.N_IC * 5, dtype=float).reshape(TMS.N_IC, 5)
        changed = nominal + 0.25
        sentinel = {"tpr": 0.0}
        with mock.patch.object(TMS.CC, "split_conformal_detection",
                               return_value=sentinel) as detector:
            self.assertIs(TMS.paired_detection(nominal, changed, 77), sentinel)
        args, kwargs = detector.call_args
        np.testing.assert_array_equal(args[0], nominal)
        np.testing.assert_array_equal(args[1], changed)
        self.assertEqual(kwargs, {"seed": 77})
        with self.assertRaises(ValueError):
            TMS.paired_detection(nominal[:-1], changed, 1)
        with self.assertRaises(ValueError):
            TMS.paired_detection(nominal, changed[:, :-1], 1)


class TestCleanFieldCache(unittest.TestCase):
    def test_metadata_distinguishes_working_mesh_reference_and_observer(self):
        metadata = TMS.cache_metadata()
        self.assertEqual(metadata["cache_schema"], TMS.CACHE_SCHEMA)
        self.assertEqual(metadata["protocol_hash"], TMS.protocol_hash())
        self.assertEqual(metadata["working_meshes"],
                         [TMS._mesh_dict(mesh) for mesh in TMS.MESH_PLAN])
        self.assertEqual(metadata["reference_mesh"], TMS._mesh_dict(TMS.REFERENCE_MESH))
        self.assertEqual(metadata["observation_grid_n"], 64)
        self.assertNotIn("observation_grid_n", metadata["reference_mesh"])
        self.assertEqual(metadata["ic_seeds"], list(range(1000, 1060)))

    def test_small_round_trip_and_metadata_mismatch_rejection(self):
        tiny_meshes = (TMS.MeshSpec("tiny", 3, 2, 9),)
        patches = (
            mock.patch.object(TMS, "N_IC", 2),
            mock.patch.object(TMS, "IC_SEEDS", (1000, 1001)),
            mock.patch.object(TMS, "OBSERVATION_GRID_N", 4),
            mock.patch.object(TMS, "MESH_PLAN", tiny_meshes),
            mock.patch.object(TMS, "ANCHOR_MESH_ID", "tiny"),
            mock.patch.object(TMS, "ALPHAS", np.array([0.9, 1.0])),
            mock.patch.object(TMS, "ALPHAS_WEAK", np.array([1.0, 0.9])),
            mock.patch.object(TMS, "ALPHAS_STRONG", np.array([1.0])),
            mock.patch.object(TMS, "DIRECTIONS", (("weaken", np.array([1.0, 0.9])),
                                                   ("strengthen", np.array([1.0])))),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                patches[5], patches[6], patches[7], patches[8]:
            fields = {"reference_grids": np.arange(32, dtype=float).reshape(2, 4, 4),
                      "working_grids": np.arange(64, dtype=float).reshape(1, 2, 2, 4, 4)}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "owned.npz")
                self.assertEqual(TMS.save_clean_cache(path, fields), path)
                loaded = TMS.load_clean_cache(path)
                np.testing.assert_array_equal(loaded["reference_grids"],
                                              fields["reference_grids"])
                np.testing.assert_array_equal(loaded["working_grids"],
                                              fields["working_grids"])
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                    reference = np.array(archive["reference_grids"])
                    working = np.array(archive["working_grids"])
                metadata["sigma"] = 0.05
                np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata)),
                                    reference_grids=reference, working_grids=working)
                with self.assertRaisesRegex(ValueError, "frozen protocol"):
                    TMS.load_clean_cache(path)


class TestReporting(unittest.TestCase):
    @staticmethod
    def _raw_results():
        limits = {
            ("coarse_45x15", "weaken"): 0.4,
            ("anchor_60x20", "weaken"): 0.2,
            ("refined_90x30", "weaken"): 0.1,
            ("coarse_45x15", "strengthen"): 0.3,
            ("anchor_60x20", "strengthen"): 0.2,
            ("refined_90x30", "strengthen"): None,
        }
        results = []
        for mesh in TMS.MESH_PLAN:
            for direction, alphas in TMS.DIRECTIONS:
                sampled = alphas[1:]
                k = len(sampled)
                limit = limits[(mesh.mesh_id, direction)]
                results.append({
                    "mesh": mesh, "direction": direction, "alphas": sampled,
                    "deltas": np.abs(sampled - 1.0), "tpr": np.linspace(0.2, 1.0, k),
                    "tpr_ci": np.column_stack((np.linspace(0.1, 0.9, k),
                                                np.linspace(0.3, 1.0, k))),
                    "fpr": np.full(k, 0.04),
                    "fpr_ci": np.column_stack((np.full(k, 0.01), np.full(k, 0.08))),
                    "limit": limit, "limit_ci": ((None, None) if limit is None
                                                   else (limit, limit)),
                    "censored_fraction": 1.0 if limit is None else 0.0,
                    "n_ic": TMS.N_IC,
                })
        return results

    def test_degradations_and_censoring_cannot_be_averaged_away(self):
        classified = TMS.classify_limits(self._raw_results())
        statuses = {(result["mesh"].mesh_id, result["direction"]):
                    result["mesh_status"] for result in classified}
        self.assertEqual(statuses[("coarse_45x15", "weaken")], "degraded")
        self.assertEqual(statuses[("refined_90x30", "weaken")], "improved")
        self.assertEqual(statuses[("refined_90x30", "strengthen")], "censored")
        self.assertFalse(TMS.verdict(classified))
        degraded = next(result for result in classified
                        if result["mesh_status"] == "degraded")
        self.assertAlmostEqual(degraded["limit_shift"], 0.2)
        self.assertAlmostEqual(degraded["degradation"], 0.2)

    def test_csv_schema_and_rows(self):
        classified = TMS.classify_limits(self._raw_results())
        rows = TMS.csv_rows(classified)
        required = {
            "mesh_id", "solver_nx", "solver_ny", "solver_seed",
            "observation_grid_n", "reference_nx", "reference_ny", "reference_seed",
            "direction", "alpha", "delta_alpha", "n_ic", "noise_model", "sigma",
            "library", "tpr", "fpr_measured", "fpr_target",
            "detection_limit_delta_alpha", "limit_shift_from_anchor",
            "degradation_delta_alpha", "mesh_status", "go_cell", "protocol_hash",
        }
        self.assertTrue(required <= set(TMS.CSV_FIELDS))
        self.assertEqual({row["type"] for row in rows}, {"nominal", "curve", "limit"})
        nominal = [row for row in rows if row["type"] == "nominal"]
        self.assertTrue(all(row["delta_alpha"] == "0.0" for row in nominal))
        self.assertTrue(all("tpr" not in row and "fpr_measured" not in row
                            for row in nominal))
        curve = [row for row in rows if row["type"] == "curve"]
        self.assertTrue(all(float(row["delta_alpha"]) > 0.0 for row in curve))
        self.assertTrue(all(int(row["observation_grid_n"]) == 64 for row in curve))
        limits = [row for row in rows if row["type"] == "limit"]
        self.assertIn("degraded", {row["mesh_status"] for row in limits})
        self.assertIn("censored", {row["mesh_status"] for row in limits})
        with tempfile.TemporaryDirectory() as tmp:
            path = TMS.write_csv(rows, os.path.join(tmp, "mesh.csv"))
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), TMS.CSV_FIELDS)
                self.assertEqual(len(list(reader)), len(rows))


class TestDriverContract(unittest.TestCase):
    def test_cli_cache_is_optional_and_dry_run_only_controls_csv(self):
        parser = TMS.build_parser()
        plain = parser.parse_args([])
        self.assertIsNone(plain.cache)
        self.assertFalse(plain.dry_run)
        self.assertEqual(parser.parse_args(["--cache"]).cache, TMS.CACHE_PATH)
        custom = parser.parse_args(["--cache", "/tmp/fields.npz", "--dry-run"])
        self.assertEqual(custom.cache, "/tmp/fields.npz")
        self.assertTrue(custom.dry_run)


if __name__ == "__main__":
    unittest.main()
