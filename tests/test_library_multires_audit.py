import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))

import library_multires_audit as audit


def _synthetic_solver(solver_key, n, ic, config):
    """Exact field plus a controlled solver-order error."""
    x = audit.cell_centers(n, config.length)
    truth = ic.values(x, config.t_final, config)
    order = 1.0 if solver_key == "euler" else 2.0
    amplitude = 0.2 + 0.02 * abs(float(ic.amplitudes[0]))
    return truth * (1.0 + amplitude * (config.length / n) ** order)


def _indistinguishable_solver(solver_key, n, ic, config):
    x = audit.cell_centers(n, config.length)
    truth = ic.values(x, config.t_final, config)
    return truth * (1.0 + 0.2 * (config.length / n) ** 1.5)


class ProtocolTests(unittest.TestCase):
    def test_full_configuration_and_criteria_are_frozen(self):
        config = audit.AuditConfig()
        self.assertEqual(config.n_ics, 60)
        self.assertEqual(config.grid_sizes, (32, 48, 72, 108))
        self.assertEqual((config.t_final, config.cfl), (0.10, 0.40))
        self.assertEqual(
            [round(config.t_final / (config.cfl * config.length / n)) for n in config.grid_sizes],
            [8, 12, 18, 27],
        )
        self.assertIsInstance(audit.GO_CRITERIA, tuple)
        self.assertEqual(audit.REQUIRED_PY_PDE_VERSION, "0.56.0")
        self.assertIs(audit.SOLVERS["euler"], audit.EulerSolver)
        self.assertIs(audit.SOLVERS["runge_kutta"], audit.RungeKuttaSolver)
        self.assertEqual(len(audit.criteria_sha256()), 64)
        config.validate()

    def test_rejects_an_unstable_fixed_cfl_design(self):
        config = audit.AuditConfig(
            n_ics=3, grid_sizes=(32, 64, 128), cfl=0.9, max_mode=2
        )
        with self.assertRaisesRegex(ValueError, "cannot isolate integrator order"):
            config.validate()

    def test_rejects_a_partial_step_that_breaks_the_exact_dt_schedule(self):
        config = audit.AuditConfig(n_ics=3, t_final=0.11)
        with self.assertRaisesRegex(ValueError, "partial final step"):
            config.validate()

    def test_exact_fourier_solution_has_documented_decay_and_translation(self):
        config = audit.AuditConfig(
            n_ics=3, grid_sizes=(8, 12, 16), max_mode=1, speed=0.7,
            diffusivity=0.02, t_final=0.03, cfl=0.01,
        )
        ic = audit.FourierIC(
            modes=np.array([1.0]), amplitudes=np.array([2.0]), phases=np.array([0.3])
        )
        x = audit.cell_centers(12, config.length)
        k = 2.0 * np.pi / config.length
        expected = 1.0 + 0.5 * np.exp(-config.diffusivity * k**2 * config.t_final) * np.sin(
            k * (x - config.speed * config.t_final) + 0.3
        )
        np.testing.assert_allclose(ic.values(x, config.t_final, config), expected)

    def test_requires_the_exact_external_library_version(self):
        with mock.patch.object(audit, "installed_py_pde_version", return_value="0.56.1"):
            with self.assertRaisesRegex(RuntimeError, "requires py-pde==0.56.0"):
                audit.require_pinned_version()


class FeatureTests(unittest.TestCase):
    def test_rates_remove_single_grid_error_magnitude(self):
        grids = (16, 32, 64, 128)
        h = 1.0 / np.asarray(grids)
        amplitudes = np.array([0.1, 2.0, 30.0])[:, None]
        errors = amplitudes * h[None, :] ** 1.75
        rates = audit.convergence_rate_features(errors, grids)
        np.testing.assert_allclose(rates, 1.75, atol=1e-12)
        np.testing.assert_allclose(
            rates,
            audit.convergence_rate_features(errors * np.array([4.0, 0.2, 9.0])[:, None], grids),
            atol=1e-12,
        )

    def test_grouped_loo_separates_orders_and_holds_out_paired_ics(self):
        euler = np.tile([1.0, 1.05, 0.95], (6, 1))
        rk = np.tile([2.0, 1.95, 2.05], (6, 1))
        self.assertEqual(audit.grouped_loo_accuracy(euler, rk), 1.0)
        self.assertEqual(audit.grouped_loo_accuracy(euler, euler.copy()), 0.5)


class AuditExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = audit.AuditConfig(
            n_ics=6,
            grid_sizes=(12, 18, 27),
            t_final=0.01,
            cfl=0.01,
            max_mode=2,
            seed=4,
        )

    def test_tiny_mock_audit_uses_the_same_controlled_ladder_and_goes(self):
        calls = []

        def recording_solver(solver_key, n, ic, config):
            calls.append((solver_key, n, config.cfl * config.length / n))
            return _synthetic_solver(solver_key, n, ic, config)

        report = audit.run_audit(self.config, solve_fn=recording_solver)
        self.assertEqual(report["py_pde_version"], "0.56.0")
        self.assertEqual(report["dt_policy"], "dt=cfl*dx")
        self.assertEqual(report["verdict"], "GO")
        self.assertTrue(all(item["passed"] for item in report["gates"]))
        self.assertAlmostEqual(report["metrics"]["euler_median_rate"], 1.0, places=8)
        self.assertAlmostEqual(report["metrics"]["rk_median_rate"], 2.0, places=6)

        for solver_key in audit.SOLVERS:
            solver_calls = [(n, dt) for key, n, dt in calls if key == solver_key]
            self.assertEqual(len(solver_calls), self.config.n_ics * len(self.config.grid_sizes))
            for n in self.config.grid_sizes:
                matching = [dt for called_n, dt in solver_calls if called_n == n]
                self.assertEqual(len(matching), self.config.n_ics)
                np.testing.assert_allclose(matching, self.config.cfl * self.config.length / n)

    def test_failed_separation_is_preserved_in_report_and_csv(self):
        report = audit.run_audit(self.config, solve_fn=_indistinguishable_solver)
        self.assertEqual(report["verdict"], "FAIL")
        failed_names = {item["gate"].name for item in report["gates"] if not item["passed"]}
        self.assertIn("paired grouped-LOO accuracy", failed_names)
        self.assertIn("median paired rate gap", failed_names)

        with tempfile.TemporaryDirectory() as directory:
            path = audit.write_csv(report, Path(directory) / "audit.csv")
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(audit.GO_CRITERIA))
        self.assertTrue(all(row["py_pde_version"] == "0.56.0" for row in rows))
        self.assertTrue(all(row["required_py_pde_version"] == "0.56.0" for row in rows))
        self.assertTrue(all(row["verdict"] == "FAIL" for row in rows))
        self.assertTrue(all(row["dt_policy"] == "dt=cfl*dx" for row in rows))

    def test_numerical_failure_details_are_not_discarded(self):
        def failing_solver(solver_key, n, ic, config):
            if solver_key == "euler" and n == 18:
                raise FloatingPointError("synthetic instability")
            return _synthetic_solver(solver_key, n, ic, config)

        report = audit.run_audit(self.config, solve_fn=failing_solver)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(len(report["failures"]), self.config.n_ics)
        self.assertTrue(all("synthetic instability" in failure for failure in report["failures"]))
        with tempfile.TemporaryDirectory() as directory:
            path = audit.write_csv(report, Path(directory) / "failed.csv")
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(all("synthetic instability" in row["numerical_failures"] for row in rows))

    def test_real_py_pde_solver_smoke_is_tiny(self):
        config = audit.AuditConfig(
            n_ics=3, grid_sizes=(8, 12, 16), t_final=0.001,
            cfl=0.005, max_mode=1, seed=8,
        )
        ic = audit.generate_initial_conditions(config)[0]
        for solver_key in audit.SOLVERS:
            field = audit.solve_one(solver_key, 8, ic, config)
            self.assertEqual(field.shape, (8,))
            self.assertTrue(np.all(np.isfinite(field)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
