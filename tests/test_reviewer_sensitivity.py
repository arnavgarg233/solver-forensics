#!/usr/bin/env python3
"""Unit tests for src/thermal/reviewer_sensitivity.py.

No FEM solve: every test operates on analytic grids, on the scattered node set
of a mesh (mesh construction only), or on synthetic signature clouds, so the
file runs in seconds.  It covers what this sweep adds on top of the validated
heated-channel solver, and nothing that merely mirrors the implementation:

  * derivative accuracy on the normalized grid, including bit-level agreement
    with the verified anchor library at n=64;
  * library membership and design-matrix contract;
  * zero-centred matched-RMS structured noise, its structure, its
    grid-independent low-pass correlation length and its determinism;
  * the three requested slices, and the shared split-conformal detector with
    nested noise replicates (measured level, determinism, sampled-grid limits
    with right-censoring, and replicates that never become independent units);
  * the output contract: paths, --quick/--dry-run switches, CSV fields, and a
    figure that renders both directions and passes its own layout guard.

Run:  /tmp/solver-forensics-py310/bin/python -m unittest discover -s tests -v
"""
import inspect
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import heated_channel as HC              # validated thermal foundation
import supg_2d_engineering as A          # verified anchor (grid, FD library)
import cross_conformal as CC             # shared detection helper
import reviewer_sensitivity as RS        # the module under test


def _grid_from(n, fn):
    """Evaluate ``fn(xi, eta)`` on the normalized n x n observation grid."""
    xi, eta, h = RS.normalized_axes(n)
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")
    return fn(XI, ETA), XI, ETA, h


def _clouds(n_ic=48, p=5, shift=0.0, seed=0, replicates=2):
    """Noise-replicate lists of paired unit-direction signature clouds."""
    rng = np.random.default_rng(seed)
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    nominal, changed = [], []
    for _ in range(replicates):
        nominal.append(unit(0.02 * rng.standard_normal((n_ic, p)) + np.eye(p)[0]))
        changed.append(unit(0.02 * rng.standard_normal((n_ic, p)) + np.eye(p)[0]
                            + shift * np.eye(p)[1]))
    return nominal, changed


class TestNormalizedGridAndLibrary(unittest.TestCase):
    def test_axes_and_anchor_spacing(self):
        for n in RS.GRIDS:
            xi, eta, h = RS.normalized_axes(n)
            np.testing.assert_allclose([xi[0], xi[-1], eta[0], eta[-1]],
                                       [0.0, 1.0, 0.0, 1.0])
            self.assertAlmostEqual(h, 1.0 / (n - 1), places=15)
        self.assertEqual(RS.normalized_axes(A.GRID_OBS)[2], A._H)
        with self.assertRaises(ValueError):
            RS.normalized_axes(6)

    def test_interpolation_reproduces_the_validated_64x64_grid(self):
        pts, _, _ = HC.make_channel_mesh(24, 8, seed=5)
        vals = np.sin(1.3 * pts[:, 0]) * np.cos(2.1 * pts[:, 1]) - 0.5 * pts[:, 1]
        np.testing.assert_allclose(
            RS.to_normalized_grid(pts, vals, A.GRID_OBS),
            HC.to_channel_grid(pts, vals), rtol=1e-12, atol=1e-12)
        ramp = RS.to_normalized_grid(pts, pts[:, 0] / HC.LX, 32)
        self.assertTrue(np.all(np.diff(ramp, axis=0) > 0.0))     # axis 0 is x
        np.testing.assert_allclose(np.diff(ramp, axis=1), 0.0, atol=1e-7)

    def test_exact_on_quadratic_and_cubic_fields(self):
        # central differences are exact for low-degree polynomials; every third
        # derivative of a quadratic must vanish identically
        U, XI, ETA, _ = _grid_from(
            40, lambda xi, eta: 1.0 + 2.0 * xi - 3.0 * eta + 4.0 * xi ** 2
            + 0.5 * eta ** 2 - 2.5 * xi * eta)
        Dlib, sl = RS.fd_library(U)
        xi, eta = XI[sl, sl], ETA[sl, sl]
        for name, target in {"u_x": 2.0 + 8.0 * xi - 2.5 * eta,
                             "u_y": -3.0 + eta - 2.5 * xi,
                             "u_xx": np.full_like(xi, 8.0),
                             "u_yy": np.full_like(xi, 1.0),
                             "u_xy": np.full_like(xi, -2.5),
                             "u_xxx": np.zeros_like(xi),
                             "u_yyy": np.zeros_like(xi),
                             "u_xxy": np.zeros_like(xi),
                             "u_xyy": np.zeros_like(xi)}.items():
            np.testing.assert_allclose(Dlib[name], target, rtol=1e-9, atol=1e-8,
                                       err_msg=name)
        U, XI, _, _ = _grid_from(
            40, lambda xi, eta: 2.0 * xi ** 3 - eta ** 3 + 3.0 * xi ** 2 * eta ** 2)
        Dlib, sl = RS.fd_library(U)
        np.testing.assert_allclose(Dlib["u_xxx"], 12.0, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(Dlib["u_yyy"], -6.0, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(Dlib["u_xyy"], 12.0 * XI[sl, sl], rtol=1e-6,
                                   atol=1e-6)

    def test_second_order_convergence(self):
        fn = lambda xi, eta: np.sin(2.0 * np.pi * xi) * np.cos(3.0 * np.pi * eta)
        errors = {}
        for n in (65, 129):
            U, XI, ETA, _ = _grid_from(n, fn)
            Dlib, sl = RS.fd_library(U)
            xi, eta = XI[sl, sl], ETA[sl, sl]
            sx, cx = np.sin(2.0 * np.pi * xi), np.cos(2.0 * np.pi * xi)
            sy, cy = np.sin(3.0 * np.pi * eta), np.cos(3.0 * np.pi * eta)
            errors[n] = {k: float(np.max(np.abs(Dlib[k] - v))) for k, v in
                         {"u_x": 2.0 * np.pi * cx * cy,
                          "u_xx": -(2.0 * np.pi) ** 2 * sx * cy,
                          "u_xy": -6.0 * np.pi ** 2 * cx * sy,
                          "u_xxx": -(2.0 * np.pi) ** 3 * cx * cy,
                          "u_xxy": (2.0 * np.pi) ** 2 * 3.0 * np.pi * sx * sy}.items()}
        for name in errors[65]:
            ratio = errors[65][name] / max(errors[129][name], 1e-300)
            self.assertGreater(ratio, 3.2, f"{name} not second order: {ratio:.2f}")

    def test_no_periodic_wraparound_inside_the_trim(self):
        # np.roll is periodic, so compare the trimmed window with explicit
        # non-periodic slicing on a field with huge jumps at the domain edges
        n = 36
        U = np.random.default_rng(11).standard_normal((n, n))
        U[0, :] += 1e4
        U[-1, :] -= 1e4
        Dlib, sl = RS.fd_library(U, ("u_xx", "u_xxx"))
        _, _, h = RS.normalized_axes(n)
        np.testing.assert_allclose(
            Dlib["u_xx"],
            ((U[3:-1, :] - 2.0 * U[2:-2, :] + U[1:-3, :]) / h ** 2)[:, sl], rtol=1e-12)
        np.testing.assert_allclose(
            Dlib["u_xxx"],
            ((U[4:, :] - 2.0 * U[3:-1, :] + 2.0 * U[1:-3, :] - U[:-4, :])
             / (2 * h ** 3))[:, sl], rtol=1e-12)

    def test_matches_the_anchor_library_and_signature_at_n64(self):
        Us, _, _, _ = _grid_from(A.GRID_OBS,
                                 lambda xi, eta: np.sin(3.0 * xi) * np.exp(-2.0 * eta))
        mine, sl_mine = RS.fd_library(Us, A.LIB)
        theirs, sl_theirs = A._fd_library(Us)
        self.assertEqual(sl_mine, sl_theirs)
        for name in A.LIB:
            np.testing.assert_allclose(mine[name], theirs[name], rtol=1e-12,
                                       atol=0.0, err_msg=name)
        Ur = Us + 0.002 * np.cos(6.0 * Us)
        np.testing.assert_allclose(RS.signature(Us, Ur, RS.LIBRARIES["base5"]),
                                   HC.sig_from_grid(Us, Ur), rtol=1e-9, atol=1e-12)

    def test_rejects_bad_grids_and_unknown_terms(self):
        with self.assertRaises(ValueError):
            RS.fd_library(np.zeros((16, 12)))
        with self.assertRaises(KeyError):
            RS.fd_library(np.zeros((32, 32)), ("u_zz",))
        with self.assertRaises(ValueError):
            RS.signature(np.zeros((32, 32)), np.zeros((16, 16)))


class TestLibraryMembership(unittest.TestCase):
    def test_names_sizes_and_nesting(self):
        sizes = {"second3": 3, "base5": 5, "first_plus_base7": 7,
                 "mixed_plus_base7": 7, "full9": 9}
        self.assertEqual(set(RS.LIBRARIES), set(sizes))
        for name, size in sizes.items():
            self.assertEqual(len(set(RS.LIBRARIES[name])), size, name)
            self.assertTrue(set(RS.LIBRARIES[name]) <= set(RS.DERIV_NAMES), name)
        self.assertEqual(RS.LIBRARIES["second3"], ("u_xx", "u_yy", "u_xy"))
        self.assertEqual(RS.LIBRARIES["base5"], tuple(A.LIB))
        self.assertEqual(set(RS.LIBRARIES["first_plus_base7"]),
                         set(A.LIB) | {"u_x", "u_y"})
        self.assertEqual(set(RS.LIBRARIES["mixed_plus_base7"]),
                         set(A.LIB) | {"u_xxy", "u_xyy"})
        # the five anchor terms keep their column indices in every superset, so
        # coefficient components stay comparable across libraries
        for name in ("base5", "first_plus_base7", "mixed_plus_base7", "full9"):
            self.assertEqual(RS.LIBRARIES[name][:5], tuple(A.LIB), name)

    def test_design_matrix_shape_order_and_memory(self):
        n = 32
        U, _, _, _ = _grid_from(n, lambda xi, eta: xi ** 2 * eta)
        Dlib, _ = RS.fd_library(U)
        for name, names in RS.LIBRARIES.items():
            Amat = RS.design_matrix(Dlib, names)
            self.assertEqual(Amat.shape, ((n - 4) ** 2, len(names)), name)
            for j, term in enumerate(names):
                np.testing.assert_allclose(Amat[:, j], Dlib[term].ravel(),
                                           rtol=1e-12, err_msg=f"{name}:{term}")
            self.assertEqual(RS.design_matrix_bytes(n, len(names)), Amat.nbytes)
            self.assertEqual(RS.signature(U, U.copy(), names).shape, (len(names),))


class TestStructuredNoise(unittest.TestCase):
    def setUp(self):
        # boundary-layer-like clean field: steep transverse gradient near eta=0
        self.clean, _, _, _ = _grid_from(
            64, lambda xi, eta: np.exp(-eta / 0.08) + 0.4 * xi + 0.2)

    def test_zero_centred_matched_rms_and_determinism(self):
        self.assertEqual(RS.NOISE_MODELS,
                         ("white", "lowpass", "gradient", "multiplicative"))
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
        for sigma in (0.01, 0.05):
            for k, model in enumerate(RS.NOISE_MODELS):
                eta = RS.structured_noise(self.clean, model, seed=100 + k, sigma=sigma)
                self.assertAlmostEqual(rms(eta) / (sigma * rms(self.clean)), 1.0,
                                       places=12, msg=f"{model} sigma={sigma}")
                self.assertAlmostEqual(float(np.mean(eta)) / (sigma * rms(self.clean)),
                                       0.0, places=12, msg=f"{model} not centred")
                np.testing.assert_array_equal(
                    eta, RS.structured_noise(self.clean, model, seed=100 + k,
                                             sigma=sigma))
                self.assertGreater(float(np.max(np.abs(
                    eta - RS.structured_noise(self.clean, model, seed=1, sigma=sigma)
                ))), 0.0, model)
                np.testing.assert_allclose(
                    RS.add_structured_noise(self.clean, model, seed=3, sigma=0.0),
                    self.clean, atol=0.0)
        with self.assertRaises(ValueError):
            RS.structured_noise(self.clean, "pink", seed=0, sigma=0.01)

    def test_lowpass_correlation_length_is_physical_across_grids(self):
        # the correlation length is a fraction of the domain side, so the
        # spectral centroid in cycles per domain must not move with n
        centroid = {}
        for n in (32, 96):
            field, _, _, _ = _grid_from(n, lambda xi, eta: 1.0 + 0.0 * xi)
            eta = RS.structured_noise(field, "lowpass", seed=5, sigma=0.05)
            power = np.abs(np.fft.rfft2(eta)) ** 2
            kx = np.abs(np.fft.fftfreq(n) * n)[:, None]
            ky = np.abs(np.fft.rfftfreq(n) * n)[None, :]
            wavenumber = np.sqrt(kx ** 2 + ky ** 2)
            centroid[n] = float(np.sum(power * wavenumber) / np.sum(power))
        self.assertAlmostEqual(centroid[96] / centroid[32], 1.0, delta=0.2)
        for n in (32, 96):                       # and inside the cutoff scale
            self.assertLess(centroid[n], 1.0 / RS.CORR_FRACTION)

    def test_noise_structure_differs_at_matched_energy(self):
        roughness = lambda v: float(np.mean(np.diff(v, n=2, axis=0) ** 2))
        white = RS.structured_noise(self.clean, "white", seed=21, sigma=0.05)
        smooth = RS.structured_noise(self.clean, "lowpass", seed=21, sigma=0.05)
        self.assertLess(roughness(smooth), 0.1 * roughness(white))
        layered = RS.structured_noise(self.clean, "gradient", seed=31, sigma=0.05)
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
        self.assertGreater(rms(layered[:, :6]), 5.0 * rms(layered[:, -6:]))
        self.assertLess(rms(white[:, :6]) / rms(white[:, -6:]), 2.0)
        field, _, _, _ = _grid_from(64, lambda xi, eta: 0.05 + 4.0 * xi)
        scaled = RS.structured_noise(field, "multiplicative", seed=41, sigma=0.05)
        self.assertGreater(rms(scaled[-8:, :]), 5.0 * rms(scaled[:8, :]))

    def test_seed_streams_are_unique_per_field_and_cell(self):
        seeds = {RS.noise_seed(n, m, a, i, r) for n in RS.GRIDS for m in range(4)
                 for a in range(11) for i in range(60) for r in range(5)}
        self.assertEqual(len(seeds), len(RS.GRIDS) * 4 * 11 * 60 * 5)
        cells = {RS.cell_seed(n, l, m, d) for n in RS.GRIDS for l in range(5)
                 for m in range(4) for d in range(2)}
        self.assertEqual(len(cells), len(RS.GRIDS) * 5 * 4 * 2)


class TestSharedDetector(unittest.TestCase):
    def test_uses_the_split_conformal_helper_not_the_old_estimators(self):
        import thermal_detection_limit as TDL
        self.assertEqual(RS.TARGET_FPR, CC.TARGET_FPR)
        self.assertEqual(RS.TARGET_SENSITIVITY, 0.95)
        self.assertFalse(hasattr(TDL, "supervised_sensitivity"))
        self.assertFalse(hasattr(TDL, "detection_limit"))
        self.assertFalse(hasattr(CC, "cross_conformal_detection"))
        self.assertEqual(TDL.CONVENTIONAL, ("thermal_pair", "Twall_max", "Nu_mean"))
        self.assertEqual(TDL.DETECTORS, ("signature", "thermal_pair", "Twall_max",
                                         "Nu_mean", "fullfield_T"))

    def test_curve_is_direction_resolved_and_skips_the_nominal_contrast(self):
        nominal, near = _clouds(shift=0.0, seed=1)
        _, far = _clouds(shift=0.6, seed=2)
        curve = RS.detection_curve({1.0: nominal, 0.9: near, 0.5: far},
                                   (1.0, 0.9, 0.5), seed=3)
        np.testing.assert_allclose(curve["deltas"], [0.1, 0.5], atol=1e-12)
        self.assertEqual((curve["n_ic"], curve["n_noise"]), (RS.QUICK_N_IC, 2))
        self.assertNotIn(0.0, curve["deltas"])        # no alpha=1 self-comparison
        self.assertLessEqual(curve["tpr"][0], 0.2)    # exchangeable pair
        self.assertGreaterEqual(curve["tpr"][1], 0.9)
        self.assertLessEqual(max(curve["fpr"]), 0.12)  # measured, not imposed
        self.assertEqual(curve["limit"], 0.5)
        self.assertAlmostEqual(curve["angles"][0], 0.0, delta=3.0)
        self.assertGreater(curve["angles"][1], 20.0)

    def test_noise_replicates_are_nested_not_extra_units(self):
        nominal, changed = _clouds(shift=0.05, seed=12, replicates=3)
        curve = RS.detection_curve({1.0: nominal, 0.5: changed}, (1.0, 0.5), seed=13)
        self.assertEqual((curve["n_ic"], curve["n_noise"]), (RS.QUICK_N_IC, 3))
        per_replicate = [CC.split_conformal_detection(nominal[r], changed[r],
                                                      seed=13 + r) for r in range(3)]
        # the reported rates are the within-IC averages of the per-replicate
        # readouts, and the interval resamples only the n_ic ICs
        self.assertAlmostEqual(curve["tpr"][0], float(np.mean(
            [cell["tpr"] for cell in per_replicate])), places=12)
        self.assertAlmostEqual(curve["fpr"][0], float(np.mean(
            [cell["fpr"] for cell in per_replicate])), places=12)
        np.testing.assert_allclose(curve["tpr_ci"][0], CC.cluster_bootstrap_ci(
            np.mean([cell["detect"] for cell in per_replicate], axis=0), seed=513),
            atol=1e-12)
        # and it is NOT the interval that pooling 3 x n_ic replicate rows as
        # independent units would give, which is the estimator being ruled out
        pooled = CC.cluster_bootstrap_ci(
            np.concatenate([cell["detect"] for cell in per_replicate]), seed=513)
        self.assertNotEqual(tuple(curve["tpr_ci"][0]), tuple(pooled))
        with self.assertRaises(ValueError):          # ragged replicate counts
            RS.detection_curve({1.0: _clouds(replicates=2)[0],
                                0.5: _clouds(replicates=3)[1]}, (1.0, 0.5), seed=1)

    def test_sampled_limit_is_censored_without_a_detected_sample(self):
        nominal, near = _clouds(shift=0.0, seed=4)
        curve = RS.detection_curve({1.0: nominal, 0.9: near}, (1.0, 0.9), seed=5)
        self.assertIsNone(curve["limit"])
        self.assertGreater(curve["censored_fraction"], 0.5)
        self.assertEqual(curve["limit_ci"], (None, None))

    def test_determinism_for_a_seed(self):
        clouds = {1.0: _clouds(seed=6)[0], 0.9: _clouds(shift=0.4, seed=7)[1]}
        first = RS.detection_curve(clouds, (1.0, 0.9), seed=8)
        again = RS.detection_curve(clouds, (1.0, 0.9), seed=8)
        np.testing.assert_array_equal(first["tpr"], again["tpr"])
        np.testing.assert_array_equal(first["fpr"], again["fpr"])
        self.assertEqual(first["limit_ci"], again["limit_ci"])

    def test_angular_separation_is_a_plain_geometric_diagnostic(self):
        nominal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(RS.mean_angular_separation(nominal, nominal), 0.0,
                               places=9)
        orth = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(RS.mean_angular_separation(nominal, orth), 90.0,
                               places=9)
        self.assertAlmostEqual(RS.mean_angular_separation(
            3.0 * np.array([[1.0, 0.0]]),
            7.0 * np.array([[0.5, np.sqrt(3.0) / 2.0]])), 60.0, places=9)
        with self.assertRaises(ValueError):
            RS.mean_angular_separation(np.zeros((3, 5)), np.zeros((4, 5)))


class TestProtocol(unittest.TestCase):
    def test_full_protocol_matches_the_thermal_experiment(self):
        import thermal_detection_limit as TDL
        cfg = RS.config(quick=False)
        self.assertEqual((cfg["pe"], cfg["n_ic"], cfg["sigma"]),
                         (100.0, HC.N_IC, RS.SIGMA))
        self.assertEqual(cfg["grids"], (32, 48, 64, 96))
        self.assertEqual(tuple(cfg["libraries"]), tuple(RS.LIBRARIES))
        self.assertEqual(cfg["noise_models"], RS.NOISE_MODELS)
        self.assertEqual((cfg["n_noise"], RS.NOISE_REPLICATES), (5, 5))
        np.testing.assert_allclose(cfg["alphas_weak"], TDL.ALPHAS_WEAK)
        np.testing.assert_allclose(cfg["alphas_strong"], TDL.ALPHAS_STRONG)
        np.testing.assert_allclose(cfg["alphas"], TDL.ALL_ALPHAS)
        self.assertEqual([name for name, _ in RS.directions(cfg)],
                         ["weaken", "strengthen"])

    def test_quick_protocol_is_a_subset_that_keeps_exact_calibration(self):
        quick, full = RS.config(quick=True), RS.config(quick=False)
        for cfg in (quick, full):                # the 21/19/8 and 29/19/12 splits
            folds = CC.choose_folds(cfg["n_ic"])
            self.assertGreaterEqual(
                cfg["n_ic"] - CC.N_CAL - int(np.ceil(cfg["n_ic"] / folds)),
                CC.MIN_TRAIN_IC, cfg["n_ic"])
        self.assertEqual(quick["n_noise"], full["n_noise"])
        self.assertLess(quick["n_ic"], full["n_ic"])
        self.assertTrue(set(quick["grids"]) <= set(full["grids"]))
        self.assertTrue(set(quick["libraries"]) <= set(full["libraries"]))
        self.assertTrue(set(quick["alphas"]) <= set(full["alphas"]))
        self.assertEqual(quick["noise_models"], full["noise_models"])

    def test_only_the_three_requested_slices_are_evaluated(self):
        cfg = RS.config(quick=False)
        cells = RS.slice_cells(cfg)
        keys = [cell for cell, _ in cells]
        self.assertEqual(len(keys), len(set(keys)))          # each cell once
        cartesian = len(cfg["grids"]) * len(cfg["libraries"]) * len(cfg["noise_models"])
        self.assertEqual(len(keys), 11)
        self.assertLess(len(keys), cartesian)
        for (n, library, noise), tags in cells:
            self.assertTrue(set(tags) <= set(RS.SLICES))
            if "grid" in tags:
                self.assertEqual((library, noise), (RS.ANCHOR_LIBRARY, RS.ANCHOR_NOISE))
            if "library" in tags:
                self.assertEqual((n, noise), (RS.ANCHOR_GRID, RS.ANCHOR_NOISE))
            if "noise" in tags:
                self.assertEqual((n, library), (RS.ANCHOR_GRID, RS.ANCHOR_LIBRARY))
        for name, expected in (("grid", cfg["grids"]), ("library", cfg["libraries"]),
                               ("noise", cfg["noise_models"])):
            covered = {cell[{"grid": 0, "library": 1, "noise": 2}[name]]
                       for cell, tags in cells if name in tags}
            self.assertEqual(covered, set(expected), name)


def _synthetic_results(cfg):
    """Result dicts covering all three slices, one censored cell per direction."""
    results = []
    for (n, library, noise), tags in RS.slice_cells(cfg):
        for direction, alphas in RS.directions(cfg):
            deltas = np.abs(alphas[1:] - 1.0)
            k = len(deltas)
            censored = noise == "gradient"
            tpr = np.linspace(0.1, 0.4 if censored else 1.0, k)
            results.append({
                "alphas": alphas[1:], "deltas": deltas, "tpr": tpr,
                "tpr_ci": np.column_stack([tpr - 0.05, tpr + 0.05]),
                "fpr": np.full(k, 0.048), "n_ic": cfg["n_ic"],
                "n_noise": cfg["n_noise"],
                "fpr_ci": np.column_stack([np.full(k, 0.01), np.full(k, 0.09)]),
                "angles": np.linspace(1.0, 20.0, k),
                "limit": None if censored else float(deltas[-1]),
                "limit_ci": (None, None) if censored else (float(deltas[-2]), None),
                "censored_fraction": 1.0 if censored else 0.1,
                "grid_n": n, "library": library, "n_terms": len(RS.LIBRARIES[library]),
                "noise": noise, "direction": direction, "slices": tags})
    return results


class TestOutputContract(unittest.TestCase):
    def test_paths_and_switches(self):
        self.assertEqual(RS.CSV_PATH, os.path.join(
            _ROOT, "results", "tables", "reviewer_sensitivity.csv"))
        self.assertEqual(RS.FIG_PATH, os.path.join(
            _ROOT, "figures", "fig_reviewer_sensitivity.png"))
        params = inspect.signature(RS.run).parameters
        self.assertIs(params["write"].default, True)
        self.assertIs(params["quick"].default, False)
        parser = RS.build_parser()
        self.assertFalse(parser.parse_args([]).quick)
        self.assertTrue(parser.parse_args(["--quick"]).quick)
        self.assertTrue(parser.parse_args(["--dry-run"]).dry_run)

    def test_csv_rows_figure_and_writer(self):
        import csv as _csv
        import tempfile
        for field in ("type", "slice", "grid_n", "library", "noise", "direction",
                      "alpha", "delta_alpha", "n_ic", "n_noise", "tpr", "tpr_lo",
                      "tpr_hi", "fpr_measured", "fpr_lo", "fpr_hi", "fpr_target",
                      "mean_angle_deg", "detection_limit_delta_alpha", "limit_lo",
                      "limit_hi", "limit_censored_fraction", "interp_ms",
                      "ref_interp_ms", "derivative_ms", "lstsq_ms",
                      "design_matrix_mb"):
            self.assertIn(field, RS.CSV_FIELDS, field)
        cfg = RS.config(quick=True)
        results = _synthetic_results(cfg)
        rows = RS._rows(results, [
            {"grid_n": 64, "interp_ms": 3.0, "ref_interp_ms": 30.0,
             "derivative_ms": 1.4, "design_rows": 3600,
             "libraries": list(cfg["libraries"]),
             "lstsq_ms": {name: 0.1 + 0.05 * k
                          for k, name in enumerate(cfg["libraries"])},
             "design_matrix_mb": {name: 0.1 for name in cfg["libraries"]}}], cfg)
        self.assertEqual({row["type"] for row in rows},
                         {"nominal", "curve", "limit", "cost_grid", "cost_library"})
        nominal = [row for row in rows if row["type"] == "nominal"]
        self.assertTrue(all(row["delta_alpha"] == "0.0" for row in nominal))
        self.assertTrue(all("tpr" not in row and "fpr_measured" not in row
                            for row in nominal))       # no fake sensitivity
        curve = [row for row in rows if row["type"] == "curve"]
        self.assertTrue(all(float(row["delta_alpha"]) > 0.0 for row in curve))
        self.assertTrue(all(int(row["n_ic"]) == cfg["n_ic"] for row in curve))
        self.assertTrue(all(int(row["n_noise"]) == cfg["n_noise"] for row in curve))
        self.assertIn(">0.5", {row["detection_limit_delta_alpha"]
                               for row in rows if row["type"] == "limit"})
        # shared cost once per grid, never repeated on a library row
        shared = [row for row in rows if row["type"] == "cost_grid"]
        self.assertEqual(len(shared), 1)
        self.assertTrue(all("derivative_ms" not in row and "interp_ms" not in row
                            for row in rows if row["type"] == "cost_library"))
        self.assertEqual(len({row["lstsq_ms"] for row in rows
                              if row["type"] == "cost_library"}),
                         len(cfg["libraries"]))
        saved = (RS.CSV_PATH, RS.FIG_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            RS.CSV_PATH = os.path.join(tmp, "reviewer_sensitivity.csv")
            RS.FIG_PATH = os.path.join(tmp, "fig_reviewer_sensitivity.png")
            try:
                csv_path, fig_path = RS.write_outputs(rows, results, cfg)
                self.assertGreater(os.path.getsize(fig_path), 0)
                with open(csv_path, newline="") as handle:
                    reader = _csv.DictReader(handle)
                    self.assertEqual(tuple(reader.fieldnames), RS.CSV_FIELDS)
                    self.assertEqual(len(list(reader)), len(rows))
            finally:
                RS.CSV_PATH, RS.FIG_PATH = saved

    def test_layout_guard_catches_an_overlapping_panel(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (left, right) = plt.subplots(1, 2, figsize=(3.0, 2.0))
        for ax in (left, right):
            ax.plot([0.0, 1.0], [0.0, 1.0])
            ax.tick_params(labelsize=RS.FONT_PT)
        box = left.get_position()                    # forced collision: the right
        right.set_position([box.x0 + 0.5 * box.width, box.y0,               # panel's
                            box.width, box.height])  # y labels land inside the left
        with self.assertRaises(RuntimeError):
            RS._check_layout(fig, {"left": left, "right": right})
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
