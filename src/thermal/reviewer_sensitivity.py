#!/usr/bin/env python3
"""
solver-forensics :: REVIEWER-REQUESTED SENSITIVITY SWEEP (heated channel, Pe=100)
=================================================================================
How much of the reported thermal detection limit depends on the observation
grid, on the chosen derivative library, and on the assumption of white Gaussian
observation noise?

Nothing about the physics or the solver changes.  The validated heated-channel
foundation (src/thermal/heated_channel.py) is reused as-is: the working 60x20
mesh at seed 2026, the fine 180x60 nominal-SUPG reference at seed 7001, the
thermal IC population (seeds 1000..), and the same measurand as
src/thermal/thermal_detection_limit.py -- the SUPG time-scale multiplier alpha,
swept 0.5..1.5 in 0.1 steps and reported separately for WEAKENING (alpha<1) and
STRENGTHENING (alpha>1).  Each configuration is solved once per IC and the
clean nodal fields are reused by every observer, so no FEM kernel is re-run.

THREE ONE-AT-A-TIME SLICES (not the full Cartesian product)
  grid slice     n in {32, 48, 64, 96} with the base5 library and white noise
  library slice  second3, base5, first_plus_base7, mixed_plus_base7, full9 at
                 the validated anchor grid n=64 with white noise
  noise slice    white (additive Gaussian), lowpass (smooth correlated),
                 gradient (boundary-layer weighted), multiplicative
                 (signal-dependent), at n=64 with base5
Each slice keeps both directions and the full alpha curve.  All four noise models
are zero-centred and rescaled to the SAME field-relative RMS, sigma*RMS(T) with
sigma=0.01, so they differ in noise STRUCTURE at matched corruption energy, and
the low-pass correlation length is a fixed fraction (CORR_FRACTION = 0.10) of
the domain side so that it means one physical length on every grid.  Each
(grid, noise model, alpha, IC) is observed under NOISE_REPLICATES = 5
independent realizations: the split-conformal readout runs once per realization
and the per-IC indicators are averaged over realizations before any interval is
formed, so replicates are nested inside an IC and never count as units.

COORDINATE NORMALIZATION (explicit, and identical to the anchor at n=64)
The channel is [0,LX]x[0,LY] with LX=3, LY=1.  Fields are interpolated onto the
normalized square (xi, eta) = (x/LX, y/LY) in [0,1]^2 at n points per side, so
the spacing is h=1/(n-1) on BOTH axes and every library term is a derivative
with respect to xi and eta.  At n=64 this reproduces supg_2d_engineering._H and
hence the published signatures bit-for-bit.  The aspect ratio LX/LY=3 is
absorbed into the coefficients, which is harmless for a unit-normalized
DIRECTION but means the components are not diffusivities.

WHAT IS REPORTED
Direction-resolved detection curves from the shared split-conformal detector
(src/measurement/cross_conformal.py), which fits each fold's classifier on its
own training ICs and thresholds on 19 disjoint calibration ICs, so the
false-alarm rate is MEASURED on untouched test ICs at a per-IC level bounded by
5%, and whose alpha=1 configuration is a nominal reference row rather than a
detector compared with itself; limits as the smallest SAMPLED positive
|delta_alpha| whose point TPR reaches 95%, right-censored when no sampled
detuning does, with IC-cluster bootstrap intervals over whole direction curves;
the mean nominal-to-changed angular separation as a descriptive geometric
diagnostic only, which is not a detector and does not predict detection; and
cost, i.e. median interpolation time per working and per fine reference field,
the median time to build the shared nine-term derivative library once per
observed field, each library's own median least-squares fit time, and
design-matrix memory.

Deterministic: one RNG stream per (grid n, noise model, alpha, IC, noise
replicate) and one per detection cell.  Artifacts are written ONLY when the
script is executed:
  results/tables/reviewer_sensitivity.csv , figures/fig_reviewer_sensitivity.png
Run:  python src/thermal/reviewer_sensitivity.py [--quick] [--dry-run]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import heated_channel as HC              # validated Pe=100 heated-channel solver
import cross_conformal as CC             # measured 5% split-conformal detection
from scipy.interpolate import griddata

CSV_PATH = os.path.join(_ROOT, "results", "tables", "reviewer_sensitivity.csv")
FIG_PATH = os.path.join(_ROOT, "figures", "fig_reviewer_sensitivity.png")

PE = 100.0
GRIDS = (32, 48, 64, 96)
DERIV_NAMES = ("u_xx", "u_yy", "u_xy", "u_xxx", "u_yyy",
               "u_x", "u_y", "u_xxy", "u_xyy")
LIBRARIES = {
    "second3": ("u_xx", "u_yy", "u_xy"),
    "base5": ("u_xx", "u_yy", "u_xy", "u_xxx", "u_yyy"),
    "first_plus_base7": ("u_xx", "u_yy", "u_xy", "u_xxx", "u_yyy", "u_x", "u_y"),
    "mixed_plus_base7": ("u_xx", "u_yy", "u_xy", "u_xxx", "u_yyy", "u_xxy", "u_xyy"),
    "full9": DERIV_NAMES,
}
NOISE_MODELS = ("white", "lowpass", "gradient", "multiplicative")
SIGMA = 0.01                 # field-relative RMS, matched across all noise models
CORR_FRACTION = 0.10         # low-pass correlation length as a fraction of the side
NOISE_REPLICATES = 5         # independent noise realizations per (grid, model, alpha, IC)
ANCHOR_GRID = 64             # validated anchor grid; library and noise slices sit here
ANCHOR_LIBRARY = "base5"     # anchor library; grid and noise slices sit here
ANCHOR_NOISE = "white"       # prior default noise; grid and library slices sit here
SLICES = ("grid", "library", "noise")
QUICK_N_IC = 48              # smallest validated ensemble: 6 folds, 21/19/8 split
TARGET_SENSITIVITY = 0.95
TARGET_FPR = CC.TARGET_FPR

CSV_FIELDS = ("type", "slice", "grid_n", "library", "n_terms", "noise",
              "direction", "alpha", "delta_alpha", "n_ic", "n_noise", "tpr",
              "tpr_lo", "tpr_hi", "fpr_measured", "fpr_lo", "fpr_hi",
              "fpr_target", "mean_angle_deg", "detection_limit_delta_alpha",
              "limit_lo", "limit_hi", "limit_censored_fraction", "interp_ms",
              "ref_interp_ms", "derivative_ms", "lstsq_ms", "design_matrix_mb",
              "design_rows")


# -----------------------------------------------------------------------------
# normalized observation grid and derivative libraries
# -----------------------------------------------------------------------------
def normalized_axes(n):
    """Return ``(xi, eta, h)`` on the unit square; physical x=xi*LX, y=eta*LY."""
    n = int(n)
    if n < 8:
        raise ValueError("the observation grid needs at least 8 points per side")
    s = np.linspace(0.0, 1.0, n)
    return s, s.copy(), float(s[1] - s[0])


def to_normalized_grid(pts, vals, n):
    """Interpolate a scattered channel field onto the normalized n x n grid.

    Cubic scattered interpolation with a nearest-neighbour fill on hull holes,
    i.e. the policy of the verified ``HC.to_channel_grid``, which this function
    reproduces exactly at ``n=64``.  The first array axis is streamwise xi.
    """
    xi, eta, _ = normalized_axes(n)
    XI, ETA = np.meshgrid(xi * HC.LX, eta * HC.LY, indexing="ij")
    query = np.column_stack((XI.ravel(), ETA.ravel()))
    out = griddata(pts, vals, query, method="cubic")
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = griddata(pts, vals, query[bad], method="nearest")
    return out.reshape(int(n), int(n))


def fd_library(U, names=DERIV_NAMES, h=None):
    """Finite-difference derivative library of the observed field.

    The anchor's stencils (``supg_2d_engineering._fd_library``) extended by the
    first derivatives and the two mixed third derivatives, taken with respect to
    the normalized coordinates with spacing ``h=1/(n-1)``.  Arrays are trimmed
    to ``slice(2, n-2)`` on both axes, exactly wide enough that no stencil point
    wraps around the periodic ``np.roll`` shifts.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("the observation grid must be square")
    n = U.shape[0]
    if n < 8:
        raise ValueError("the observation grid needs at least 8 points per side")
    if h is None:
        _, _, h = normalized_axes(n)
    r = lambda k, axis: np.roll(U, k, axis)
    d1 = lambda axis: (r(-1, axis) - r(1, axis)) / (2 * h)
    d2 = lambda axis: (r(-1, axis) - 2 * U + r(1, axis)) / h ** 2
    d3 = lambda axis: (r(-2, axis) - 2 * r(-1, axis)
                       + 2 * r(1, axis) - r(2, axis)) / (2 * h ** 3)
    cross = lambda f, axis: (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * h)
    # u_xy keeps the anchor's own 4-point form, expression for expression
    mixed_xy = lambda: (np.roll(r(-1, 0), -1, 1) - np.roll(r(-1, 0), 1, 1)
                        - np.roll(r(1, 0), -1, 1)
                        + np.roll(r(1, 0), 1, 1)) / (4 * h ** 2)
    stencil = {
        "u_x": lambda: d1(0), "u_y": lambda: d1(1),
        "u_xx": lambda: d2(0), "u_yy": lambda: d2(1), "u_xy": mixed_xy,
        "u_xxx": lambda: d3(0), "u_yyy": lambda: d3(1),
        "u_xxy": lambda: cross(d2(0), 1), "u_xyy": lambda: cross(d2(1), 0),
    }
    sl = slice(2, n - 2)
    for name in names:
        if name not in stencil:
            raise KeyError(f"unknown library term: {name}")
    return {name: stencil[name]()[sl, sl] for name in names}, sl


def design_matrix(Dlib, names):
    """Least-squares design matrix with the library's column order preserved."""
    return np.column_stack([Dlib[name].ravel() for name in names])


def design_matrix_bytes(n, n_terms):
    """Memory of one float64 design matrix on the trimmed n x n grid."""
    return int((int(n) - 4) ** 2 * int(n_terms) * 8)


def _fit_direction(Dlib, residual, names):
    """Unit least-squares coefficient direction of one library on one residual."""
    c, *_ = np.linalg.lstsq(design_matrix(Dlib, names), residual, rcond=None)
    nrm = float(np.linalg.norm(c))
    return c / nrm if nrm > 0 else c


def signature(Us, Ur, names=LIBRARIES[ANCHOR_LIBRARY], h=None):
    """Unit modified-equation coefficient direction of ``Us`` against ``Ur``.

    Identical to ``HC.sig_from_grid`` for the anchor library on the validated
    64x64 grid; generalized to any grid size and any named library.
    """
    Us, Ur = np.asarray(Us, dtype=float), np.asarray(Ur, dtype=float)
    if Us.shape != Ur.shape:
        raise ValueError("solver and reference grids must have the same shape")
    Dlib, sl = fd_library(Us, names, h)
    return _fit_direction(Dlib, (Us - Ur)[sl, sl].ravel(), names)


# -----------------------------------------------------------------------------
# structured observation noise at matched field-relative RMS
# -----------------------------------------------------------------------------
def noise_seed(grid_n, noise_index, alpha_index, ic_index, replicate):
    """One stable RNG stream per observed field, keyed by the actual grid size."""
    return (613_000 + 1_000_000 * int(grid_n) + 100_000 * int(noise_index)
            + 10_000 * int(replicate) + 100 * int(alpha_index) + int(ic_index))


def cell_seed(grid_n, library_index, noise_index, direction_index):
    """One stable split-conformal stream per detection cell."""
    return (742_000 + 1_000_000 * int(grid_n) + 100_000 * int(library_index)
            + 10_000 * int(noise_index) + 1_000 * int(direction_index))


def _match_rms(eta, target_rms):
    """Zero-centre the perturbation, then match its RMS to ``target_rms``."""
    eta = eta - float(np.mean(eta))
    current = float(np.sqrt(np.mean(eta ** 2)))
    return eta if current <= 0.0 else eta * (target_rms / current)


def _lowpass(xi, fraction=CORR_FRACTION):
    """White noise shaped by a Gaussian spectral envelope: smooth and correlated.

    The correlation length is ``fraction`` of the domain side rather than a fixed
    number of grid points, so the cutoff wavenumber in cycles per domain side is
    grid-independent and the corruption is the same physical field at every n.
    At n=64 a fraction of 0.10 is 6.4 grid points, i.e. the anchor's old choice.
    """
    n0, n1 = xi.shape
    kx, ky = np.fft.fftfreq(n0) * n0, np.fft.rfftfreq(n1) * n1
    kc = 1.0 / (2.0 * np.pi * float(fraction))        # cycles per domain side
    env = np.exp(-((kx[:, None] / kc) ** 2 + (ky[None, :] / kc) ** 2))
    env[0, 0] = 0.0                                   # zero-mean corruption
    return np.fft.irfft2(np.fft.rfft2(xi) * env, s=xi.shape)


def _gradient_weight(clean):
    """Normalized gradient magnitude of the clean field.

    The steep gradients sit in the thermal boundary layer on the heated wall and
    at the heater edges, so this weight makes the local noise-to-signal ratio
    strongly nonuniform while the global RMS stays matched to white Gaussian.
    """
    gx, gy = np.gradient(np.asarray(clean, dtype=float))
    w = np.sqrt(gx ** 2 + gy ** 2)
    peak = float(np.max(w))
    return w / peak if peak > 0.0 else np.ones_like(w)


def structured_noise(clean, model, seed, sigma=SIGMA):
    """Additive perturbation for ``model``, zero-centred and rescaled so that
    ``RMS(eta) == sigma*RMS(clean)`` exactly: the models differ only in spatial
    structure, never in corruption energy or mean level."""
    clean = np.asarray(clean, dtype=float)
    if model not in NOISE_MODELS:
        raise ValueError(f"unknown noise model: {model}")
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.0:
        return np.zeros_like(clean)
    xi = np.random.default_rng(int(seed)).standard_normal(clean.shape)
    eta = {"white": lambda: xi,
           "lowpass": lambda: _lowpass(xi),
           "gradient": lambda: _gradient_weight(clean) * xi,
           "multiplicative": lambda: clean * xi}[model]()
    return _match_rms(eta, sigma * float(np.sqrt(np.mean(clean ** 2))))


def add_structured_noise(clean, model, seed, sigma=SIGMA):
    """Observed field: clean field plus its matched-RMS structured corruption."""
    clean = np.asarray(clean, dtype=float)
    return clean + structured_noise(clean, model, seed, sigma)


# -----------------------------------------------------------------------------
# detection curves and angular separation
# -----------------------------------------------------------------------------
def mean_angular_separation(nominal, changed):
    """Mean per-IC angle in degrees between paired unit signature directions.

    A descriptive geometric diagnostic, not a detector.  The chord form
    ``2*arcsin(|a-b|/2)`` stays accurate for the small rotations produced by
    weak detuning, where the arccos form loses most of its significant digits.
    """
    a = np.atleast_2d(np.asarray(nominal, dtype=float))
    b = np.atleast_2d(np.asarray(changed, dtype=float))
    if a.shape != b.shape:
        raise ValueError("paired signature sets must have the same shape")
    def unit(v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norm > 0.0, norm, 1.0)
    chord = np.linalg.norm(unit(a) - unit(b), axis=1)
    return float(np.mean(np.degrees(
        2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0)))))


def detection_curve(signatures_by_alpha, alphas, seed):
    """Measured detection curve and sampled-grid limit for one direction.

    ``signatures_by_alpha`` maps alpha to a LIST of ``n_noise`` (n_ic, n_terms)
    signature arrays, one per independent noise realization, and must contain the
    nominal alpha=1 configuration.  The complete split-conformal readout runs
    once per realization and the per-IC detection and alarm indicators are
    averaged over realizations before any interval is formed, so a noise replicate
    is nested inside its IC and never counts as an independent unit.  alpha=1 is
    never compared with itself: only positive detunings carry a TPR, a measured
    false-alarm rate and intervals.
    """
    nominal = signatures_by_alpha[1.0]
    alphas = np.asarray([a for a in np.asarray(alphas, dtype=float)
                         if not np.isclose(a, 1.0)], dtype=float)
    deltas = np.abs(alphas - 1.0)
    detect, alarm, angles = [], [], []
    for index, alpha in enumerate(alphas):
        changed = signatures_by_alpha[float(alpha)]
        if len(changed) != len(nominal):
            raise ValueError("every alpha needs the same number of noise replicates")
        cells = [CC.split_conformal_detection(nominal[r], changed[r],
                                              seed=int(seed) + 10 * index + r)
                 for r in range(len(changed))]
        detect.append(np.mean([cell["detect"] for cell in cells], axis=0))
        alarm.append(np.mean([cell["alarm"] for cell in cells], axis=0))
        angles.append(float(np.mean([mean_angular_separation(nominal[r], changed[r])
                                     for r in range(len(changed))])))
    detect, alarm = np.vstack(detect), np.vstack(alarm)
    tpr = detect.mean(axis=1)
    ci = [CC.cluster_bootstrap_ci(np.column_stack([detect[j], alarm[j]]),
                                  seed=int(seed) + 500 + j)
          for j in range(len(alphas))]
    boot = CC.bootstrap_limit(deltas, detect, TARGET_SENSITIVITY,
                              seed=int(seed) + 900)
    return {"alphas": alphas, "deltas": deltas, "tpr": tpr,
            "tpr_ci": np.array([bounds[0] for bounds in ci]),
            "fpr": alarm.mean(axis=1),
            "fpr_ci": np.array([bounds[1] for bounds in ci]),
            "n_ic": detect.shape[1], "n_noise": len(nominal),
            "angles": np.array(angles), "limit_ci": boot["limit_ci"],
            "limit": CC.sampled_limit(deltas, tpr, TARGET_SENSITIVITY),
            "censored_fraction": boot["censored_fraction"]}


# -----------------------------------------------------------------------------
# protocol
# -----------------------------------------------------------------------------
def config(quick=False):
    """Full reviewer protocol, or a smaller subset for verification.

    The quick protocol keeps the exact 5% calibration and the measured level of
    the full one, so it still needs ``QUICK_N_IC`` ICs; it shrinks the alpha
    grid, the grid list and the library list instead.
    """
    if quick:
        weak, strong = np.array([1.0, 0.9, 0.5]), np.array([1.0, 1.1, 1.5])
        cfg = {"n_ic": QUICK_N_IC, "grids": (32, ANCHOR_GRID),
               "libraries": (ANCHOR_LIBRARY, "full9"), "quick": True}
    else:
        weak = np.round(np.array([1.0 - 0.1 * k for k in range(6)]), 3)
        strong = np.round(np.array([1.0 + 0.1 * k for k in range(6)]), 3)
        cfg = {"n_ic": HC.N_IC, "grids": GRIDS,
               "libraries": tuple(LIBRARIES), "quick": False}
    cfg.update({"pe": PE, "noise_models": NOISE_MODELS, "sigma": SIGMA,
                "n_noise": NOISE_REPLICATES, "alphas_weak": weak,
                "alphas_strong": strong,
                "alphas": np.round(np.array(sorted(set(weak) | set(strong))), 3)})
    return cfg


def directions(cfg):
    return (("weaken", cfg["alphas_weak"]), ("strengthen", cfg["alphas_strong"]))


def slice_cells(cfg):
    """The three requested slices as a deduplicated (grid, library, noise) list.

    Only one dimension moves at a time: the grid sweep holds the anchor library
    and white noise, the library sweep holds the anchor grid and white noise,
    the noise sweep holds the anchor grid and library.  This is a small fraction
    of the full Cartesian product, and each cell is evaluated once.
    """
    tags = {}
    for n in cfg["grids"]:
        tags.setdefault((int(n), ANCHOR_LIBRARY, ANCHOR_NOISE), set()).add("grid")
    for name in cfg["libraries"]:
        tags.setdefault((ANCHOR_GRID, name, ANCHOR_NOISE), set()).add("library")
    for model in cfg["noise_models"]:
        tags.setdefault((ANCHOR_GRID, ANCHOR_LIBRARY, model), set()).add("noise")
    return [(cell, tuple(sorted(names))) for cell, names in tags.items()]


def solve_clean_fields(cfg):
    """One FEM solve per configuration; nodal fields reused by every observer.

    Working and fine reference solves both come from ``HC.assemble_channel`` on
    the validated meshes, so this sweep adds no solver of its own.
    """
    ics = [HC.make_thermal_ic(1000 + i) for i in range(cfg["n_ic"])]
    a_th = HC.thermal_diffusivity(cfg["pe"])
    pts, elems, tags = HC.make_channel_mesh(60, 20, seed=2026)
    geom = HC.channel_mesh_geometry(pts, elems, a_th)
    ref_pts, ref_elems, ref_tags = HC.make_channel_mesh(180, 60, seed=7001)
    ref_geom = HC.channel_mesh_geometry(ref_pts, ref_elems, a_th)
    ref_nodal = [HC.assemble_channel("supg", ref_pts, ref_elems, ref_tags,
                                     cfg["pe"], ic=ic, alpha=1.0, geom=ref_geom)
                 for ic in ics]
    work_nodal = {float(alpha): [
        HC.assemble_channel("supg", pts, elems, tags, cfg["pe"], ic=ic,
                            alpha=float(alpha), geom=geom) for ic in ics]
        for alpha in cfg["alphas"]}
    print(f"  {len(ics)} fine reference solves + "
          f"{len(cfg['alphas']) * len(ics)} working solves cached (Pe={cfg['pe']:.0f})")
    return {"ics": ics, "pts": pts, "ref_pts": ref_pts,
            "ref_nodal": ref_nodal, "work_nodal": work_nodal}


def _timed_grids(pts, nodal_fields, n):
    """Interpolate a list of nodal fields to grid ``n``, timing each field."""
    grids, seconds = [], []
    for T in nodal_fields:
        started = time.perf_counter()
        grids.append(to_normalized_grid(pts, T, n))
        seconds.append(time.perf_counter() - started)
    return grids, seconds


def observe_on_grid(fields, cfg, n, pairs):
    """Interpolate to grid ``n``, corrupt, and recover the requested signatures.

    ``pairs`` lists the (library, noise) combinations this grid needs, so only
    the three requested slices are ever observed, and every (noise model, alpha,
    IC) is observed under ``cfg["n_noise"]`` independent noise realizations.
    Returns ``(signatures, cost)`` with ``signatures[noise][library][alpha]`` a
    list of ``n_noise`` ``(n_ic, n_terms)`` arrays.  The shared nine-term
    derivative library is built once per observed field and timed on its own;
    each library's least-squares fit is timed separately.
    """
    ref_grids, ref_times = _timed_grids(fields["ref_pts"], fields["ref_nodal"], n)
    work_grids, work_times = {}, []
    for alpha in cfg["alphas"]:
        grids, seconds = _timed_grids(fields["pts"],
                                      fields["work_nodal"][float(alpha)], n)
        work_grids[float(alpha)] = grids
        work_times.extend(seconds)

    _, _, h = normalized_axes(n)
    wanted = {}
    for library, noise in pairs:
        wanted.setdefault(noise, []).append(library)
    signatures = {noise: {library: {} for library in libraries}
                  for noise, libraries in wanted.items()}
    deriv_times, lstsq_times = [], {}
    for noise, libraries in wanted.items():
        for alpha_index, alpha in enumerate(cfg["alphas"]):
            buffers = {library: np.empty((cfg["n_noise"], cfg["n_ic"],
                                          len(LIBRARIES[library])))
                       for library in libraries}
            for replicate in range(cfg["n_noise"]):
                for i in range(cfg["n_ic"]):
                    observed = add_structured_noise(
                        work_grids[float(alpha)][i], noise,
                        noise_seed(n, NOISE_MODELS.index(noise), alpha_index, i,
                                   replicate), cfg["sigma"])
                    started = time.perf_counter()
                    Dlib, sl = fd_library(observed, DERIV_NAMES, h)
                    deriv_times.append(time.perf_counter() - started)
                    residual = (observed - ref_grids[i])[sl, sl].ravel()
                    for library in libraries:
                        started = time.perf_counter()
                        buffers[library][replicate, i] = _fit_direction(
                            Dlib, residual, LIBRARIES[library])
                        lstsq_times.setdefault(library, []).append(
                            time.perf_counter() - started)
            for library in libraries:
                signatures[noise][library][float(alpha)] = list(buffers[library])

    cost = {"grid_n": int(n), "design_rows": int((n - 4) ** 2),
            "interp_ms": 1.0e3 * float(np.median(work_times)),
            "ref_interp_ms": 1.0e3 * float(np.median(ref_times)),
            "derivative_ms": 1.0e3 * float(np.median(deriv_times)),
            "libraries": sorted(lstsq_times),
            "lstsq_ms": {library: 1.0e3 * float(np.median(times))
                         for library, times in lstsq_times.items()},
            "design_matrix_mb": {
                library: design_matrix_bytes(n, len(LIBRARIES[library])) / 1.0e6
                for library in lstsq_times}}
    print(f"  n={n:3d}: interp {cost['interp_ms']:7.2f} ms/working field "
          f"({cost['ref_interp_ms']:7.2f} ms/fine reference field) | shared "
          f"derivative library {cost['derivative_ms']:6.2f} ms/field | least squares "
          + ", ".join(f"{name} {value:.2f} ms"
                      for name, value in cost["lstsq_ms"].items())
          + f" | design {cost['design_rows']} rows")
    return signatures, cost


def evaluate(signatures, cfg, cells):
    """Measured detection curves and sampled-grid limits for one grid's cells."""
    results = []
    for (grid_n, library, noise), tags in cells:
        for index, (direction, alphas) in enumerate(directions(cfg)):
            curve = detection_curve(
                signatures[noise][library], alphas,
                cell_seed(grid_n, tuple(LIBRARIES).index(library),
                          NOISE_MODELS.index(noise), index))
            curve.update({"grid_n": int(grid_n), "library": library,
                          "n_terms": len(LIBRARIES[library]), "noise": noise,
                          "direction": direction, "slices": tags})
            results.append(curve)
    return results


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------
def _max_delta(cfg):
    return float(np.max(np.abs(cfg["alphas"] - 1.0)))


def _rows(results, costs, cfg):
    """CSV rows: one nominal reference per cell, the curve, the limit, the cost."""
    rows = []
    max_delta = _max_delta(cfg)
    for r in results:
        common = {"slice": "|".join(r["slices"]), "grid_n": r["grid_n"],
                  "library": r["library"], "n_terms": r["n_terms"],
                  "noise": r["noise"], "direction": r["direction"],
                  "n_ic": r["n_ic"], "n_noise": r["n_noise"],
                  "fpr_target": f"{TARGET_FPR:.2f}"}
        rows.append({"type": "nominal", "alpha": "1.0", "delta_alpha": "0.0",
                     **common})
        for j, (alpha, delta) in enumerate(zip(r["alphas"], r["deltas"])):
            rows.append({"type": "curve", "alpha": f"{alpha:.1f}",
                         "delta_alpha": f"{delta:.1f}",
                         "tpr": f"{r['tpr'][j]:.6f}",
                         "tpr_lo": f"{r['tpr_ci'][j][0]:.6f}",
                         "tpr_hi": f"{r['tpr_ci'][j][1]:.6f}",
                         "fpr_measured": f"{r['fpr'][j]:.6f}",
                         "fpr_lo": f"{r['fpr_ci'][j][0]:.6f}",
                         "fpr_hi": f"{r['fpr_ci'][j][1]:.6f}",
                         "mean_angle_deg": f"{r['angles'][j]:.4f}", **common})
        lo, hi = r["limit_ci"]
        rows.append({"type": "limit", "mean_angle_deg": f"{r['angles'][-1]:.4f}",
                     "detection_limit_delta_alpha": CC.limit_text(r["limit"], max_delta),
                     "limit_lo": CC.limit_text(lo, max_delta),
                     "limit_hi": CC.limit_text(hi, max_delta),
                     "limit_censored_fraction": f"{r['censored_fraction']:.4f}",
                     **common})
    for cost in costs:
        # shared work is reported once per grid; each library carries only its
        # own least-squares fit, so no amortized time is repeated on a row
        rows.append({"type": "cost_grid", "grid_n": cost["grid_n"],
                     "interp_ms": f"{cost['interp_ms']:.4f}",
                     "ref_interp_ms": f"{cost['ref_interp_ms']:.4f}",
                     "derivative_ms": f"{cost['derivative_ms']:.4f}",
                     "design_rows": cost["design_rows"]})
        for library in cost["libraries"]:
            rows.append({"type": "cost_library", "grid_n": cost["grid_n"],
                         "library": library, "n_terms": len(LIBRARIES[library]),
                         "lstsq_ms": f"{cost['lstsq_ms'][library]:.4f}",
                         "design_matrix_mb": f"{cost['design_matrix_mb'][library]:.4f}",
                         "design_rows": cost["design_rows"]})
    return rows


def _print_cells(results, cfg):
    """One line per cell and direction: the whole alpha curve, then the limit."""
    max_delta = _max_delta(cfg)
    deltas = results[0]["deltas"]
    print(f"\nMEASURED DETECTION AND SAMPLED-GRID LIMITS :: n={cfg['n_ic']} ICs is the"
          " independent unit and the effective sample size; the"
          f" {cfg['n_noise']} noise realizations per (alpha, IC) are nested inside their"
          " IC and are averaged into its indicators before any interval is formed."
          "  FPR is the false-alarm rate MEASURED on untouched test ICs at a per-IC"
          f" level bounded by {TARGET_FPR:.2f}; the limit is the smallest SAMPLED"
          f" |delta_alpha| with TPR >= {TARGET_SENSITIVITY:.2f}, '>' is right-censored"
          f" above {max_delta:.1f}, and no unmeasured gap is interpolated.  Per-point"
          " intervals are in the CSV.")
    print("  slice               n  library           noise           direction"
          + "".join(f"{'TPR@' + format(d, '.1f'):>9s}" for d in deltas)
          + "   FPR [min, max]      limit         95% CI  censored  angle [deg]")
    print("  " + "-" * (81 + 9 * len(deltas) + 60))
    for r in results:
        lo, hi = r["limit_ci"]
        print(f"  {'|'.join(r['slices']):<19s} {r['grid_n']:3d}  {r['library']:<17s}"
              f" {r['noise']:<15s} {r['direction']:>10s}"
              + "".join(f"{value:9.3f}" for value in r["tpr"])
              + f"   [{r['fpr'].min():.3f}, {r['fpr'].max():.3f}]"
              f"  {CC.limit_text(r['limit'], max_delta):>9}"
              f"  [{CC.limit_text(lo, max_delta):>5}, {CC.limit_text(hi, max_delta):>5}]"
              f"  {r['censored_fraction']:8.2f}  {r['angles'][-1]:11.2f}")


def _print_cost(costs):
    print("\nCOST :: shared work is reported once per grid, then each library's own"
          " least-squares fit (no time is amortized across libraries)")
    print("  grid  design rows  interp working [ms]  interp reference [ms]  "
          "shared derivative library [ms/field]")
    print("  " + "-" * 104)
    for cost in costs:
        print(f"  {cost['grid_n']:4d}  {cost['design_rows']:11d}  "
              f"{cost['interp_ms']:18.2f}  {cost['ref_interp_ms']:21.2f}  "
              f"{cost['derivative_ms']:36.2f}")
    print("  grid  library            n_terms  least squares [ms/fit]  design matrix [MB]")
    print("  " + "-" * 104)
    for cost in costs:
        for library in cost["libraries"]:
            print(f"  {cost['grid_n']:4d}  {library:<17s}  {len(LIBRARIES[library]):7d}"
                  f"  {cost['lstsq_ms'][library]:22.3f}"
                  f"  {cost['design_matrix_mb'][library]:18.3f}")


def _print_summary(results, cfg):
    max_delta = _max_delta(cfg)
    resolved = [r["limit"] for r in results if r["limit"] is not None]
    every_fpr = np.concatenate([r["fpr"] for r in results])
    print("\nSUMMARY (measurement, no new claim)")
    print(f"  cells: {len(results)} = {len(slice_cells(cfg))} one-at-a-time"
          f" configurations x 2 directions; {len(resolved)} reach"
          f" {TARGET_SENSITIVITY:.0%} TPR at a sampled detuning and"
          f" {len(results) - len(resolved)} are right-censored above {max_delta:.1f}.")
    if resolved:
        print(f"  sampled-grid limit over the resolved cells: min={min(resolved):.1f},"
              f" median={np.median(resolved):.1f}, max={max(resolved):.1f}"
              " in |delta_alpha|.")
    print(f"  measured false-alarm rate over all {every_fpr.size} evaluated points:"
          f" median={np.median(every_fpr):.3f}, max={every_fpr.max():.3f}, against the"
          f" {TARGET_FPR:.2f} per-IC level; per-point intervals are in the CSV.")
    for name in SLICES:
        cell = [r for r in results if name in r["slices"]]
        print(f"    {name:<8s} slice: {len(cell):2d} cells,"
              f" {sum(r['limit'] is None for r in cell)} censored, limits"
              f" {sorted({CC.limit_text(r['limit'], max_delta) for r in cell})}")
    print(f"  All noise models sit at the same field-relative RMS sigma="
          f"{cfg['sigma']:.2f}, so differences between them are structural.  The mean"
          " angular separation is a descriptive geometric diagnostic; the detection"
          " numbers come from the classifier, not from that angle.")


def write_outputs(rows, results, cfg):
    """Write the reviewer CSV and figure.  Called only from an executed run."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure(results, cfg)
    print(f"\nmetrics -> {CSV_PATH}")
    print(f"figure  -> {FIG_PATH}")
    return CSV_PATH, FIG_PATH


# -----------------------------------------------------------------------------
# figure: both directions, measured level, right-censored limits, grayscale
# -----------------------------------------------------------------------------
FONT_PT = 7.0
GRAYS = ("0.0", "0.35", "0.55", "0.70", "0.82")
MARKERS = ("o", "s", "^", "D", "v")
STYLE = {"weaken": ("-", None), "strengthen": ("--", "white")}


def _check_layout(fig, panels, min_pt=FONT_PT):
    """Render-time guard: no label is too small, off-canvas, or in another panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    furniture = {}

    def record(name, item):
        text = item.get_text() if hasattr(item, "get_text") else "legend"
        if not text or not item.get_visible():
            return
        size = getattr(item, "get_fontsize", lambda: min_pt)()
        if size < min_pt - 1.0e-9:
            raise RuntimeError(f"{name}: {text!r} is {size:.1f} pt < {min_pt} pt")
        box = item.get_window_extent(renderer)
        if box.width <= 0.0 or box.height <= 0.0:
            return
        if (box.x0 < -0.5 or box.y0 < -0.5
                or box.x1 > width + 0.5 or box.y1 > height + 0.5):
            raise RuntimeError(f"{name}: {text!r} leaves the canvas")
        furniture.setdefault(name, []).append((text, box))

    for name, ax in panels.items():
        items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            items += [label for loc, label in zip(axis.get_ticklocs(),
                                                  axis.get_ticklabels())
                      if low <= loc <= high]
        if ax.get_legend() is not None:
            items.append(ax.get_legend())
        for item in items:
            record(name, item)
    if getattr(fig, "_suptitle", None) is not None:
        record("figure title", fig._suptitle)
    for name, ax in panels.items():
        for other, items in furniture.items():
            for text, box in items:
                if other != name and ax.bbox.overlaps(box):
                    raise RuntimeError(f"{other} label {text!r} overlaps panel {name}")


def _curve_panel(ax, results, cfg, keys, key_of, title):
    """TPR against |delta_alpha| for both directions, one line per key."""
    for index, key in enumerate(keys):
        gray, marker = GRAYS[index % len(GRAYS)], MARKERS[index % len(MARKERS)]
        for direction, _ in directions(cfg):
            match = [r for r in results
                     if key_of(r) == key and r["direction"] == direction]
            if not match:
                continue
            ls, face = STYLE[direction]
            ax.plot(match[0]["deltas"], match[0]["tpr"], marker=marker, ms=3.4,
                    lw=1.2, color=gray, ls=ls, markerfacecolor=face or gray,
                    markeredgecolor=gray,
                    label=str(key) if direction == "weaken" else None)
    ax.axhline(TARGET_SENSITIVITY, color="0.2", ls=":", lw=0.9)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel(r"$|\Delta\alpha|$", fontsize=FONT_PT + 0.5)
    ax.set_ylabel("TPR (solid weaken, dashed strengthen)", fontsize=FONT_PT + 0.5)
    ax.set_title(title, fontsize=FONT_PT + 0.5)
    ax.tick_params(labelsize=FONT_PT)
    ax.legend(fontsize=FONT_PT, loc="upper left", frameon=False,
              handlelength=1.7, labelspacing=0.25, borderpad=0.2)


def _fpr_panel(ax, results):
    """Measured false-alarm rate of every cell against the conformal level."""
    for r in results:
        ax.plot(r["deltas"], r["fpr"], color="0.55", lw=0.8,
                ls=STYLE[r["direction"]][0], marker=".", ms=2.6)
    ax.axhline(TARGET_FPR, color="0.0", ls=":", lw=1.1)
    ax.set_ylim(0.0, max(0.12, max(r["fpr"].max() for r in results) + 0.02))
    ax.set_xlabel(r"$|\Delta\alpha|$", fontsize=FONT_PT + 0.5)
    ax.set_ylabel("measured false-alarm rate", fontsize=FONT_PT + 0.5)
    ax.set_title(f"(b) every cell, both directions;\ndotted line = {TARGET_FPR:.2f}"
                 " conformal level", fontsize=FONT_PT + 0.5)
    ax.tick_params(labelsize=FONT_PT)


def _limit_panel(ax, results, cfg):
    """Sampled-grid limits per cell and direction, with right-censored arrows."""
    max_delta = _max_delta(cfg)
    censored_y = max_delta + 0.08
    rank = {"grid": lambda r: r["grid_n"],
            "library": lambda r: tuple(LIBRARIES).index(r["library"]),
            "noise": lambda r: NOISE_MODELS.index(r["noise"])}
    ticks, labels, edges, position = [], [], [], 0.0
    for name in SLICES:
        for cell in sorted((r for r in results if name in r["slices"]
                            and r["direction"] == "weaken"), key=rank[name]):
            key = (cell["grid_n"], cell["library"], cell["noise"])
            for direction, _ in directions(cfg):
                match = [r for r in results if r["direction"] == direction
                         and (r["grid_n"], r["library"], r["noise"]) == key]
                if not match:
                    continue
                face = STYLE[direction][1]
                x = position + (-0.13 if direction == "weaken" else 0.13)
                if match[0]["limit"] is None:
                    ax.plot([x], [censored_y], marker="^", ms=5.0, color="0.0",
                            markerfacecolor=face or "0.0")
                    continue
                lo, hi = match[0]["limit_ci"]
                ax.plot([x, x], [match[0]["limit"] if lo is None else lo,
                                 censored_y if hi is None else hi],
                        color="0.5", lw=1.0)
                ax.plot([x], [match[0]["limit"]], ms=4.4, color="0.0",
                        marker="o" if direction == "weaken" else "s",
                        markerfacecolor=face or "0.0")
            ticks.append(position)
            labels.append({"grid": f"n={key[0]}", "library": key[1],
                           "noise": key[2]}[name])
            position += 1.0
        edges.append(position - 0.5)
        position += 0.6
    for edge in edges[:-1]:
        ax.axvline(edge, color="0.85", lw=0.8)
    ax.axhline(censored_y, color="0.85", lw=0.8, ls="--")
    ax.text(-0.45, censored_y + 0.01, f"right-censored (>{max_delta:.1f})",
            fontsize=FONT_PT, va="bottom")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FONT_PT)
    ax.set_xlim(-0.6, position - 0.9)
    ax.set_ylim(0.0, censored_y + 0.09)
    ax.set_ylabel(r"sampled-grid limit $|\Delta\alpha|$", fontsize=FONT_PT + 0.5)
    ax.set_title("(c) grid | library | noise slices: filled circle weakening, open"
                 " square strengthening, bars 95% IC-cluster bootstrap",
                 fontsize=FONT_PT + 0.5)
    ax.tick_params(labelsize=FONT_PT)


def figure(results, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.4, 6.2))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.99, top=0.88, bottom=0.185,
                          hspace=0.7, wspace=0.24)
    curves = fig.add_subplot(gs[0, 0])
    rates = fig.add_subplot(gs[0, 1])
    limits = fig.add_subplot(gs[1, :])
    _curve_panel(curves, [r for r in results if "noise" in r["slices"]], cfg,
                 list(cfg["noise_models"]), lambda r: r["noise"],
                 f"(a) noise structure at matched RMS, n={ANCHOR_GRID},"
                 f" {ANCHOR_LIBRARY}")
    _fpr_panel(rates, results)
    _limit_panel(limits, results, cfg)
    fig.suptitle("Reviewer sensitivity of the Pe=100 thermal detection limit:"
                 " observation grid, derivative library, noise structure",
                 fontsize=FONT_PT + 2.0, y=0.975)
    _check_layout(fig, {"(a)": curves, "(b)": rates, "(c)": limits})
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)
    return FIG_PATH


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
def run(quick=False, write=True):
    """Execute the sweep; ``write=False`` runs the full pipeline and writes nothing."""
    started = time.perf_counter()
    cfg = config(quick)
    cells = slice_cells(cfg)
    print("=" * 104)
    print("ICHMT REVIEWER SENSITIVITY :: OBSERVATION GRID, DERIVATIVE LIBRARY,"
          " AND STRUCTURED NOISE (Pe=100)")
    print("=" * 104)
    print(f"Protocol{' [QUICK]' if cfg['quick'] else ''}: Pe={cfg['pe']:.0f} |"
          f" {cfg['n_ic']} ICs (seeds 1000..{1000 + cfg['n_ic'] - 1}) | working mesh"
          " 60x20 seed 2026 | fine reference 180x60 seed 7001 |"
          f" alpha={[float(a) for a in cfg['alphas']]} reported as"
          f" weaken={[float(a) for a in cfg['alphas_weak']]} and"
          f" strengthen={[float(a) for a in cfg['alphas_strong']]}")
    print(f"          {len(cells)} one-at-a-time configurations:"
          f" grids={list(cfg['grids'])} at {ANCHOR_LIBRARY}+{ANCHOR_NOISE} |"
          f" libraries={list(cfg['libraries'])} at n={ANCHOR_GRID}+{ANCHOR_NOISE} |"
          f" noise={list(cfg['noise_models'])} at n={ANCHOR_GRID}+{ANCHOR_LIBRARY},"
          f" zero-centred at matched RMS sigma={cfg['sigma']:.2f} with a low-pass"
          f" correlation length of {CORR_FRACTION:.2f} of the domain side, and"
          f" {cfg['n_noise']} independent noise realizations per (grid, model, alpha,"
          " IC).  Derivatives are taken in (xi,eta)=(x/3,y/1) on [0,1]^2 with"
          " h=1/(n-1) on both axes, so n=64 reproduces the validated anchor grid.")
    folds = CC.choose_folds(cfg["n_ic"])
    print(f"Detection: shared split-conformal detector, {CC.REPEATS} repeats of"
          f" {folds} disjoint splits by IC"
          f" ({cfg['n_ic'] - CC.N_CAL - cfg['n_ic'] // folds} training,"
          f" {CC.N_CAL} calibration and {cfg['n_ic'] // folds} test ICs per fold)."
          "  One classifier per fold is fitted on its training ICs alone and then"
          " scores both the calibration nominal ICs and the untouched test ICs, so"
          f" the per-IC false-alarm probability is bounded by {TARGET_FPR:.2f} exactly"
          " under exchangeability and the rate is MEASURED.  The readout is repeated"
          " per noise realization and averaged within each IC.  Limits use the sampled"
          f" alpha grid only, are right-censored above {_max_delta(cfg):.1f}, and carry"
          f" {CC.N_BOOT}-replicate IC-cluster bootstrap intervals.")

    print("\nCLEAN FIELDS (validated heated-channel solver; one solve per configuration)")
    fields = solve_clean_fields(cfg)

    print("\nOBSERVATION AND SIGNATURE RECOVERY")
    results, costs = [], []
    for n in cfg["grids"]:
        grid_cells = [(cell, tags) for cell, tags in cells if cell[0] == int(n)]
        if not grid_cells:
            continue
        signatures, cost = observe_on_grid(
            fields, cfg, n, sorted({(cell[1], cell[2]) for cell, _ in grid_cells}))
        costs.append(cost)
        results.extend(evaluate(signatures, cfg, grid_cells))

    _print_cells(results, cfg)
    _print_cost(costs)
    _print_summary(results, cfg)

    rows = _rows(results, costs, cfg)
    paths = write_outputs(rows, results, cfg) if write else (None, None)
    if not write:
        print("\n[dry run] no CSV or figure written")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.1f}")
    return {"config": cfg, "results": results, "costs": costs, "rows": rows,
            "paths": paths, "runtime_seconds": elapsed}


def build_parser():
    parser = argparse.ArgumentParser(
        description="reviewer-requested sensitivity sweep for the Pe=100 heated channel")
    parser.add_argument("--quick", action="store_true",
                        help="smaller IC/alpha/grid subset for verification")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the pipeline without writing the CSV or figure")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(quick=args.quick, write=not args.dry_run)


if __name__ == "__main__":
    main()
