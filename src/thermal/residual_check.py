import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import heated_channel as HC       # the validated thermal foundation
import supg_2d_engineering as A    # stat helpers: A.feats, A.cv_acc, A.perm_floor, A._clf, A.LIB
from sklearn.model_selection import GroupKFold, cross_val_predict
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
import time


"""ICHMT robustness spot-check: discrepancy versus governing-equation residual.

The FEM solve, mesh generation, and fine reference are all supplied by the
validated heated-channel foundation.  This analysis only changes how the
observed 64x64 solver field is converted into a five-coefficient signature.
"""

PE = 100.0
ALPHAS = (0.0, 0.5, 1.0, 1.5)
DETUNED_ALPHAS = (0.0, 0.5, 1.5)
NOISE = (0.0, 0.01)
N_IC = 48
WORKING_MESH = (60, 20, 2026)
IC_SEED_BASE = 1000
NOISE_SEED_BASE = 412000
TARGET_FPR = 0.05
DETECT_BAR = 0.50
CSV_FIELDS = (
    "construction", "alpha", "noise", "detection_sensitivity",
    "sig_dist_from_nominal",
)


# signature detector: held-out classifier P(changed), threshold on nominal
# scores at 5% FPR, sensitivity=TPR
def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    X = A.feats(np.vstack([nominal, changed]))
    y = np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(changed))]
    k = min(5, len(nominal))
    proba = cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k), method="predict_proba")[:, 1]
    thr = np.percentile(proba[y == 0], 100 * (1 - target_fpr))
    return float(np.mean(proba[y == 1] > thr))


# scalar detector for a 1-D thermal output v: two-sided |v - median(nominal)|,
# LOO-free threshold at 5% FPR.  It is retained with the repo recipe so this
# spot-check has the same detector vocabulary as the thermal baselines.
def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    nom = np.asarray(nom_vals, float)
    chg = np.asarray(chg_vals, float)
    center = np.median(nom)
    thr = np.percentile(np.abs(nom - center), 100 * (1 - target_fpr))
    return float(np.mean(np.abs(chg - center) > thr))


def _unit_direction(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    norm = np.linalg.norm(coefficients)
    return coefficients / norm if norm > 0.0 else coefficients


def _channel_fd_library(U, Lx=HC.LX, Ly=HC.LY):
    """Return physical-coordinate central-FD derivatives on the shared interior.

    This mirrors ``A._fd_library`` while using the heated-channel window's
    physical spacings.  The first axis is x and the second axis is y, as in
    ``HC.to_channel_grid``.  The same five derivative directions in ``A.LIB``
    are returned, so the residual fit and the existing signature have the same
    coefficient ordering.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] < 5 or U.shape[1] < 5:
        raise ValueError("U must be a two-dimensional grid with at least 5 points per axis")
    nx, ny = U.shape
    hx = float(Lx) / (nx - 1)
    hy = float(Ly) / (ny - 1)
    if hx <= 0.0 or hy <= 0.0:
        raise ValueError("grid window lengths must be positive")

    uxx = (np.roll(U, -1, 0) - 2.0 * U + np.roll(U, 1, 0)) / hx ** 2
    uyy = (np.roll(U, -1, 1) - 2.0 * U + np.roll(U, 1, 1)) / hy ** 2
    uxy = (
        np.roll(np.roll(U, -1, 0), -1, 1)
        - np.roll(np.roll(U, -1, 0), 1, 1)
        - np.roll(np.roll(U, 1, 0), -1, 1)
        + np.roll(np.roll(U, 1, 0), 1, 1)
    ) / (4.0 * hx * hy)
    # Keep the third-derivative stencil/sign convention used by A._fd_library.
    uxxx = (
        np.roll(U, -2, 0) - 2.0 * np.roll(U, -1, 0)
        + 2.0 * np.roll(U, 1, 0) - np.roll(U, 2, 0)
    ) / (2.0 * hx ** 3)
    uyyy = (
        np.roll(U, -2, 1) - 2.0 * np.roll(U, -1, 1)
        + 2.0 * np.roll(U, 1, 1) - np.roll(U, 2, 1)
    ) / (2.0 * hy ** 3)
    sl = (slice(2, nx - 2), slice(2, ny - 2))
    return {
        "u_xx": uxx[sl],
        "u_yy": uyy[sl],
        "u_xy": uxy[sl],
        "u_xxx": uxxx[sl],
        "u_yyy": uyyy[sl],
    }, sl, hx, hy


def pde_residual_signature(Us, a_th, Lx=HC.LX, Ly=HC.LY):
    """Fit the continuous channel PDE residual to the shared derivative library."""
    Us = np.asarray(Us, dtype=float)
    Dlib, sl, hx, _ = _channel_fd_library(Us, Lx=Lx, Ly=Ly)
    du_dx = (np.roll(Us, -1, 0) - np.roll(Us, 1, 0)) / (2.0 * hx)
    uxx = (np.roll(Us, -1, 0) - 2.0 * Us + np.roll(Us, 1, 0)) / hx ** 2
    hy = float(Ly) / (Us.shape[1] - 1)
    uyy = (np.roll(Us, -1, 1) - 2.0 * Us + np.roll(Us, 1, 1)) / hy ** 2
    y = np.linspace(0.0, float(Ly), Us.shape[1])
    ux = HC.channel_velocity(y, Ly=Ly, Umax=HC.UMAX)
    residual = ux[None, :] * du_dx - float(a_th) * (uxx + uyy)

    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    rhs = residual[sl].ravel()
    coefficients, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)
    return _unit_direction(coefficients)


def _add_grid_noise(Us, sigma, seed):
    """Add deterministic RMS-relative Gaussian observation noise after the solve."""
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(Us ** 2))
    return Us + float(sigma) * rms * rng.standard_normal(Us.shape)


def _noise_seed(sigma_index, alpha_index, ic_index):
    """Stable stream per observed (noise, alpha, IC) field."""
    return NOISE_SEED_BASE + 100000 * sigma_index + 1000 * alpha_index + ic_index


def _solve_clean_grids(pts, elems, tags, geom, ics):
    """Solve each SUPG tau-scale/IC pair once and retain only observed grids."""
    clean = {}
    for alpha in ALPHAS:
        grids = []
        for ic in ics:
            T = HC.assemble_channel(
                "supg", pts, elems, tags, PE, ic=ic, alpha=float(alpha),
                geom=geom,
            )
            grids.append(HC.to_channel_grid(pts, T))
        clean[float(alpha)] = np.asarray(grids, dtype=float)
    return clean


def _observed_signature_sets(clean, refs, a_th, sigma, sigma_index):
    """Build both signatures from one noisy grid per (alpha, IC)."""
    observed = {}
    for alpha_index, alpha in enumerate(ALPHAS):
        discrepancy = np.empty((N_IC, len(A.LIB)), dtype=float)
        pde_residual = np.empty((N_IC, len(A.LIB)), dtype=float)
        for ic_index, (Us, reference) in enumerate(zip(clean[float(alpha)], refs)):
            noisy = _add_grid_noise(
                Us, sigma,
                _noise_seed(sigma_index, alpha_index, ic_index),
            )
            discrepancy[ic_index] = HC.sig_from_grid(noisy, reference)
            pde_residual[ic_index] = pde_residual_signature(noisy, a_th)
        observed[float(alpha)] = {
            "discrepancy": discrepancy,
            "pde_residual": pde_residual,
        }
    return observed


def _mean_direction(signatures):
    return _unit_direction(np.mean(np.asarray(signatures, dtype=float), axis=0))


def _direction_distance(direction, nominal_direction):
    cosine = float(np.dot(_unit_direction(direction), _unit_direction(nominal_direction)))
    return float(1.0 - abs(np.clip(cosine, -1.0, 1.0)))


def _collect_rows(clean, refs, a_th):
    rows = []
    nominal_cross_distance = {}
    sensitivity_map = {}
    for sigma_index, sigma in enumerate(NOISE):
        observed = _observed_signature_sets(clean, refs, a_th, sigma, sigma_index)
        nominal = observed[1.0]
        nominal_directions = {
            construction: _mean_direction(nominal[construction])
            for construction in ("discrepancy", "pde_residual")
        }
        nominal_cross_distance[float(sigma)] = _direction_distance(
            nominal_directions["discrepancy"], nominal_directions["pde_residual"])

        for construction in ("discrepancy", "pde_residual"):
            nominal_direction = nominal_directions[construction]
            for alpha in ALPHAS:
                changed = observed[float(alpha)][construction]
                sensitivity = supervised_sensitivity(
                    nominal[construction], changed, target_fpr=TARGET_FPR,
                )
                distance = _direction_distance(
                    _mean_direction(changed), nominal_direction,
                )
                sensitivity_map[(construction, float(sigma), float(alpha))] = sensitivity
                rows.append({
                    "construction": construction,
                    "alpha": float(alpha),
                    "noise": float(sigma),
                    "detection_sensitivity": sensitivity,
                    "sig_dist_from_nominal": distance,
                })
    return rows, nominal_cross_distance, sensitivity_map


def _write_csv(rows):
    path = os.path.join(TAB, "residual_check.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "construction": row["construction"],
                "alpha": f"{row['alpha']:.1f}",
                "noise": f"{row['noise']:.2f}",
                "detection_sensitivity": f"{row['detection_sensitivity']:.6f}",
                "sig_dist_from_nominal": f"{row['sig_dist_from_nominal']:.6f}",
            })
    return path


def _print_result_table(rows):
    print("\nRESULT TABLE :: 5%-FPR held-out sensitivity and mean signature distance")
    print("  construction   alpha  noise  detection_sensitivity  sig_dist_from_nominal")
    print("  " + "-" * 76)
    for row in rows:
        print(
            f"  {row['construction']:<13} {row['alpha']:5.1f}  {row['noise']:5.2f}"
            f"        {row['detection_sensitivity']:8.3f}              {row['sig_dist_from_nominal']:8.3f}"
        )


def _print_nominal_cross_distance(distances):
    print("\nNOMINAL CONSTRUCTION DIRECTION DISTANCE")
    print("  noise  1-|cos(mean discrepancy, mean pde_residual)|")
    print("  " + "-" * 55)
    for sigma in NOISE:
        print(f"  {sigma:5.2f}                         {distances[float(sigma)]:8.3f}")


def _verdict(sensitivity_map):
    comparisons = []
    for sigma in NOISE:
        for alpha in DETUNED_ALPHAS:
            discrepancy = sensitivity_map[("discrepancy", float(sigma), float(alpha))]
            pde_residual = sensitivity_map[("pde_residual", float(sigma), float(alpha))]
            comparisons.append({
                "sigma": float(sigma),
                "alpha": float(alpha),
                "discrepancy": discrepancy,
                "pde_residual": pde_residual,
                "discrepancy_detects": discrepancy >= DETECT_BAR,
                "pde_detects": pde_residual >= DETECT_BAR,
            })

    consistent = all(item["discrepancy_detects"] == item["pde_detects"] for item in comparisons)
    pde_noise_failures = [
        item for item in comparisons
        if item["sigma"] > 0.0
        and item["discrepancy_detects"]
        and not item["pde_detects"]
    ]
    if consistent:
        verdict = "GO"
        label = "ROBUST"
    elif pde_noise_failures:
        verdict = "CHECK"
        label = "DISCREPANCY_PREFERABLE"
    else:
        verdict = "CHECK"
        label = "NOT_ROBUST"

    print("\n" + "=" * 104)
    print(f"[{verdict} / VERDICT] {label}")
    if consistent:
        print(
            f"  Both constructions make the same >= {DETECT_BAR:.2f} detection decisions"
            " for every detuned alpha at both noise levels."
        )
    elif pde_noise_failures:
        details = "; ".join(
            f"alpha={item['alpha']:g}, noise={item['sigma']:.2f}"
            f" (discrepancy={item['discrepancy']:.3f}, PDE={item['pde_residual']:.3f})"
            for item in pde_noise_failures
        )
        print("  PDE residual fails where discrepancy succeeds: " + details)
        print(
            "  The discrepancy construction is preferable because the PDE residual"
            " amplifies observation noise through its second derivatives."
        )
    else:
        details = "; ".join(
            f"alpha={item['alpha']:g}, noise={item['sigma']:.2f}"
            f" (discrepancy={item['discrepancy']:.3f}, PDE={item['pde_residual']:.3f})"
            for item in comparisons
            if item["discrepancy_detects"] != item["pde_detects"]
        )
        print("  The constructions disagree on: " + details)
        print("  This spot-check does not support a robust equivalence verdict.")
    return verdict, comparisons


def main():
    started = time.perf_counter()
    print("=" * 104)
    print("ICHMT :: PDE-RESIDUAL SIGNATURE ROBUSTNESS SPOT-CHECK")
    print("Solution discrepancy versus continuous governing-equation residual")
    print("=" * 104)
    print(
        f"Protocol: Pe={PE:.0f} | {N_IC} ICs (seeds {IC_SEED_BASE}..{IC_SEED_BASE + N_IC - 1})"
        f" | working mesh={WORKING_MESH[0]}x{WORKING_MESH[1]}, seed={WORKING_MESH[2]}"
    )
    print(f"          alphas={list(ALPHAS)} | noise={list(NOISE)} | nominal alpha=1.0")
    print(
        "Observation rule: solve each clean SUPG field once per (alpha, IC), then add"
        " deterministic RMS-relative Gaussian noise before both signature constructions."
    )
    print(
        "PDE residual: r_L = u_x(y) * du/dx - a_th * (d2u/dx2 + d2u/dy2),"
        " f=0, fitted to the same A.LIB directions."
    )

    ics = [HC.make_thermal_ic(IC_SEED_BASE + i) for i in range(N_IC)]
    pts, elems, tags = HC.make_channel_mesh(*WORKING_MESH)
    a_th = HC.thermal_diffusivity(PE)
    geom = HC.channel_mesh_geometry(pts, elems, a_th)

    print("\nCaching fine nominal-SUPG reference grids once per IC ...")
    refs = HC.reference_channel_grids(ics, Pe=PE)
    print(f"  cached {len(refs)} fine reference grids")
    print("Caching clean working-mesh SUPG grids once per alpha and IC ...")
    clean = _solve_clean_grids(pts, elems, tags, geom, ics)
    print(f"  cached {sum(len(grids) for grids in clean.values())} clean solver grids")

    rows, nominal_cross_distance, sensitivity_map = _collect_rows(clean, refs, a_th)
    _print_result_table(rows)
    _print_nominal_cross_distance(nominal_cross_distance)
    print(
        f"\nchance/FPR: binary classifier chance=0.500; threshold calibrated at target FPR={TARGET_FPR:.3f}"
        " (95.0% nominal specificity) from held-out nominal probabilities."
    )
    print(
        f"decision bar for the verdict: sensitivity >= {DETECT_BAR:.2f} means detect;"
        " alpha=1.0 rows are the nominal self-control."
    )
    csv_path = _write_csv(rows)
    print(f"CSV -> {csv_path}")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    _verdict(sensitivity_map)


if __name__ == "__main__":
    main()
