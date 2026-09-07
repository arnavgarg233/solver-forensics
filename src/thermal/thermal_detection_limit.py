import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))
import heated_channel as HC       # the validated thermal foundation
import supg_2d_engineering as A    # grid/library anchor: A.GRID_OBS, A.LIB
import cross_conformal as CC       # measured 5% split-conformal detection
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
import time


# The comparison deliberately varies only the SUPG tau scale.  The directions
# are reported independently because an under- and over-stabilized method need
# not leave the same thermal fingerprint.
PES = (50, 100, 200)
ALPHAS_WEAK = np.array((1.0, 0.9, 0.8, 0.7, 0.6, 0.5), dtype=float)
ALPHAS_STRONG = np.array((1.0, 1.1, 1.2, 1.3, 1.4, 1.5), dtype=float)
DIRECTIONS = (("weaken", ALPHAS_WEAK), ("strengthen", ALPHAS_STRONG))
ALL_ALPHAS = np.array(sorted(set(ALPHAS_WEAK) | set(ALPHAS_STRONG)), dtype=float)
SIGMAS = (0.0, 0.01, 0.05)
TARGET_FPR = CC.TARGET_FPR
TARGET_SENSITIVITY = 0.95
N_IC = HC.N_IC
MAX_DELTA = 0.5

# Every detector, including the one-dimensional thermal outputs and the joint
# [Twall_max, Nu_mean] pair, is read out by the SAME supervised split-conformal
# classifier, so the comparison arms differ only in what the detector observes.
DETECTORS = ("signature", "thermal_pair", "Twall_max", "Nu_mean", "fullfield_T")
CONVENTIONAL = ("thermal_pair", "Twall_max", "Nu_mean")
# "sensitivity" keeps its name: it is the measured TPR column that the committed
# figure builder (src/thermal/make_thermal_figures.py) already reads.
CSV_FIELDS = (
    "type", "detector", "direction", "Pe", "sigma", "alpha", "delta_alpha",
    "n_ic", "sensitivity", "tpr_lo", "tpr_hi", "fpr_measured", "fpr_lo",
    "fpr_hi", "fpr_target", "detection_limit_delta_alpha", "limit_lo",
    "limit_hi", "limit_censored_fraction",
)


def _cell_seed(pe_index, sigma_index, direction_index, detector_index, alpha_index):
    """One stable split-conformal stream per (Pe, sigma, direction, detector, alpha)."""
    return (531_000 + 100_000 * pe_index + 10_000 * sigma_index
            + 1_000 * direction_index + 100 * detector_index + alpha_index)


def _noise_seed(pe_index, sigma_index, alpha_index, ic_index):
    """One stable stream per observed field; all detectors consume that field."""
    return 417_000 + 10_000 * pe_index + 1_000 * sigma_index + 100 * alpha_index + ic_index


def _add_grid_noise(grid, sigma, seed):
    """RMS-relative Gaussian observation noise used throughout the repository."""
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(grid ** 2))
    return grid + sigma * rms * rng.standard_normal(grid.shape)


def _grid_thermal_scalars(grid, ic, a_th):
    """Recover Twall_max and Nu_mean from the same noisy 64x64 observation.

    The grid's first axis is x and its second is y.  We select the bottom-wall
    columns within the IC-specific heater interval, then compute a
    velocity-weighted bulk temperature at each selected x column.  This is a
    deliberately simple observation-space analogue of HC.thermal_outputs:
    it prevents the scalar baselines from seeing clean nodal FEM temperatures
    while the signature sees noisy grid data.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.shape != (A.GRID_OBS, A.GRID_OBS):
        raise ValueError("thermal scalar extraction expects the validated 64x64 grid")

    x = np.linspace(0.0, HC.LX, A.GRID_OBS)
    y = np.linspace(0.0, HC.LY, A.GRID_OBS)
    heater = (x >= ic["xh0"]) & (x <= ic["xh1"])
    if not np.any(heater):
        raise RuntimeError("the IC heater interval has no observed grid columns")

    wall = grid[heater, 0]
    velocity = HC.channel_velocity(y, Ly=HC.LY, Umax=HC.UMAX)
    flow = float(np.trapezoid(velocity, y))
    if flow <= 0.0:
        raise RuntimeError("nonpositive grid flow integral")
    bulk = np.trapezoid(grid[heater, :] * velocity[None, :], y, axis=1) / flow
    delta_t = wall - bulk
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = (float(ic["g_flux"]) / float(a_th)) * (2.0 * HC.LY) / delta_t
    values = {"Twall_max": float(np.max(wall)), "Nu_mean": float(np.mean(nu))}
    if not np.all(np.isfinite(list(values.values()))):
        raise RuntimeError("noisy-grid thermal scalar extraction produced a non-finite value")
    return values


def _build_clean_cache(pts, elems, tags, pe, ics, geom):
    """Solve each working SUPG configuration only once per IC.

    Both clean grids and the exact validated FEM thermal outputs are retained.
    The latter satisfy the requested cache requirement and provide a clean
    diagnostic trail; detection itself uses the noisy-grid scalar measurements
    made later by _grid_thermal_scalars.
    """
    a_th = HC.thermal_diffusivity(pe)
    clean_grids = {}
    clean_outputs = {}
    for alpha in ALL_ALPHAS:
        grids = np.empty((N_IC, A.GRID_OBS, A.GRID_OBS), dtype=float)
        twall = np.empty(N_IC, dtype=float)
        nu = np.empty(N_IC, dtype=float)
        for i, ic in enumerate(ics):
            temperature, active_tags, _ = HC.assemble_channel(
                "supg", pts, elems, tags, pe, ic=ic, alpha=float(alpha),
                geom=geom, return_meta=True,
            )
            grids[i] = HC.to_channel_grid(pts, temperature, tags["Lx"], tags["Ly"])
            output = HC.thermal_outputs(
                pts, temperature, active_tags, a_th, g_flux=ic["g_flux"],
            )
            twall[i] = output["Twall_max"]
            nu[i] = output["Nu_mean"]
        clean_grids[float(alpha)] = grids
        clean_outputs[float(alpha)] = {"Twall_max": twall, "Nu_mean": nu}
    return clean_grids, clean_outputs


def _observe_detectors(clean_grids, refs, ics, a_th, pe_index, sigma_index, sigma):
    """Compute every detector input from one noisy grid per (alpha, IC).

    The single ``observed`` grid below feeds the signature, both conventional
    thermal scalars, their joint pair, and the naive full-field reference
    distance.  Thus their performance differences cannot come from unequal noise
    realizations, and every arm is read out by the same classifier afterwards.
    """
    observed = {}
    for alpha_index, alpha in enumerate(ALL_ALPHAS):
        signatures = np.empty((N_IC, len(A.LIB)), dtype=float)
        twall = np.empty(N_IC, dtype=float)
        nu = np.empty(N_IC, dtype=float)
        fullfield = np.empty(N_IC, dtype=float)
        for i, ic in enumerate(ics):
            noisy_grid = _add_grid_noise(
                clean_grids[float(alpha)][i], sigma,
                _noise_seed(pe_index, sigma_index, alpha_index, i),
            )
            signatures[i] = HC.sig_from_grid(noisy_grid, refs[i])
            scalars = _grid_thermal_scalars(noisy_grid, ic, a_th)
            twall[i] = scalars["Twall_max"]
            nu[i] = scalars["Nu_mean"]
            fullfield[i] = np.linalg.norm(noisy_grid - refs[i])
        observed[float(alpha)] = {
            "signature": signatures,
            "thermal_pair": np.column_stack([twall, nu]),
            "Twall_max": twall,
            "Nu_mean": nu,
            "fullfield_T": fullfield,
        }
    return observed


def _direction_result(observed, alphas, pe_index, sigma_index, direction_index):
    """Measured detection curves and sampled-grid limits for one direction.

    alpha=1 is the nominal reference and is never compared with itself, so only
    the positive detunings carry a TPR, a measured false-alarm rate and
    intervals.  The limit is the smallest SAMPLED positive delta whose point TPR
    reaches the target, right-censored when no sampled delta does; its interval
    resamples IC-level indicator vectors across the complete direction curve.
    """
    alphas = np.asarray([a for a in np.asarray(alphas, float)
                         if not np.isclose(a, 1.0)], dtype=float)
    deltas = np.abs(alphas - 1.0)
    nominal = observed[1.0]
    tpr, tpr_ci, fpr, fpr_ci, limits = {}, {}, {}, {}, {}
    for detector_index, detector in enumerate(DETECTORS):
        cells = [
            CC.split_conformal_detection(
                nominal[detector], observed[float(alpha)][detector],
                seed=_cell_seed(pe_index, sigma_index, direction_index,
                                detector_index, alpha_index))
            for alpha_index, alpha in enumerate(alphas)
        ]
        tpr[detector] = np.array([cell["tpr"] for cell in cells])
        fpr[detector] = np.array([cell["fpr"] for cell in cells])
        tpr_ci[detector] = np.array([cell["tpr_ci"] for cell in cells])
        fpr_ci[detector] = np.array([cell["fpr_ci"] for cell in cells])
        limits[detector] = {
            "limit": CC.sampled_limit(deltas, tpr[detector], TARGET_SENSITIVITY),
            **CC.bootstrap_limit(
                deltas, np.vstack([cell["detect"] for cell in cells]),
                TARGET_SENSITIVITY,
                seed=_cell_seed(pe_index, sigma_index, direction_index,
                                detector_index, 90)),
        }
    return {"alphas": alphas, "deltas": deltas, "tpr": tpr, "tpr_ci": tpr_ci,
            "fpr": fpr, "fpr_ci": fpr_ci, "limits": limits, "n": N_IC}


def _csv_rows(result):
    """One nominal reference row, one measured curve row per detuning, one limit row.

    The delta=0 nominal configuration carries no sensitivity: it is the reference
    the detuned rows are measured against, not a detector against itself.
    """
    common = {"direction": result["direction"], "Pe": result["Pe"],
              "sigma": f"{result['sigma']:.2f}", "n_ic": result["n"],
              "fpr_target": f"{TARGET_FPR:.2f}"}
    rows = []
    for detector in DETECTORS:
        rows.append({"type": "nominal", "detector": detector, "alpha": "1.0",
                     "delta_alpha": "0.0", **common})
        for j, (alpha, delta) in enumerate(zip(result["alphas"], result["deltas"])):
            rows.append({
                "type": "curve", "detector": detector, "alpha": f"{alpha:.1f}",
                "delta_alpha": f"{delta:.1f}",
                "sensitivity": f"{result['tpr'][detector][j]:.6f}",
                "tpr_lo": f"{result['tpr_ci'][detector][j][0]:.6f}",
                "tpr_hi": f"{result['tpr_ci'][detector][j][1]:.6f}",
                "fpr_measured": f"{result['fpr'][detector][j]:.6f}",
                "fpr_lo": f"{result['fpr_ci'][detector][j][0]:.6f}",
                "fpr_hi": f"{result['fpr_ci'][detector][j][1]:.6f}", **common})
        limit = result["limits"][detector]
        lo, hi = limit["limit_ci"]
        rows.append({
            "type": "limit", "detector": detector,
            "detection_limit_delta_alpha": CC.limit_text(limit["limit"], MAX_DELTA),
            "limit_lo": CC.limit_text(lo, MAX_DELTA),
            "limit_hi": CC.limit_text(hi, MAX_DELTA),
            "limit_censored_fraction": f"{limit['censored_fraction']:.4f}", **common})
    return rows


def _write_csv(rows):
    path = os.path.join(TAB, "thermal_detection_limit.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _limit_for_comparison(value):
    return np.inf if value is None else float(value)


def _print_curve_table(result):
    print(f"\nRESULT TABLE :: Pe={result['Pe']} | sigma={result['sigma']:.2f} | "
          f"{result['direction']} SUPG detuning | n={result['n']} ICs (independent unit)")
    print("  TPR of each detector at its measured split-conformal threshold")
    print("  alpha  delta_a" + "".join(f"{name:>14s}" for name in DETECTORS))
    print("  " + "-" * (15 + 14 * len(DETECTORS)))
    for j, (alpha, delta) in enumerate(zip(result["alphas"], result["deltas"])):
        print(f"  {alpha:5.1f}   {delta:5.1f}  "
              + "".join(f"{result['tpr'][name][j]:14.3f}" for name in DETECTORS))
    print("  measured false-alarm rate over this curve [min, max], target "
          f"{TARGET_FPR:.2f}:")
    for name in DETECTORS:
        rates = result["fpr"][name]
        print(f"    {name:<13s} [{rates.min():.3f}, {rates.max():.3f}]")


def _print_limit_comparison(results):
    print("\nHEADLINE COMPARISON :: sampled-grid detection limits (smallest sampled"
          f" delta_alpha with TPR >= {TARGET_SENSITIVITY:.2f}; '>' means right-censored)")
    print("  Pe  sigma  direction   signature  95% IC-bootstrap CI  censored"
          + "".join(f"{name:>14s}" for name in CONVENTIONAL + ("fullfield_T",))
          + "  sig <= all")
    print("  " + "-" * 128)
    for result in results:
        limits = result["limits"]
        signature = limits["signature"]
        lo, hi = signature["limit_ci"]
        no_worse = _no_worse_than_conventional(limits)
        print(f"  {result['Pe']:3d}  {result['sigma']:5.2f}  {result['direction']:>10}"
              f"  {CC.limit_text(signature['limit'], MAX_DELTA):>10}"
              f"  [{CC.limit_text(lo, MAX_DELTA):>6}, {CC.limit_text(hi, MAX_DELTA):>6}]"
              f"  {signature['censored_fraction']:8.2f}"
              + "".join(f"{CC.limit_text(limits[name]['limit'], MAX_DELTA):>14}"
                        for name in CONVENTIONAL + ("fullfield_T",))
              + f"  {'yes' if no_worse else 'no':>10}")


def _no_worse_than_conventional(limits):
    """Whether a finite signature limit is no larger than every matched arm."""
    signature = limits["signature"]["limit"]
    return signature is not None and all(
        signature <= _limit_for_comparison(limits[name]["limit"])
        for name in CONVENTIONAL + ("fullfield_T",))


def _print_noise_summary(results):
    print("\nSIGMA=0.01 CHECK :: smallest sampled detuning the signature detects")
    print("  Pe  direction   delta_a  signature" + "".join(
        f"{name:>14s}" for name in CONVENTIONAL)
        + "  all conventional arms still below target?")
    print("  " + "-" * 118)
    sigma_results = [result for result in results if np.isclose(result["sigma"], 0.01)]
    buried_count = 0
    for result in sigma_results:
        limit = result["limits"]["signature"]["limit"]
        if limit is None:
            print(f"  {result['Pe']:3d}  {result['direction']:>10}      >0.5"
                  "         --  no sampled detuning reaches the target")
            continue
        j = int(np.argmin(np.abs(result["deltas"] - limit)))
        others = [result["tpr"][name][j] for name in CONVENTIONAL]
        buried = all(value < TARGET_SENSITIVITY for value in others)
        buried_count += int(buried)
        print(f"  {result['Pe']:3d}  {result['direction']:>10}     {result['deltas'][j]:5.1f}"
              f"     {result['tpr']['signature'][j]:6.3f}"
              + "".join(f"{value:14.3f}" for value in others)
              + f"  {'yes' if buried else 'no':>10}")
    print(f"  Summary: {buried_count}/{len(sigma_results)} Pe/direction cells reach the"
          " target with the signature while every conventional arm, read out by the same"
          " classifier, stays below it at that sampled detuning.")
    return buried_count, len(sigma_results)


def _verdict(results, buried_count, sigma_cells):
    no_worse = [_no_worse_than_conventional(result["limits"]) for result in results]
    strict = sum(
        result["limits"]["signature"]["limit"] is not None
        and all(result["limits"]["signature"]["limit"]
                < _limit_for_comparison(result["limits"][name]["limit"])
                for name in CONVENTIONAL)
        for result in results)
    censored = sum(result["limits"]["signature"]["limit"] is None for result in results)
    every_fpr = np.concatenate([result["fpr"][name]
                                for result in results for name in DETECTORS])
    n_cells = len(no_worse)
    n_no_worse = sum(no_worse)
    verdict_go = n_no_worse > n_cells / 2.0
    print("\nFALSE-ALARM RATE: every detector shares one supervised split-conformal"
          f" read-out.  Each test fold's classifier is fitted on its own training ICs"
          f" only, and that single model scores both the {CC.N_CAL} calibration nominal"
          " ICs and the untouched test ICs, so the per-IC false-alarm probability is"
          f" bounded by {CC.N_CAL + 1 - CC.CAL_RANK}/{CC.N_CAL + 1} = {TARGET_FPR:.2f}"
          " exactly under exchangeability.  Training, calibration and test ICs are"
          " pairwise disjoint, and the rate above is MEASURED on the test ICs.")
    print(f"  measured over all {every_fpr.size} detector/detuning cells:"
          f" median={np.median(every_fpr):.3f}, max={every_fpr.max():.3f}"
          f" (target {TARGET_FPR:.2f}); each cell's 95% IC-cluster interval is in the"
          " CSV, and nominal alpha=1 appears only as a reference row.")
    print("\n" + "=" * 104)
    print(f"[{'GO' if verdict_go else 'CHECK'} / VERDICT] thermal detection limit of silent SUPG tau-scale changes")
    print("=" * 104)
    print(f"  Signature limit <= every matched conventional arm in {n_no_worse}/{n_cells}"
          f" Pe/sigma/direction cells (strictly smaller than all three thermal arms in"
          f" {strict}/{n_cells}); the signature limit is right-censored above"
          f" delta_alpha={MAX_DELTA:.1f} in {censored}/{n_cells} cells.")
    print(f"  At sigma=0.01, {buried_count}/{sigma_cells} cells reach the target with the"
          " signature while every conventional arm stays below it at that sampled detuning.")
    if verdict_go:
        print("  The output signature detects the numerical change before conventional thermal outputs reliably reveal it.")
    else:
        print("  The requested headline is not supported in a majority of this sweep; thermal outputs are competitive"
              " in the cells marked 'no' above.")
    return verdict_go


def main():
    start = time.perf_counter()
    if N_IC != 60:
        raise RuntimeError(f"expected the validated 60-IC thermal ensemble, got {N_IC}")

    print("=" * 104)
    print("ICHMT :: THERMAL DETECTION LIMIT FOR SILENT SUPG STABILIZATION DETUNING")
    print("Modified-equation signature versus noisy conventional heat-transfer outputs")
    print("=" * 104)
    print(f"Protocol: Pe={list(PES)} | {N_IC} ICs (seeds 1000..{1000 + N_IC - 1}) | "
          f"working mesh=60x20, seed=2026")
    print(f"          weaken={[float(alpha) for alpha in ALPHAS_WEAK]} | "
          f"strengthen={[float(alpha) for alpha in ALPHAS_STRONG]} | "
          f"sigma={list(SIGMAS)} | target TPR={TARGET_SENSITIVITY:.2f}")
    print("Observation rule: one RMS-relative noisy 64x64 grid feeds signature, thermal_pair,"
          " Twall_max, Nu_mean, and fullfield_T; grid scalars use the IC-specific heated"
          " bottom-wall extraction.")
    folds = CC.choose_folds(N_IC)
    print(f"Detection: every detector, including the one-dimensional thermal outputs, is read"
          f" out by the same supervised split-conformal classifier ({CC.REPEATS} repeats of"
          f" {folds} disjoint splits by IC: {N_IC - CC.N_CAL - N_IC // folds} training,"
          f" {CC.N_CAL} calibration and {N_IC // folds} test ICs per fold, threshold = the"
          f" maximum of the {CC.N_CAL} calibration nominal scores).  The per-IC false-alarm"
          f" probability is bounded by {TARGET_FPR:.2f} exactly under exchangeability and is"
          " MEASURED on the untouched test ICs; nominal alpha=1 is a reference row only,"
          " never compared with itself.")
    print(f"Limits: smallest SAMPLED positive delta_alpha whose point TPR reaches"
          f" {TARGET_SENSITIVITY:.2f}, right-censored above {MAX_DELTA:.1f} when none does."
          f"  Intervals: {CC.N_BOOT}-replicate IC-cluster percentile bootstrap; n={N_IC} ICs"
          " is the effective sample size (repeats and folds never enter it).")

    ics = [HC.make_thermal_ic(1000 + i) for i in range(N_IC)]
    pts, elems, tags = HC.make_channel_mesh(60, 20, seed=2026)
    csv_rows = []
    results = []

    for pe_index, pe in enumerate(PES):
        a_th = HC.thermal_diffusivity(pe)
        geom = HC.channel_mesh_geometry(pts, elems, a_th)
        print(f"\nPe={pe}: fine nominal-SUPG references (once per IC; reused across all alphas)")
        refs = HC.reference_channel_grids(ics, Pe=pe)
        print(f"  cached {len(refs)} fine reference grids")

        print(f"Pe={pe}: working SUPG solves ({len(ALL_ALPHAS)} alphas x {N_IC} ICs; clean once each)")
        clean_grids, clean_outputs = _build_clean_cache(pts, elems, tags, pe, ics, geom)
        nominal_output = clean_outputs[1.0]
        print(f"  cached clean grids and FEM outputs; nominal Twall_max=[{nominal_output['Twall_max'].min():.3f},"
              f" {nominal_output['Twall_max'].max():.3f}], Nu_mean=[{nominal_output['Nu_mean'].min():.3f},"
              f" {nominal_output['Nu_mean'].max():.3f}]")

        for sigma_index, sigma in enumerate(SIGMAS):
            print(f"Pe={pe}: observing all detector inputs at sigma={sigma:.2f}")
            observed = _observe_detectors(
                clean_grids, refs, ics, a_th, pe_index, sigma_index, sigma,
            )
            for direction_index, (direction, alphas) in enumerate(DIRECTIONS):
                result = _direction_result(observed, alphas, pe_index, sigma_index,
                                           direction_index)
                result.update({"Pe": int(pe), "sigma": float(sigma),
                               "direction": direction})
                results.append(result)
                csv_rows.extend(_csv_rows(result))

    for result in results:
        _print_curve_table(result)
    _print_limit_comparison(results)
    buried_count, sigma_cells = _print_noise_summary(results)
    _verdict(results, buried_count, sigma_cells)

    csv_path = _write_csv(csv_rows)
    elapsed = time.perf_counter() - start
    print(f"\nmetrics -> {csv_path}")
    print(f"RUNTIME: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
