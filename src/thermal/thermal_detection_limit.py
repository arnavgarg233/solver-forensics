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


# The comparison deliberately varies only the SUPG tau scale.  The directions
# are reported independently because an under- and over-stabilized method need
# not leave the same thermal fingerprint.
PES = (50, 100, 200)
ALPHAS_WEAK = np.array((1.0, 0.9, 0.8, 0.7, 0.6, 0.5), dtype=float)
ALPHAS_STRONG = np.array((1.0, 1.1, 1.2, 1.3, 1.4, 1.5), dtype=float)
DIRECTIONS = (("weaken", ALPHAS_WEAK), ("strengthen", ALPHAS_STRONG))
ALL_ALPHAS = np.array(sorted(set(ALPHAS_WEAK) | set(ALPHAS_STRONG)), dtype=float)
SIGMAS = (0.0, 0.01, 0.05)
TARGET_FPR = 0.05
TARGET_SENSITIVITY = 0.95
N_IC = HC.N_IC

DETECTORS = ("signature", "Twall_max", "Nu_mean", "fullfield_T")
CSV_FIELDS = (
    "type", "detector", "direction", "Pe", "sigma", "alpha",
    "delta_alpha", "sensitivity", "detection_limit_delta_alpha",
)


# signature detector: held-out classifier P(changed), threshold on nominal
# scores at 5% FPR, sensitivity=TPR
def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    X = A.feats(np.vstack([nominal, changed]))
    y = np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(changed))]
    k = min(5, len(nominal))
    proba = cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k),
                              method="predict_proba")[:, 1]
    thr = np.percentile(proba[y == 0], 100 * (1 - target_fpr))
    return float(np.mean(proba[y == 1] > thr))


# scalar detector for a 1-D thermal output v: two-sided |v - median(nominal)|,
# LOO-free threshold at 5% FPR
def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    nom = np.asarray(nom_vals, float)
    chg = np.asarray(chg_vals, float)
    center = np.median(nom)
    thr = np.percentile(np.abs(nom - center), 100 * (1 - target_fpr))
    return float(np.mean(np.abs(chg - center) > thr))


def detection_limit(deltas, sensitivities):
    """First 95%-sensitivity crossing, linearly interpolated in delta_alpha."""
    deltas = np.asarray(deltas, dtype=float)
    sensitivities = np.asarray(sensitivities, dtype=float)
    hit = np.flatnonzero((deltas > 0.0) & (sensitivities >= TARGET_SENSITIVITY))
    if len(hit) == 0:
        return None
    j = int(hit[0])
    if j == 0:
        return float(deltas[0])
    x0, y0 = float(deltas[j - 1]), float(sensitivities[j - 1])
    x1, y1 = float(deltas[j]), float(sensitivities[j])
    if y1 <= y0:
        return x1
    return float(x0 + (TARGET_SENSITIVITY - y0) * (x1 - x0) / (y1 - y0))


def _limit_text(value, direction):
    if value is None:
        return ">0.5 (strengthen)" if direction == "strengthen" else ">0.5"
    return f"{value:.3f}"


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
    thermal scalars, and the naive full-field reference distance.  Thus their
    performance differences cannot come from unequal noise realizations.
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
            "Twall_max": twall,
            "Nu_mean": nu,
            "fullfield_T": fullfield,
        }
    return observed


def _direction_result(observed, alphas):
    """Calculate all detector curves and interpolated limits for one direction."""
    deltas = np.abs(alphas - 1.0)
    nominal = observed[1.0]
    sensitivity = {}
    limits = {}
    for detector in DETECTORS:
        calculator = supervised_sensitivity if detector == "signature" else scalar_sensitivity
        values = np.array(
            [calculator(nominal[detector], observed[float(alpha)][detector])
             for alpha in alphas],
            dtype=float,
        )
        sensitivity[detector] = values
        limits[detector] = detection_limit(deltas, values)
    return deltas, sensitivity, limits


def _write_csv(rows):
    path = os.path.join(TAB, "thermal_detection_limit.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _limit_for_comparison(value):
    return np.inf if value is None else float(value)


def _strictly_smaller(signature_limit, baseline_limit):
    """Whether a finite signature crossing occurs before a baseline crossing."""
    return signature_limit is not None and (
        baseline_limit is None or signature_limit < baseline_limit - 1.0e-12
    )


def _print_curve_table(result):
    print(f"\nRESULT TABLE :: Pe={result['Pe']} | {result['direction']} SUPG detuning")
    print("  sigma  alpha  delta_a  signature  Twall_max  Nu_mean  fullfield_T")
    print("  " + "-" * 70)
    for sigma, alpha, delta, signature, twall, nu, fullfield in zip(
        np.repeat(result["sigma"], len(result["alphas"])), result["alphas"], result["deltas"],
        result["sensitivity"]["signature"], result["sensitivity"]["Twall_max"],
        result["sensitivity"]["Nu_mean"], result["sensitivity"]["fullfield_T"],
    ):
        print(f"  {sigma:5.2f}  {alpha:5.1f}   {delta:5.1f}     {signature:6.3f}"
              f"     {twall:6.3f}   {nu:6.3f}      {fullfield:6.3f}")


def _print_limit_comparison(results):
    print("\nHEADLINE COMPARISON :: 95% sensitivity detection limits (delta_alpha)")
    print("  Pe  sigma  direction      signature  Twall_max  sig<Tw  Nu_mean  sig<Nu  fullfield_T  sig<field  sig <= both")
    print("  " + "-" * 116)
    for result in results:
        limit = result["limits"]
        sig = _limit_for_comparison(limit["signature"])
        better_both = (
            limit["signature"] is not None
            and sig <= _limit_for_comparison(limit["Twall_max"])
            and sig <= _limit_for_comparison(limit["Nu_mean"])
        )
        print(f"  {result['Pe']:3d}  {result['sigma']:5.2f}  {result['direction']:>10}"
              f"  {_limit_text(limit['signature'], result['direction']):>17}"
              f"  {_limit_text(limit['Twall_max'], result['direction']):>9}"
              f"  {'yes' if _strictly_smaller(limit['signature'], limit['Twall_max']) else 'no':>6}"
              f"  {_limit_text(limit['Nu_mean'], result['direction']):>7}"
              f"  {'yes' if _strictly_smaller(limit['signature'], limit['Nu_mean']) else 'no':>6}"
              f"  {_limit_text(limit['fullfield_T'], result['direction']):>11}"
              f"  {'yes' if _strictly_smaller(limit['signature'], limit['fullfield_T']) else 'no':>9}"
              f"  {'yes' if better_both else 'no':>11}")


def _print_noise_summary(results):
    print("\nSIGMA=0.01 CHECK :: first sampled signature detection point")
    print("  Pe  direction   delta_a  signature  Twall_max  Nu_mean  both conventional outputs still <95%?")
    print("  " + "-" * 94)
    sigma_results = [result for result in results if np.isclose(result["sigma"], 0.01)]
    buried_count = 0
    for result in sigma_results:
        signature = result["sensitivity"]["signature"]
        hit = np.flatnonzero((result["deltas"] > 0.0) & (signature >= TARGET_SENSITIVITY))
        if len(hit) == 0:
            print(f"  {result['Pe']:3d}  {result['direction']:>10}       >0.5       --        --       --  no signature 95% crossing")
            continue
        j = int(hit[0])
        twall = result["sensitivity"]["Twall_max"][j]
        nu = result["sensitivity"]["Nu_mean"][j]
        buried = twall < TARGET_SENSITIVITY and nu < TARGET_SENSITIVITY
        buried_count += int(buried)
        print(f"  {result['Pe']:3d}  {result['direction']:>10}     {result['deltas'][j]:5.1f}"
              f"     {signature[j]:6.3f}     {twall:6.3f}   {nu:6.3f}"
              f"              {'yes' if buried else 'no'}")
    print(f"  Summary: {buried_count}/{len(sigma_results)} Pe/direction cells have a 95%-sensitive"
          " signature while both noisy conventional outputs remain below 95% at that sampled detuning.")
    return buried_count, len(sigma_results)


def _verdict(results, buried_count, sigma_cells):
    conventional_cells = []
    strict_both = 0
    fullfield_no_worse = 0
    for result in results:
        limit = result["limits"]
        sig = _limit_for_comparison(limit["signature"])
        twall = _limit_for_comparison(limit["Twall_max"])
        nu = _limit_for_comparison(limit["Nu_mean"])
        field = _limit_for_comparison(limit["fullfield_T"])
        is_no_worse = limit["signature"] is not None and sig <= twall and sig <= nu
        conventional_cells.append(is_no_worse)
        strict_both += int(limit["signature"] is not None and sig < twall and sig < nu)
        fullfield_no_worse += int(limit["signature"] is not None and sig <= field)

    n_cells = len(conventional_cells)
    n_no_worse = sum(conventional_cells)
    verdict_go = n_no_worse > n_cells / 2.0
    print("\nCHANCE / FALSE-POSITIVE RATE: all detector thresholds are calibrated to a 5.0%"
          " nominal false-positive target.  At alpha=1, strict '>' thresholding may report below"
          " 5.0% when held-out scores are tied; it never grants a detector an uncalibrated threshold.")
    print("\n" + "=" * 104)
    print(f"[{'GO' if verdict_go else 'CHECK'} / VERDICT] thermal detection limit of silent SUPG tau-scale changes")
    print("=" * 104)
    print(f"  Signature limit <= BOTH Twall_max and Nu_mean limits in {n_no_worse}/{n_cells} Pe/sigma/direction cells"
          f" (strictly smaller than both in {strict_both}/{n_cells});"
          f" signature limit <= naive full-field distance in {fullfield_no_worse}/{n_cells} cells.")
    print(f"  At sigma=0.01, {buried_count}/{sigma_cells} cells reach 95% signature sensitivity while both"
          " conventional noisy-grid outputs are still below 95% at the first signature-detected detuning.")
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
          f"sigma={list(SIGMAS)} | target FPR={TARGET_FPR:.2f}, target TPR={TARGET_SENSITIVITY:.2f}")
    print("Observation rule: one RMS-relative noisy 64x64 grid feeds signature, Twall_max, Nu_mean,"
          " and fullfield_T; grid scalars use the IC-specific heated bottom-wall extraction.")

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
            for direction, alphas in DIRECTIONS:
                deltas, sensitivity, limits = _direction_result(observed, alphas)
                result = {
                    "Pe": int(pe), "sigma": float(sigma), "direction": direction,
                    "alphas": alphas, "deltas": deltas, "sensitivity": sensitivity,
                    "limits": limits,
                }
                results.append(result)
                for detector in DETECTORS:
                    for alpha, delta, value in zip(alphas, deltas, sensitivity[detector]):
                        csv_rows.append({
                            "type": "curve", "detector": detector, "direction": direction,
                            "Pe": int(pe), "sigma": f"{sigma:.2f}", "alpha": f"{alpha:.1f}",
                            "delta_alpha": f"{delta:.1f}", "sensitivity": f"{value:.6f}",
                        })
                    csv_rows.append({
                        "type": "limit", "detector": detector, "direction": direction,
                        "Pe": int(pe), "sigma": f"{sigma:.2f}",
                        "detection_limit_delta_alpha": _limit_text(limits[detector], direction),
                    })

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
