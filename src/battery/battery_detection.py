import os, sys, numpy as np
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT,"src","battery")); import battery_module as B
sys.path.insert(0, os.path.join(_ROOT,"src","audit"));   import supg_2d_engineering as A
from sklearn.model_selection import GroupKFold, cross_val_predict
TAB=os.path.join(_ROOT,"results","tables"); os.makedirs(TAB, exist_ok=True)

"""Detection limits for silent SUPG-strength changes in the validated battery module.

This is a numerical/physical-plausibility analysis only.  It uses the foundation
module's representative literature-sourced properties and deterministic operating
condition population; it neither introduces experimental data nor claims an
experimental match.
"""

import csv
import time


ALPHAS = np.array([
    0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0,
    1.05, 1.1, 1.15, 1.25, 1.5, 2.0,
], dtype=float)
SIGMAS = (0.0, 0.01, 0.05)
N_IC = 48
TARGET_FPR = 0.05
TARGET_SENSITIVITY = 0.90
N_BOOTSTRAP = 300

MAIN_POINT = "2C_nominal"
POINTS = (
    (MAIN_POINT, B.Q_BY_CRATE["2C"], B.UBAR_NOMINAL, True),
    ("1C_nominal", B.Q_BY_CRATE["1C"], B.UBAR_NOMINAL, False),
    ("3C_nominal", B.Q_BY_CRATE["3C"], B.UBAR_NOMINAL, False),
    ("2C_low_flow", B.Q_BY_CRATE["2C"], 0.06, False),
    ("2C_high_flow", B.Q_BY_CRATE["2C"], 0.16, False),
)
MAIN_DETECTORS = (
    "signature", "Tmax", "dTmax", "sigma_cell", "Tout",
    "fullfield_T", "residual_norm",
)
CROSS_DETECTORS = ("signature", "Tmax", "dTmax")
CSV_FIELDS = (
    "type", "detector", "direction", "operating_point", "sigma", "alpha",
    "delta_alpha", "sensitivity", "detection_limit_delta", "ci_lo", "ci_hi",
    "frac_censored",
)


# Detector recipes requested for direct comparability with the engineering anchor.
def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    X=A.feats(np.vstack([nominal, changed])); y=np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g=np.r_[np.arange(len(nominal)), np.arange(len(changed))]; k=min(5, len(nominal))
    p=cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k), method="predict_proba")[:,1]
    thr=np.percentile(p[y==0], 100*(1-target_fpr)); return float(np.mean(p[y==1] > thr))


def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    nom=np.asarray(nom_vals,float); chg=np.asarray(chg_vals,float); c=np.median(nom)
    thr=np.percentile(np.abs(nom-c), 100*(1-target_fpr)); return float(np.mean(np.abs(chg-c) > thr))


def _add_grid_noise(Us, sigma, seed):
    """Apply one deterministic RMS-relative observation-noise realization."""
    rng=np.random.default_rng(seed)
    rms=np.sqrt(np.mean(Us**2))
    return Us + sigma*rms*rng.standard_normal(Us.shape)


def _noise_seed(point_index, sigma_index, alpha_index):
    """Stable independent stream for a point/noise/alpha observed-grid ensemble."""
    return 431_000 + 100_000 * point_index + 10_000 * sigma_index + 100 * alpha_index


def _direction_alphas(direction):
    """Detuned alphas ordered from the smallest to largest absolute change."""
    if direction == "weaken":
        values = ALPHAS[ALPHAS < 1.0]
    elif direction == "strengthen":
        values = ALPHAS[ALPHAS > 1.0]
    else:
        raise ValueError(direction)
    return np.array(sorted(values, key=lambda alpha: abs(float(alpha) - 1.0)), dtype=float)


def _detection_limit(deltas, sensitivities):
    """First 90%-sensitivity crossing, linearly interpolated in |alpha-1|."""
    deltas = np.asarray(deltas, dtype=float)
    sensitivities = np.asarray(sensitivities, dtype=float)
    order = np.argsort(deltas)
    deltas, sensitivities = deltas[order], sensitivities[order]
    hit = np.flatnonzero((deltas > 0.0) & (sensitivities >= TARGET_SENSITIVITY))
    if len(hit) == 0:
        return None
    j = int(hit[0])
    if j == 0:
        return float(deltas[j])
    x0, y0 = float(deltas[j - 1]), float(sensitivities[j - 1])
    x1, y1 = float(deltas[j]), float(sensitivities[j])
    if y1 <= y0:
        return x1
    crossing = x0 + (TARGET_SENSITIVITY - y0) * (x1 - x0) / (y1 - y0)
    return float(np.clip(crossing, x0, x1))


def _limit_text(value, direction):
    if value is None:
        return ">1.0 (strengthen)" if direction == "strengthen" else ">1.0"
    return f"{float(value):.3f}"


def _csv_float(value, digits=6):
    return "" if value is None else f"{float(value):.{digits}f}"


def _build_clean_cache(pts, elems, tags, geom, regions, ics, q_base, Ubar):
    """Solve every requested alpha/IC exactly once and retain grids plus outputs."""
    clean_grids = {}
    clean_outputs = {}
    max_energy_error = 0.0
    for alpha in ALPHAS:
        grids = np.empty((N_IC, A.GRID_OBS, A.GRID_OBS), dtype=float)
        outputs = {name: np.empty(N_IC, dtype=float)
                   for name in ("Tmax", "dTmax", "sigma_cell", "Tout")}
        for i, ic in enumerate(ics):
            T, _, meta = B.assemble_module(
                "supg", pts, elems, tags, q_base=q_base, Ubar=Ubar, ic=ic,
                alpha=float(alpha), geom=geom, return_meta=True,
            )
            out = B.thermal_outputs_module(
                pts, T, regions, tags=tags, Ubar=meta["Ubar"],
                total_generation=meta["total_generation"],
            )
            grids[i] = B.to_module_grid(pts, T, window=B.SIGNATURE_WINDOW)
            for name in outputs:
                outputs[name][i] = out[name]
            max_energy_error = max(max_energy_error, abs(float(out["energy_err"])))
        clean_grids[float(alpha)] = grids
        clean_outputs[float(alpha)] = outputs
    return clean_grids, clean_outputs, max_energy_error


def _observe_window_detectors(clean_grids, refs, sigma, point_index, sigma_index):
    """Build grid-observation detector inputs without any additional FEM solves."""
    observed = {}
    for alpha_index, alpha in enumerate(ALPHAS):
        signatures = np.empty((N_IC, len(A.LIB)), dtype=float)
        fullfield = np.empty(N_IC, dtype=float)
        residual = np.empty(N_IC, dtype=float)
        rng_seed = _noise_seed(point_index, sigma_index, alpha_index)
        for i in range(N_IC):
            noisy_grid = _add_grid_noise(clean_grids[float(alpha)][i], sigma, rng_seed + i)
            difference = noisy_grid - refs[i]
            signatures[i] = B.sig_from_grid(noisy_grid, refs[i])
            fullfield[i] = np.linalg.norm(difference)
            residual[i] = fullfield[i] / (np.linalg.norm(refs[i]) + 1.0e-30)
        observed[float(alpha)] = {
            "signature": signatures,
            "fullfield_T": fullfield,
            "residual_norm": residual,
        }
    return observed


def _detector_values(detector, observed, outputs, alpha):
    if detector in observed[float(alpha)]:
        return observed[float(alpha)][detector]
    return outputs[float(alpha)][detector]


def _direction_result(observed, outputs, direction, detectors):
    """Return curves and limits with an alpha=1 baseline anchor for interpolation."""
    alphas = _direction_alphas(direction)
    deltas = np.abs(alphas - 1.0)
    baselines, curves, limits = {}, {}, {}
    nominal_alpha = 1.0
    for detector in detectors:
        calculator = supervised_sensitivity if detector == "signature" else scalar_sensitivity
        nominal = _detector_values(detector, observed, outputs, nominal_alpha)
        baseline = calculator(nominal, nominal)
        baselines[detector] = baseline
        sensitivity = np.array([
            calculator(nominal, _detector_values(detector, observed, outputs, alpha))
            for alpha in alphas
        ], dtype=float)
        curves[detector] = sensitivity
        limits[detector] = _detection_limit(
            np.r_[0.0, deltas], np.r_[baseline, sensitivity],
        )
    return {"direction": direction, "alphas": alphas, "deltas": deltas,
            "baselines": baselines,
            "curves": curves, "limits": limits}


def _bootstrap_signature_limit(observed, direction, seed):
    """Paired-IC bootstrap of the signature limit, retaining right-censoring."""
    rng = np.random.default_rng(seed)
    alphas = _direction_alphas(direction)
    deltas = np.abs(alphas - 1.0)
    nominal = observed[1.0]["signature"]
    changed = [observed[float(alpha)]["signature"] for alpha in alphas]
    values = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.integers(0, N_IC, size=N_IC)
        base = supervised_sensitivity(nominal[draw], nominal[draw])
        sensitivity = np.array([
            supervised_sensitivity(nominal[draw], candidate[draw])
            for candidate in changed
        ], dtype=float)
        values.append(_detection_limit(np.r_[0.0, deltas], np.r_[base, sensitivity]))

    censored = sum(value is None for value in values)
    frac_censored = censored / N_BOOTSTRAP
    finite = np.array([value for value in values if value is not None], dtype=float)
    if len(finite) == 0:
        return None, None, frac_censored, True
    ci_lo = float(np.percentile(finite, 2.5))
    if frac_censored > 0.05:
        return ci_lo, None, frac_censored, True
    return ci_lo, float(np.percentile(finite, 97.5)), frac_censored, False


def _ci_text(ci_lo, ci_hi, lower_bound):
    if lower_bound:
        if ci_lo is None:
            return "right-censored lower bound >1.0"
        return f"[{ci_lo:.3f}, >1.0] lower bound"
    return f"[{ci_lo:.3f}, {ci_hi:.3f}]"


def _append_rows(rows, result, operating_point, sigma, ci_by_detector):
    """Append the requested curve and limit rows in a single stable CSV schema."""
    for detector, sensitivity in result["curves"].items():
        curve_points = [(1.0, 0.0, result["baselines"][detector])]
        curve_points.extend(zip(result["alphas"], result["deltas"], sensitivity))
        for alpha, delta, value in curve_points:
            rows.append({
                "type": "curve", "detector": detector,
                "direction": result["direction"], "operating_point": operating_point,
                "sigma": _csv_float(sigma, 2), "alpha": _csv_float(alpha),
                "delta_alpha": _csv_float(delta), "sensitivity": _csv_float(value),
                "detection_limit_delta": "", "ci_lo": "", "ci_hi": "",
                "frac_censored": "",
            })
        ci_lo, ci_hi, frac_censored, lower_bound = ci_by_detector.get(
            detector, (None, None, None, False),
        )
        if lower_bound:
            csv_ci_hi = ">1.0 (right-censored lower bound)"
        else:
            csv_ci_hi = _csv_float(ci_hi)
        rows.append({
            "type": "limit", "detector": detector,
            "direction": result["direction"], "operating_point": operating_point,
            "sigma": _csv_float(sigma, 2), "alpha": "", "delta_alpha": "",
            "sensitivity": "",
            "detection_limit_delta": _limit_text(
                result["limits"][detector], result["direction"],
            ),
            "ci_lo": _csv_float(ci_lo), "ci_hi": csv_ci_hi,
            "frac_censored": _csv_float(frac_censored),
        })


def _write_csv(rows):
    path = os.path.join(TAB, "battery_detection.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _limit_for_compare(value):
    return np.inf if value is None else float(value)


def _main_limit_table(results):
    print("\nRESULT TABLE :: 2C nominal detection limits at 90% sensitivity")
    print("  sigma  direction  detector       limit |alpha-1|       signature 95% CI       censored")
    print("  " + "-" * 92)
    for result in results:
        for detector in MAIN_DETECTORS:
            ci_lo, ci_hi, fraction, lower_bound = result["ci"].get(
                detector, (None, None, None, False),
            )
            ci_text = _ci_text(ci_lo, ci_hi, lower_bound) if detector == "signature" else "--"
            censored = "--" if fraction is None else f"{100.0 * fraction:.1f}%"
            print(f"  {result['sigma']:5.2f}  {result['direction']:>10}  {detector:<14} "
                  f"{_limit_text(result['limits'][detector], result['direction']):>18} "
                  f"{ci_text:>24} {censored:>10}")


def _cross_limit_table(results):
    print("\nCROSS-OPERATING-POINT SUMMARY :: sigma=0.01, 90% sensitivity limit")
    print("  operating point  direction        signature        Tmax       dTmax")
    print("  " + "-" * 72)
    for result in results:
        limits = result["limits"]
        print(f"  {result['operating_point']:<16} {result['direction']:>10} "
              f"{_limit_text(limits['signature'], result['direction']):>16} "
              f"{_limit_text(limits['Tmax'], result['direction']):>11} "
              f"{_limit_text(limits['dTmax'], result['direction']):>11}")


def _verdict(main_results, cross_results):
    """Score the requested majority rule, using only baselines run at each point."""
    comparison_cells = []
    competitive = []
    for result in main_results:
        baselines = tuple(name for name in MAIN_DETECTORS if name != "signature")
        signature_limit = result["limits"]["signature"]
        sig = _limit_for_compare(signature_limit)
        baseline_limits = {name: _limit_for_compare(result["limits"][name]) for name in baselines}
        # A shared right-censoring bound (">1") is not evidence of an ordering.
        no_worse = signature_limit is not None and all(
            sig <= value + 1.0e-12 for value in baseline_limits.values()
        )
        comparison_cells.append(no_worse)
        tied_or_better = ([name for name, value in baseline_limits.items()
                            if value <= sig + 1.0e-12]
                           if signature_limit is not None else
                           [name for name in baselines if result["limits"][name] is not None])
        if tied_or_better:
            competitive.append((result["operating_point"], result["sigma"],
                                result["direction"], ",".join(tied_or_better)))
    for result in cross_results:
        baselines = ("Tmax", "dTmax")
        signature_limit = result["limits"]["signature"]
        sig = _limit_for_compare(signature_limit)
        baseline_limits = {name: _limit_for_compare(result["limits"][name]) for name in baselines}
        no_worse = signature_limit is not None and all(
            sig <= value + 1.0e-12 for value in baseline_limits.values()
        )
        comparison_cells.append(no_worse)
        tied_or_better = ([name for name, value in baseline_limits.items()
                            if value <= sig + 1.0e-12]
                           if signature_limit is not None else
                           [name for name in baselines if result["limits"][name] is not None])
        if tied_or_better:
            competitive.append((result["operating_point"], result["sigma"],
                                result["direction"], ",".join(tied_or_better)))
    wins = int(sum(comparison_cells))
    total = len(comparison_cells)
    return wins, total, wins > total / 2.0, competitive


def main():
    started = time.perf_counter()
    if not np.array_equal(ALPHAS, np.array([
        0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0,
        1.05, 1.1, 1.15, 1.25, 1.5, 2.0,
    ])):
        raise RuntimeError("the prescribed alpha sweep was altered")

    print("=" * 100)
    print("JOURNAL OF ENERGY STORAGE :: BATTERY THERMAL DETECTION OF SILENT SUPG DETUNING")
    print("Modified-equation signature versus conventional conjugate battery thermal indicators")
    print("=" * 100)
    print("Numerical/physical-plausibility analysis only: representative literature-sourced module "
          "properties; no fabricated experimental data and no experimental-match claim.")
    print(f"Protocol: {N_IC} deterministic ICs (seeds 1000..{1000 + N_IC - 1}); "
          f"working mesh n_cell_x=16, seed=2026; alphas={ALPHAS.tolist()}")
    print(f"Noise: sigma={list(SIGMAS)} RMS-relative grid noise; target FPR={TARGET_FPR:.2f}; "
          f"target sensitivity={TARGET_SENSITIVITY:.2f}; signature bootstrap B={N_BOOTSTRAP}")
    print("IC population varies source scale, five cell-wise source factors, coolant flow, and inlet ramp; "
          "the monolithic solid-fluid solve and energy closure are retained in every case.")

    ics = [B.make_module_ic(1000 + i) for i in range(N_IC)]
    pts, elems, tags = B.make_module_mesh(n_cell_x=16, seed=2026)
    geom = B.module_mesh_geometry(pts, elems)
    regions = B.material_regions(pts, elems)
    print(f"Working mesh: {len(pts)} nodes, {len(elems)} triangles; signature grid={A.GRID_OBS}x{A.GRID_OBS}.")

    csv_rows = []
    main_results = []
    cross_results = []
    for point_index, (point_name, q_base, Ubar, is_main) in enumerate(POINTS):
        detectors = MAIN_DETECTORS if is_main else CROSS_DETECTORS
        sigmas = SIGMAS if is_main else (0.01,)
        print(f"\n{point_name}: fine nominal-SUPG references once per IC, then "
              f"{len(ALPHAS)} clean working solves per IC")
        refs = B.reference_module_grids(ics, q_base=q_base, Ubar=Ubar)
        clean_grids, clean_outputs, max_energy_error = _build_clean_cache(
            pts, elems, tags, geom, regions, ics, q_base, Ubar,
        )
        print(f"  cached {len(refs)} references and {len(ALPHAS) * N_IC} working solves; "
              f"max working energy error={max_energy_error:.3e}")

        for sigma_index, sigma in enumerate(sigmas):
            observed = _observe_window_detectors(
                clean_grids, refs, sigma, point_index, sigma_index,
            )
            for direction_index, direction in enumerate(("weaken", "strengthen")):
                result = _direction_result(observed, clean_outputs, direction, detectors)
                result.update({"operating_point": point_name, "sigma": sigma})
                ci_by_detector = {}
                if is_main:
                    ci_by_detector["signature"] = _bootstrap_signature_limit(
                        observed, direction,
                        seed=892_000 + 10_000 * sigma_index + 1_000 * direction_index,
                    )
                result["ci"] = ci_by_detector
                _append_rows(csv_rows, result, point_name, sigma, ci_by_detector)
                if is_main:
                    main_results.append(result)
                else:
                    cross_results.append(result)

    _main_limit_table(main_results)
    _cross_limit_table(cross_results)
    print("\nCHANCE/FPR: binary held-out signature chance=0.500; every decision threshold is "
          "calibrated at FPR=0.050 (specificity=0.950) from nominal scores or nominal two-sided tails.")

    sigma01 = [result for result in main_results if np.isclose(result["sigma"], 0.01)]
    signature_limits = [(result["limits"]["signature"], result["direction"])
                        for result in sigma01]
    finite_signature_limits = [(value, direction) for value, direction in signature_limits
                               if value is not None]
    if finite_signature_limits:
        smallest_limit, smallest_direction = min(finite_signature_limits, key=lambda pair: pair[0])
        smallest_text = f"{smallest_limit:.3f} ({smallest_direction})"
    else:
        smallest_text = ">1.0 in both directions"
    print(f"Smallest signature-resolved |alpha-1| at 2C nominal, sigma=0.01: {smallest_text}.")

    wins, total, verdict_go, competitive = _verdict(main_results, cross_results)
    if competitive:
        detail = "; ".join(
            f"{point}/sigma={sigma:.2f}/{direction}: {names}"
            for point, sigma, direction, names in competitive
        )
        print("Thermal output competitive (equal or lower detection limit) in: " + detail)
    else:
        print("Thermal output competitive (equal or lower detection limit): none.")
    print(f"Majority accounting: signature is no worse than every available thermal baseline in "
          f"{wins}/{total} operating-point/sigma/direction cells (all six baselines at 2C nominal; "
          "Tmax and dTmax only at the four lighter cross-point checks).")

    csv_path = _write_csv(csv_rows)
    elapsed = time.perf_counter() - started
    print(f"CSV -> {csv_path}")
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    final_tag = "GO" if verdict_go else "NO-GO"
    print(f"[{final_tag} / VERDICT] The signature meets the requested majority criterion in "
          f"{wins}/{total} cells; the resolved 2C, sigma=0.01 minimum is {smallest_text}.")
    return {"rows": csv_rows, "main": main_results, "cross": cross_results,
            "csv": csv_path, "runtime_seconds": elapsed, "go": verdict_go}


if __name__ == "__main__":
    main()
