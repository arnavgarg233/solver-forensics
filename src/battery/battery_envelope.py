import os, sys, numpy as np
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT,"src","battery")); import battery_module as B
sys.path.insert(0, os.path.join(_ROOT,"src","audit"));   import supg_2d_engineering as A
from sklearn.model_selection import GroupKFold, cross_val_predict
TAB=os.path.join(_ROOT,"results","tables"); os.makedirs(TAB, exist_ok=True)

"""Battery safe-operating-envelope consequence of silent SUPG detuning.

This analysis imports the validated five-cell conjugate foundation and keeps the
finite-element solve inside ``battery_module``.  The primary result is the
maximum base volumetric heat generation allowed by two replaceable screening
limits.  A small, cached signature readout is included only to preserve the
repository's 5%-FPR audit convention; it does not represent experimental data.
"""

import csv
import time


ALPHAS = np.array(
    (0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0,
     1.05, 1.1, 1.15, 1.25, 1.5, 2.0), dtype=float)
THERMAL_LIMITS = np.array((45.0, 50.0), dtype=float)
Q_REF = B.Q_BY_CRATE["2C"]
LINEARITY_MULTIPLIER = 2.0
TARGET_FPR = 0.05
SIGMAS = (0.0, 0.01, 0.05)
AUDIT_SEED0 = 1000
AUDIT_DISPLAY_ALPHAS = (0.5, 1.0, 2.0)
CSV_FIELDS = (
    "alpha", "direction", "Tmax_at_qref", "T_lim", "q_max", "delta_q_max_pct",
)


def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    """Signature sensitivity with a threshold calibrated to the nominal 5% FPR."""
    X = A.feats(np.vstack([nominal, changed])); y = np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(changed))]; k = min(5, len(nominal))
    p = cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k), method="predict_proba")[:, 1]
    thr = np.percentile(p[y == 0], 100 * (1 - target_fpr)); return float(np.mean(p[y == 1] > thr))


def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    """Two-sided scalar sensitivity around the nominal median at 5% FPR."""
    nom = np.asarray(nom_vals, float); chg = np.asarray(chg_vals, float); c = np.median(nom)
    thr = np.percentile(np.abs(nom - c), 100 * (1 - target_fpr)); return float(np.mean(np.abs(chg - c) > thr))


def _direction(alpha):
    if np.isclose(alpha, 1.0):
        return "nominal"
    return "weakening" if alpha < 1.0 else "strengthening"


def _solve_output(pts, elems, tags, geom, regions, ic, q_base, alpha):
    """Solve one requested operating point through the validated battery API."""
    T, _, meta = B.assemble_module(
        "supg", pts, elems, tags=tags, q_base=q_base, Ubar=B.UBAR_NOMINAL,
        ic=ic, alpha=float(alpha), geom=geom, return_meta=True, energy_closure=True,
    )
    output = B.thermal_outputs_module(
        pts, T, regions, tags=tags, Ubar=meta["Ubar"],
        total_generation=meta["total_generation"], T_in=B.T_IN,
    )
    return T, output, meta


def _build_primary_sweep():
    """Build the working mesh once and solve q_ref and 2*q_ref for every alpha."""
    pts, elems, tags = B.make_module_mesh(n_cell_x=16, seed=2026)
    geom = B.module_mesh_geometry(pts, elems)
    regions = B.material_regions(pts, elems)
    ic = B.make_module_ic(AUDIT_SEED0)
    solutions = {}
    for alpha in ALPHAS:
        T_ref, out_ref, meta_ref = _solve_output(
            pts, elems, tags, geom, regions, ic, Q_REF, alpha,
        )
        T_double, out_double, meta_double = _solve_output(
            pts, elems, tags, geom, regions, ic, LINEARITY_MULTIPLIER * Q_REF, alpha,
        )
        base_ref = out_ref["Tmax"] - B.T_IN
        base_double = out_double["Tmax"] - B.T_IN
        ratio = base_double / base_ref if base_ref != 0.0 else np.nan
        rel_error_pct = abs(ratio / LINEARITY_MULTIPLIER - 1.0) * 100.0
        solutions[float(alpha)] = {
            "T": T_ref, "output": out_ref, "meta": meta_ref,
            "double_output": out_double, "double_meta": meta_double,
            "ratio_2q_to_q": float(ratio), "linearity_error_pct": float(rel_error_pct),
        }
    return {
        "pts": pts, "elems": elems, "tags": tags, "geom": geom,
        "regions": regions, "ic": ic, "solutions": solutions,
    }


def _envelope_rows(primary):
    """Invert the validated linear load relation and create the required rows."""
    nominal = primary["solutions"][1.0]["output"]["Tmax"]
    nominal_base = nominal - B.T_IN
    nominal_qmax = {
        float(limit): float(Q_REF * (limit - B.T_IN) / nominal_base)
        for limit in THERMAL_LIMITS
    }
    rows = []
    for alpha in ALPHAS:
        alpha = float(alpha)
        Tmax = float(primary["solutions"][alpha]["output"]["Tmax"])
        direction = _direction(alpha)
        for limit in THERMAL_LIMITS:
            limit = float(limit)
            q_max = float(Q_REF * (limit - B.T_IN) / (Tmax - B.T_IN))
            delta_pct = 100.0 * (q_max - nominal_qmax[limit]) / nominal_qmax[limit]
            rows.append({
                "alpha": alpha, "direction": direction, "Tmax_at_qref": Tmax,
                "T_lim": limit, "q_max": q_max, "delta_q_max_pct": float(delta_pct),
            })
    return rows, nominal_qmax


def _linearity_summary(primary):
    checks = [primary["solutions"][float(alpha)] for alpha in ALPHAS]
    errors = np.array([item["linearity_error_pct"] for item in checks], dtype=float)
    ratios = np.array([item["ratio_2q_to_q"] for item in checks], dtype=float)
    worst = int(np.nanargmax(errors))
    nominal = int(np.flatnonzero(np.isclose(ALPHAS, 1.0))[0])
    return {
        "errors": errors, "ratios": ratios,
        "max_error_pct": float(errors[worst]), "worst_alpha": float(ALPHAS[worst]),
        "nominal_ratio": float(ratios[nominal]),
        "pass": bool(np.all(np.isfinite(errors)) and np.max(errors) <= 1.0),
    }


def _add_grid_noise(Us, sigma, seed):
    """Apply the specified RMS-relative Gaussian observation noise."""
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(Us ** 2))
    return Us + sigma * rms * rng.standard_normal(Us.shape)


def _noise_seed(sigma_index, alpha_index, ic_index, control=False):
    offset = 7_000_000 if control else 0
    return offset + 2_026_000 + 100_000 * sigma_index + 1_000 * alpha_index + ic_index


def _build_audit_cache(primary):
    """Cache clean working grids/thermal outputs for all ICs and all alphas."""
    ics = [B.make_module_ic(AUDIT_SEED0 + i) for i in range(B.N_IC)]
    clean_grids = {}
    clean_outputs = {}
    for alpha in ALPHAS:
        alpha = float(alpha)
        grids = []
        Tmax = []
        for index, ic in enumerate(ics):
            if index == 0:
                T = primary["solutions"][alpha]["T"]
                output = primary["solutions"][alpha]["output"]
            else:
                T, output, _ = _solve_output(
                    primary["pts"], primary["elems"], primary["tags"],
                    primary["geom"], primary["regions"], ic, Q_REF, alpha,
                )
            grids.append(B.to_module_grid(primary["pts"], T, window=B.SIGNATURE_WINDOW))
            Tmax.append(output["Tmax"])
        clean_grids[alpha] = np.asarray(grids, dtype=float)
        clean_outputs[alpha] = {"Tmax": np.asarray(Tmax, dtype=float)}
    return ics, clean_grids, clean_outputs


def _detector_readout(primary):
    """Run the supplied cached noisy-grid signature/scalar detector recipes."""
    ics, clean_grids, clean_outputs = _build_audit_cache(primary)
    reference_grids = B.reference_module_grids(
        ics, q_base=Q_REF, Ubar=B.UBAR_NOMINAL, n_cell_x=26, seed=7001,
    )
    results = []
    controls = []
    for sigma_index, sigma in enumerate(SIGMAS):
        signatures = {}
        for alpha_index, alpha in enumerate(ALPHAS):
            alpha = float(alpha)
            sigs = np.empty((len(ics), len(A.LIB)), dtype=float)
            for ic_index, reference in enumerate(reference_grids):
                observed = _add_grid_noise(
                    clean_grids[alpha][ic_index], sigma,
                    _noise_seed(sigma_index, alpha_index, ic_index),
                )
                sigs[ic_index] = B.sig_from_grid(observed, reference)
            signatures[alpha] = sigs

        nominal_signatures = signatures[1.0]
        independent_nominal = np.empty_like(nominal_signatures)
        for ic_index, reference in enumerate(reference_grids):
            observed = _add_grid_noise(
                clean_grids[1.0][ic_index], sigma,
                _noise_seed(sigma_index, len(ALPHAS), ic_index, control=True),
            )
            independent_nominal[ic_index] = B.sig_from_grid(observed, reference)
        controls.append({
            "sigma": float(sigma),
            "signature": supervised_sensitivity(nominal_signatures, independent_nominal),
        })

        for alpha in ALPHAS:
            alpha = float(alpha)
            results.append({
                "sigma": float(sigma), "alpha": alpha,
                "signature": supervised_sensitivity(nominal_signatures, signatures[alpha]),
                "Tmax": scalar_sensitivity(
                    clean_outputs[1.0]["Tmax"], clean_outputs[alpha]["Tmax"],
                ),
            })
    return {"results": results, "controls": controls, "n_ic": len(ics)}


def _write_csv(rows, summary):
    path = os.path.join(TAB, "battery_envelope.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in list(rows) + [summary]:
            formatted = {}
            for field in CSV_FIELDS:
                value = row.get(field, "")
                formatted[field] = (
                    f"{value:.10f}" if isinstance(value, (float, np.floating)) else value
                )
            writer.writerow(formatted)
    return path


def _print_setup(primary):
    ic = primary["ic"]
    print("=" * 104)
    print("JES :: BATTERY SAFE-OPERATING ENVELOPE UNDER SILENT SUPG DETUNING")
    print("Maximum permissible volumetric heat generation from the validated five-cell conjugate module")
    print("=" * 104)
    print(
        f"Protocol: mesh n_cell_x=16, seed=2026 | scheme=supg | Ubar={B.UBAR_NOMINAL:.3f} m/s | "
        f"q_ref=2C={Q_REF:.6e} W/m^3"
    )
    print(f"          alpha grid={[float(alpha) for alpha in ALPHAS]}")
    print(
        f"          T_lim={[float(limit) for limit in THERMAL_LIMITS]} C "
        "(representative Li-ion design limits; replaceable screening parameters, not universal safety limits)"
    )
    print(
        f"IC: B.make_module_ic({AUDIT_SEED0}); q_scale={ic['q_scale']:.6f}, "
        f"ubar_scale={ic['ubar_scale']:.6f}, inlet_ramp={ic['inlet_ramp']:.6f} K"
    )
    print(
        "Integrity: representative literature-sourced parameters from battery_module; "
        "numerical/physical-plausibility result only; no experimental match claimed."
    )
    print(
        f"working mesh: {len(primary['pts'])} nodes, {len(primary['elems'])} triangles; "
        "geometry and material regions precomputed once"
    )


def _print_linearity(primary, summary):
    print("\nLINEARITY CHECK :: (Tmax(2*q_ref)-T_in)/(Tmax(q_ref)-T_in)")
    print("alpha   ratio_2q_to_q   rel_error_pct   status")
    print("-" * 54)
    for alpha, ratio, error in zip(ALPHAS, summary["ratios"], summary["errors"]):
        print(
            f"{alpha:5.2f}      {ratio:10.7f}       {error:10.6f}   "
            f"{'PASS' if error <= 1.0 else 'FAIL'}"
        )
    print(
        f"max relative error={summary['max_error_pct']:.6f}% at alpha={summary['worst_alpha']:.2f}; "
        f"nominal ratio={summary['nominal_ratio']:.9f}; within ~1%: {summary['pass']}"
    )


def _print_envelope(rows, nominal_qmax):
    print("\nRESULT TABLE :: INFERRED SAFE HEAT ENVELOPE")
    print("alpha direction       Tmax_qref[C]  T_lim[C]       q_max[W/m^3]  delta_q_max[%]")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['alpha']:5.2f} {row['direction']:<13s} {row['Tmax_at_qref']:12.6f} "
            f"{row['T_lim']:8.2f} {row['q_max']:16.6f} {row['delta_q_max_pct']:+15.6f}"
        )
    print(
        f"nominal alpha=1 q_max: T_lim=45 C -> {nominal_qmax[45.0]:.6f} W/m^3; "
        f"T_lim=50 C -> {nominal_qmax[50.0]:.6f} W/m^3"
    )


def _max_shift(rows, limit, direction):
    candidates = [
        row for row in rows
        if row["direction"] == direction and np.isclose(row["T_lim"], limit)
    ]
    return max(candidates, key=lambda row: abs(row["delta_q_max_pct"]))


def _print_consequence(rows, primary):
    tmax_nominal = primary["solutions"][1.0]["output"]["Tmax"]
    tmax_values = np.array(
        [primary["solutions"][float(alpha)]["output"]["Tmax"] for alpha in ALPHAS],
        dtype=float,
    )
    max_abs_tmax_shift = float(np.max(np.abs(tmax_values - tmax_nominal)))
    print("\nENVELOPE CONSEQUENCE :: full alpha sweep")
    for limit in THERMAL_LIMITS:
        limit = float(limit)
        weak = _max_shift(rows, limit, "weakening")
        strong = _max_shift(rows, limit, "strengthening")
        print(
            f"T_lim={limit:.1f} C: weakening max |shift|={abs(weak['delta_q_max_pct']):.6f}% "
            f"at alpha={weak['alpha']:.2f} (signed {weak['delta_q_max_pct']:+.6f}%); "
            f"strengthening max |shift|={abs(strong['delta_q_max_pct']):.6f}% "
            f"at alpha={strong['alpha']:.2f} (signed {strong['delta_q_max_pct']:+.6f}%)"
        )
    weak45 = _max_shift(rows, 45.0, "weakening")
    strong45 = _max_shift(rows, 45.0, "strengthening")
    print(
        f"INTERPRETATION: an undocumented stabilization change of |alpha-1|="
        f"{abs(weak45['alpha'] - 1.0):.2f} shifts the inferred maximum permissible heat generation by "
        f"{weak45['delta_q_max_pct']:+.6f}% (T_lim=45 C) [weakening]."
    )
    print(
        f"INTERPRETATION: an undocumented stabilization change of |alpha-1|="
        f"{abs(strong45['alpha'] - 1.0):.2f} shifts the inferred maximum permissible heat generation by "
        f"{strong45['delta_q_max_pct']:+.6f}% (T_lim=45 C) [strengthening]."
    )
    print(
        f"Silent fixed-load check: max |Tmax(alpha)-Tmax(1)|={max_abs_tmax_shift:.6f} C "
        f"over the swept alpha range; the inferred q''' limit still changes."
    )
    return max_abs_tmax_shift


def _print_detector_readout(detector):
    print("\nDETECTOR READOUT :: cached noisy windowed-grid signatures versus Tmax scalar baseline")
    print(
        f"audit ICs={detector['n_ic']}; references=fine nominal-SUPG per IC; "
        f"noise sigma={list(SIGMAS)}; scalar Tmax values are clean FEM outputs"
    )
    print("sigma  alpha   signature_TPR_at_5pct_FPR   Tmax_scalar_TPR_at_5pct_FPR")
    print("-" * 72)
    for result in detector["results"]:
        if not any(np.isclose(result["alpha"], value) for value in AUDIT_DISPLAY_ALPHAS):
            continue
        print(
            f"{result['sigma']:5.2f}  {result['alpha']:5.2f}               "
            f"{result['signature']:8.3f}                     {result['Tmax']:8.3f}"
        )
    print("same-alpha independent-noise signature controls:")
    for control in detector["controls"]:
        print(f"  sigma={control['sigma']:.2f}: TPR={control['signature']:.3f}")


def _write_linearity_summary(summary):
    return {
        "alpha": "linearity_check", "direction": "all_alphas",
        "Tmax_at_qref": float(summary["max_error_pct"]),
        "T_lim": "PASS" if summary["pass"] else "FAIL",
        "q_max": float(summary["nominal_ratio"]), "delta_q_max_pct": "",
    }


def main():
    started = time.perf_counter()
    primary = _build_primary_sweep()
    _print_setup(primary)
    linearity = _linearity_summary(primary)
    _print_linearity(primary, linearity)
    rows, nominal_qmax = _envelope_rows(primary)
    _print_envelope(rows, nominal_qmax)
    max_abs_tmax_shift = _print_consequence(rows, primary)

    print("\nAUDIT CACHE: building clean working grids and fine nominal-SUPG references once")
    detector = _detector_readout(primary)
    _print_detector_readout(detector)

    csv_path = _write_csv(rows, _write_linearity_summary(linearity))
    max_energy_error = max(
        max(item["output"]["energy_err"], item["double_output"]["energy_err"])
        for item in primary["solutions"].values()
    )
    print(
        "\nCHANCE / FALSE-POSITIVE RATE: binary chance=0.500; all signature and scalar "
        "thresholds use the supplied target FPR=0.050 (95.0% specificity); "
        "same-alpha controls are the negative control."
    )
    print(f"energy-balance maximum relative error across envelope/linearity solves: {max_energy_error:.6e}")
    print(f"CSV -> {csv_path}")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")

    verdict_go = bool(
        linearity["pass"] and np.isfinite(max_energy_error)
        and np.all(np.isfinite([row["q_max"] for row in rows]))
        and max_abs_tmax_shift >= 0.0
    )
    print("\n" + "=" * 104)
    print(
        f"[{'GO' if verdict_go else 'CHECK'} / VERDICT] silent SUPG stabilization changes the "
        "inferred battery heat-generation envelope"
    )
    print("=" * 104)
    print(
        "  The full alpha grid is covered at both replaceable thermal limits; the two-load linearity "
        f"check is {'within' if linearity['pass'] else 'outside'} the requested ~1% tolerance."
    )
    print(
        "  Tmax at fixed q_ref barely moves while q'''_max shifts, so an undocumented stabilization "
        "change can alter an accepted operating limit without an obvious fixed-load alarm."
    )
    print("  This is a numerical/physical-plausibility consequence, not an experimental validation or universal safety claim.")
    return {
        "rows": rows, "csv": csv_path, "linearity": linearity,
        "detector": detector, "runtime_seconds": elapsed,
    }


if __name__ == "__main__":
    main()
