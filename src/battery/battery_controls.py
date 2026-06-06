import os, sys, numpy as np
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT,"src","battery")); import battery_module as B
sys.path.insert(0, os.path.join(_ROOT,"src","audit"));   import supg_2d_engineering as A
from sklearn.model_selection import GroupKFold, cross_val_predict
TAB=os.path.join(_ROOT,"results","tables"); os.makedirs(TAB, exist_ok=True)

import csv
import time


"""Selectivity audit for the validated five-cell conjugate battery module.

The audit uses only the representative numerical/physical parameters exposed by
``battery_module``.  It is a numerical and physical-plausibility study, not an
experimental data match.  Every signature is formed from a working-mesh field
minus a fine nominal-SUPG reference at that field's own operating condition.
"""


TARGET_FPR = 0.05
SIGMA = 0.01
N_IC = 48
IC_SEEDS = tuple(1000 + i for i in range(N_IC))
WORK_N_CELL_X = 16
WORK_SEED = 2026
FINE_N_CELL_X = 26
FINE_SEED = 7001
NOISE_SEED = 41000
GRID_COARSE_SEED = 3001
GRID_FINE_SEED = 3002
GRID_COARSE_REFERENCE_SEED = 41001
GRID_FINE_REFERENCE_SEED = 41002


def supervised_sensitivity(nominal, changed, target_fpr=0.05):   # signature: held-out P(changed)
    X=A.feats(np.vstack([nominal, changed])); y=np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g=np.r_[np.arange(len(nominal)), np.arange(len(changed))]; k=min(5, len(nominal))
    p=cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k), method="predict_proba")[:,1]
    thr=np.percentile(p[y==0], 100*(1-target_fpr)); return float(np.mean(p[y==1] > thr))


def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):     # 1-D battery output: two-sided |v-median(nominal)|
    nom=np.asarray(nom_vals,float); chg=np.asarray(chg_vals,float); c=np.median(nom)
    thr=np.percentile(np.abs(nom-c), 100*(1-target_fpr)); return float(np.mean(np.abs(chg-c) > thr))


def decide(grid_controlled, sensitivity, fpr=0.05, detect_thr=0.50):
    if sensitivity <= 0.15: return "NO-FAULT"
    if not grid_controlled and sensitivity > detect_thr: return "INDETERMINATE"
    return "DETECT"


def _copy_ic(ic, **updates):
    data = dict(ic)
    data["cell_jitter"] = np.array(ic["cell_jitter"], dtype=float, copy=True)
    data.update(updates)
    if "cell_jitter" in data:
        data["cell_jitter"] = np.array(data["cell_jitter"], dtype=float, copy=True)
    return data


def _ramped_population(ics, ramp):
    return [_copy_ic(ic, inlet_ramp=float(ramp)) for ic in ics]


def _one_cell_population(ics, cell_index, factor):
    changed = []
    for ic in ics:
        jitter = np.array(ic["cell_jitter"], dtype=float, copy=True)
        jitter[int(cell_index)] = float(factor)
        changed.append(_copy_ic(ic, cell_jitter=jitter))
    return changed


def _clean_grids(pts, elems, tags, geom, ics, q_base, Ubar,
                 scheme="supg", alpha=1.0, nu_art=None):
    grids = []
    for ic in ics:
        kwargs = {
            "q_base": q_base,
            "Ubar": Ubar,
            "ic": ic,
            "alpha": alpha,
            "geom": geom,
        }
        if scheme == "artvisc":
            kwargs["nu_art"] = nu_art
        T = B.assemble_module(scheme, pts, elems, tags=tags, **kwargs)
        grids.append(B.to_module_grid(pts, T, window=B.SIGNATURE_WINDOW))
    return np.asarray(grids, dtype=float)


def _noisy_signatures(clean_grids, ref_grids, sigma=SIGMA, seed=NOISE_SEED):
    rng=np.random.default_rng(seed)
    signatures = []
    for Us, Ur in zip(clean_grids, ref_grids):
        Us_noisy = Us
        if sigma > 0:
            rms=np.sqrt(np.mean(Us**2)); Us_noisy=Us + sigma*rms*rng.standard_normal(Us.shape)
        signatures.append(B.sig_from_grid(Us_noisy, Ur))
    return np.asarray(signatures, dtype=float)


def _condition_signatures(pts, elems, tags, geom, ics, q_base, Ubar,
                          scheme="supg", alpha=1.0, nu_art=None,
                          reference_grids=None, reference_mesh=None,
                          reference_seed=FINE_SEED, noise_seed=NOISE_SEED):
    clean = _clean_grids(pts, elems, tags, geom, ics, q_base, Ubar,
                         scheme=scheme, alpha=alpha, nu_art=nu_art)
    if reference_grids is None:
        reference_grids = B.reference_module_grids(
            ics,
            fine_mesh=reference_mesh,
            q_base=q_base,
            Ubar=Ubar,
            n_cell_x=FINE_N_CELL_X,
            seed=reference_seed,
            window=B.SIGNATURE_WINDOW,
        )
    return _noisy_signatures(clean, np.asarray(reference_grids, dtype=float),
                             sigma=SIGMA, seed=noise_seed)


def _check_signatures(signatures, name):
    signatures = np.asarray(signatures, dtype=float)
    if signatures.ndim != 2 or signatures.shape[1] != len(A.LIB):
        raise RuntimeError(f"{name} has unexpected signature shape {signatures.shape}")
    if not np.all(np.isfinite(signatures)):
        raise RuntimeError(f"{name} contains non-finite signature values")
    return signatures


def _grid_sensitivity(fine_signatures, coarse_signatures, target_fpr=TARGET_FPR):
    """Compare the two explicitly observed resolutions for the grid control."""
    return supervised_sensitivity(fine_signatures, coarse_signatures,
                                  target_fpr=target_fpr)


def _record(results, nominal, changed, row, change_type, description,
            desired, grid_controlled=True, detector=supervised_sensitivity):
    changed = _check_signatures(changed, f"row {row} changed population")
    sensitivity = detector(nominal, changed, target_fpr=TARGET_FPR)
    decision = decide(grid_controlled, sensitivity, fpr=TARGET_FPR,
                      detect_thr=0.50)
    result = {
        "row": int(row),
        "change_type": change_type,
        "description": description,
        "sensitivity": float(sensitivity),
        "desired": desired,
        "decision": decision,
        "correct": int(decision == desired),
    }
    results.append(result)
    return result


def _write_csv(results):
    path = os.path.join(TAB, "battery_controls.csv")
    fields = ("row", "change_type", "description", "sensitivity",
              "desired", "decision", "correct")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["sensitivity"] = f"{result['sensitivity']:.6f}"
            writer.writerow(row)
    return path


def _print_result_table(results):
    print("\nRESULT TABLE :: battery controls audit, sigma=0.01")
    print(" row  type        sensitivity  desired          decision          correct  description")
    print(" " + "-" * 128)
    for result in results:
        print(f" {result['row']:>3d}  {result['change_type']:<10s}"
              f" {result['sensitivity']:>11.3f}  {result['desired']:<15s}"
              f" {result['decision']:<17s} {result['correct']:>7d}  "
              f"{result['description']}")


def main():
    started = time.perf_counter()
    print("=" * 96)
    print("BATTERY CONTROLS AUDIT: SELECTIVITY OF MODIFIED-EQUATION SIGNATURES")
    print("validated five-cell conjugate module; numerical changes vs matched physical/operating controls")
    print("=" * 96)
    print("representative literature-sourced parameters are imported from battery_module; no experimental match claimed")
    print("contact resistance: not in the model; omitted rather than fabricated")
    print(f"baseline: 2C, Ubar={B.UBAR_NOMINAL:.3f} m/s, N_IC={N_IC}, sigma={SIGMA:.2f}; "
          f"working mesh n_cell_x={WORK_N_CELL_X}, seed={WORK_SEED}")
    print(f"reference: fine nominal SUPG n_cell_x={FINE_N_CELL_X}, seed={FINE_SEED}, "
          f"window grid={A.GRID_OBS}x{A.GRID_OBS}")

    q_nominal = B.Q_BY_CRATE["2C"]
    U_nominal = B.UBAR_NOMINAL
    ics = [B.make_module_ic(seed) for seed in IC_SEEDS]

    working_mesh = B.make_module_mesh(n_cell_x=WORK_N_CELL_X, seed=WORK_SEED)
    pts, elems, tags = working_mesh
    geom = B.module_mesh_geometry(pts, elems)
    print(f"working mesh: {len(pts)} nodes, {len(elems)} triangles")

    print("[cache] solving baseline fine nominal-SUPG references once per IC ...")
    baseline_refs = np.asarray(B.reference_module_grids(
        ics,
        q_base=q_nominal,
        Ubar=U_nominal,
        n_cell_x=FINE_N_CELL_X,
        seed=FINE_SEED,
        window=B.SIGNATURE_WINDOW,
    ), dtype=float)
    print("[cache] solving baseline working-mesh nominal SUPG fields once per IC ...")
    baseline_clean = _clean_grids(
        pts, elems, tags, geom, ics, q_nominal, U_nominal,
        scheme="supg", alpha=1.0,
    )
    nominal_sigs = _check_signatures(
        _noisy_signatures(baseline_clean, baseline_refs,
                          sigma=SIGMA, seed=NOISE_SEED),
        "baseline nominal population",
    )
    nu_art = B.matched_artificial_diffusion(
        geom, Ubar=B.UBAR_NOMINAL, alpha=1.0,
    )
    print(f"matched ArtVisc nu_art={nu_art:.6e} m^2/s")

    results = []

    print("[row 1/9] SUPG detuning alpha=0.5 ...")
    row1 = _record(
        results, nominal_sigs,
        _condition_signatures(
            pts, elems, tags, geom, ics, q_nominal, U_nominal,
            scheme="supg", alpha=0.5, reference_grids=baseline_refs,
            noise_seed=NOISE_SEED + 100,
        ),
        1, "numerical", "SUPG detuning alpha=0.5 vs nominal SUPG alpha=1",
        "DETECT",
    )

    print("[row 2/9] Galerkin replacement ...")
    _record(
        results, nominal_sigs,
        _condition_signatures(
            pts, elems, tags, geom, ics, q_nominal, U_nominal,
            scheme="galerkin", alpha=0.0, reference_grids=baseline_refs,
            noise_seed=NOISE_SEED + 200,
        ),
        2, "numerical", "Galerkin replaces nominal SUPG",
        "DETECT",
    )

    print("[row 3/9] matched isotropic ArtVisc ...")
    row3 = _record(
        results, nominal_sigs,
        _condition_signatures(
            pts, elems, tags, geom, ics, q_nominal, U_nominal,
            scheme="artvisc", alpha=1.0, nu_art=nu_art,
            reference_grids=baseline_refs, noise_seed=NOISE_SEED + 300,
        ),
        3, "numerical", "matched isotropic ArtVisc vs nominal SUPG; SUPG-vs-ArtVisc separability",
        "DETECT",
    )

    print("[row 4/9] heat-generation changes with matched references ...")
    heat_sigs = []
    for index, scale in enumerate((0.90, 1.10, 0.80, 1.20)):
        heat_sigs.append(_condition_signatures(
            pts, elems, tags, geom, ics, q_nominal * scale, U_nominal,
            scheme="supg", alpha=1.0,
            reference_seed=FINE_SEED + 10 + index,
            noise_seed=NOISE_SEED + 400 + index,
        ))
    _record(
        results, nominal_sigs, np.vstack(heat_sigs),
        4, "operating", "heat generation q_base scaled by -10%, +10%, -20%, +20% (matched references)",
        "NO-FAULT",
    )

    print("[row 5/9] coolant-flow changes with matched references ...")
    flow_sigs = []
    for index, scale in enumerate((0.90, 1.10, 0.80, 1.20)):
        flow_sigs.append(_condition_signatures(
            pts, elems, tags, geom, ics, q_nominal, U_nominal * scale,
            scheme="supg", alpha=1.0,
            reference_seed=FINE_SEED + 20 + index,
            noise_seed=NOISE_SEED + 500 + index,
        ))
    _record(
        results, nominal_sigs, np.vstack(flow_sigs),
        5, "operating", "coolant Ubar scaled by -10%, +10%, -20%, +20% (matched references)",
        "NO-FAULT",
    )

    print("[row 6/9] inlet-profile ramp changes with matched references ...")
    inlet_sigs = []
    for index, ramp in enumerate((3.0, -3.0)):
        inlet_ics = _ramped_population(ics, ramp)
        inlet_sigs.append(_condition_signatures(
            pts, elems, tags, geom, inlet_ics, q_nominal, U_nominal,
            scheme="supg", alpha=1.0,
            reference_seed=FINE_SEED + 30 + index,
            noise_seed=NOISE_SEED + 600 + index,
        ))
    _record(
        results, nominal_sigs, np.vstack(inlet_sigs),
        6, "operating", "inlet-ramp operating profile set to +3 K and -3 K (matched references)",
        "NO-FAULT",
    )

    print("[row 7/9] cell conductivity changes with monkeypatch/restore ...")
    conductivity_sigs = []
    old_kb = B.KB
    old_cell_k = B.K_BY_REGION["cell"]
    try:
        for index, scale in enumerate((0.90, 1.10)):
            B.KB = old_kb * scale
            B.K_BY_REGION["cell"] = old_cell_k * scale
            conductivity_geom = B.module_mesh_geometry(pts, elems)
            conductivity_sigs.append(_condition_signatures(
                pts, elems, tags, conductivity_geom, ics, q_nominal, U_nominal,
                scheme="supg", alpha=1.0,
                reference_seed=FINE_SEED + 40 + index,
                noise_seed=NOISE_SEED + 700 + index,
            ))
    finally:
        B.KB = old_kb
        B.K_BY_REGION["cell"] = old_cell_k
    if B.KB != old_kb or B.K_BY_REGION["cell"] != old_cell_k:
        raise RuntimeError("cell conductivity monkeypatch was not restored")
    _record(
        results, nominal_sigs, np.vstack(conductivity_sigs),
        7, "operating", "cell conductivity KB scaled by -10% and +10% (matched references)",
        "NO-FAULT",
    )

    print("[row 8/9] one-cell heat imbalance with matched references ...")
    imbalance_sigs = []
    center_cell = B.NCELL // 2
    for index, factor in enumerate((1.10, 0.90)):
        imbalance_ics = _one_cell_population(ics, center_cell, factor)
        imbalance_sigs.append(_condition_signatures(
            pts, elems, tags, geom, imbalance_ics, q_nominal, U_nominal,
            scheme="supg", alpha=1.0,
            reference_seed=FINE_SEED + 50 + index,
            noise_seed=NOISE_SEED + 800 + index,
        ))
    _record(
        results, nominal_sigs, np.vstack(imbalance_sigs),
        8, "operating", "one-cell heat-generation factor set to +10% and -10% (physical change; matched references)",
        "NO-FAULT",
    )

    print("[row 9/9] grid-only change with per-mesh matched fine references ...")
    coarse_mesh = B.make_module_mesh(n_cell_x=11, seed=GRID_COARSE_SEED)
    fine_mesh = B.make_module_mesh(n_cell_x=22, seed=GRID_FINE_SEED)
    coarse_pts, coarse_elems, coarse_tags = coarse_mesh
    fine_pts, fine_elems, fine_tags = fine_mesh
    coarse_geom = B.module_mesh_geometry(coarse_pts, coarse_elems)
    fine_geom = B.module_mesh_geometry(fine_pts, fine_elems)
    coarse_reference_mesh = B.make_module_mesh(
        n_cell_x=FINE_N_CELL_X, seed=GRID_COARSE_REFERENCE_SEED,
    )
    fine_reference_mesh = B.make_module_mesh(
        n_cell_x=FINE_N_CELL_X, seed=GRID_FINE_REFERENCE_SEED,
    )
    coarse_grid_sigs = _condition_signatures(
        coarse_pts, coarse_elems, coarse_tags, coarse_geom, ics,
        q_nominal, U_nominal, scheme="supg", alpha=1.0,
        reference_mesh=coarse_reference_mesh,
        noise_seed=NOISE_SEED + 900,
    )
    fine_grid_sigs = _condition_signatures(
        fine_pts, fine_elems, fine_tags, fine_geom, ics,
        q_nominal, U_nominal, scheme="supg", alpha=1.0,
        reference_mesh=fine_reference_mesh,
        noise_seed=NOISE_SEED + 1000,
    )
    _record(
        results, fine_grid_sigs, coarse_grid_sigs,
        9, "grid", "same nominal SUPG on coarse n_cell_x=11 and fine n_cell_x=22 meshes; grid uncontrolled",
        "INDETERMINATE", grid_controlled=False, detector=_grid_sensitivity,
    )

    _print_result_table(results)
    print(f"\nchance/FPR: target false-positive rate={TARGET_FPR:.2f} (95% specificity); "
          f"threshold is the {100 * (1 - TARGET_FPR):.1f}th percentile of nominal held-out probabilities")
    print("decision thresholds: sensitivity <=0.15 -> NO-FAULT; grid-uncontrolled sensitivity >0.50 -> INDETERMINATE")
    print(f"SUPG-vs-ArtVisc separability: sensitivity={row3['sensitivity']:.3f} (row 3)")

    correct = sum(result["correct"] for result in results)
    mismatches = [
        f"row {result['row']} ({result['decision']} != {result['desired']})"
        for result in results if not result["correct"]
    ]
    verdict = "GO" if correct == len(results) else "CHECK"
    print(f"\n[{verdict} / VERDICT] {correct}/{len(results)} correct; "
          "target = numerical DETECT, operating NO-FAULT, grid INDETERMINATE")
    if mismatches:
        print("mismatches: " + "; ".join(mismatches))
    csv_path = _write_csv(results)
    print(f"CSV -> {csv_path}")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    return {"results": results, "csv": csv_path,
            "runtime_seconds": elapsed, "correct": correct}


if __name__ == "__main__":
    main()
