#!/usr/bin/env python3
"""Predeclared battery coupling domain map for subtle SUPG detuning.

The fixed endpoint is split-conformal signature TPR for alpha=0.5 versus
nominal SUPG at 1% RMS-relative grid noise. Cell effective conductivity is the
coupling coordinate. Every ladder point has its own conductivity-matched fine
nominal reference. The original alpha=0.5 baseline miss is retained as a frozen
historical result and is not reclassified by this experiment.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "battery"))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import battery_module as B
import supg_2d_engineering as A
import cross_conformal as CC


class CouplingSpec(NamedTuple):
    coupling_id: str
    conductivity: float


N_IC = 48
IC_SEEDS = tuple(range(1000, 1000 + N_IC))
FOLDS = 6
N_CAL = 19
CAL_RANK = 19
WORK_N_CELL_X = 16
WORK_SEED = 2026
REFERENCE_N_CELL_X = 26
REFERENCE_SEED = 7001
Q_BASE = B.Q_BY_CRATE["2C"]
UBAR = B.UBAR_NOMINAL
SIGMA = 0.01
TARGET_ALPHA = 0.5
NOMINAL_ALPHA = 1.0
COUPLING_LADDER = (
    CouplingSpec("k05", 5.0),
    CouplingSpec("k10", 10.0),
    CouplingSpec("k20_baseline", 20.0),
    CouplingSpec("k40", 40.0),
    CouplingSpec("k80", 80.0),
)
BASELINE_CONDUCTIVITY = 20.0
ARMS = (
    ("nominal", "supg", NOMINAL_ALPHA),
    ("target", "supg", TARGET_ALPHA),
    ("galerkin_positive", "galerkin", 0.0),
)
NEGATIVE_ARM = "nominal_independent_noise"
TARGET_FPR = CC.TARGET_FPR
POSITIVE_CONTROL_MIN_TPR = 0.80
NEGATIVE_CONTROL_MAX_TPR = 0.20
MAX_MEASURED_FPR = 0.20
SYSTEMATIC_MIN_TPR_RANGE = 0.20
SYSTEMATIC_MIN_ABS_SPEARMAN = 0.80
MAX_ENERGY_ERROR = 0.01
FROZEN_BASELINE_SENSITIVITY = 0.083333
FROZEN_BASELINE_DECISION = "NO-FAULT"
FIXED_ENDPOINT = (
    "paired split-conformal signature TPR for SUPG alpha=0.5 versus alpha=1.0 "
    "at each cell-conductivity ladder point under 1% RMS-relative noise"
)
GO_CRITERION = (
    "solver health passes, all nominal-independent-noise TPRs are at most 0.20, "
    "all Galerkin positive-control TPRs are at least 0.80, and every measured "
    "false-alarm rate is at most 0.20"
)
REPORT_CRITERION = (
    "report monotone coupling dependence only when endpoint TPR range is at least "
    "0.20, absolute Spearman rho is at least 0.80, and the paired extreme-ladder "
    "95% bootstrap contrast interval excludes zero; otherwise report no resolved "
    "systematic dependence without altering the frozen baseline miss"
)
CACHE_SCHEMA = "battery-coupling-sensitivity-clean-v1"
CACHE_PATH = os.path.join(
    _ROOT, "results", "cache", "battery_coupling_sensitivity_clean.npz"
)
CSV_PATH = os.path.join(
    _ROOT, "results", "tables", "battery_coupling_sensitivity.csv"
)
CSV_FIELDS = (
    "row_type", "coupling_id", "cell_conductivity_w_mk",
    "coupling_ratio_to_baseline", "arm", "scheme", "alpha", "n_ic",
    "sigma", "folds", "n_cal", "cal_rank", "tpr", "tpr_lo", "tpr_hi",
    "fpr_measured", "fpr_lo", "fpr_hi", "fpr_target",
    "max_energy_error", "max_linear_residual", "endpoint_tpr_range",
    "spearman_rho", "extreme_contrast", "contrast_lo", "contrast_hi",
    "negative_control_pass", "positive_control_pass", "solver_health_pass",
    "systematic_dependence", "report_status", "frozen_baseline_sensitivity",
    "frozen_baseline_decision", "protocol_hash",
)


def coupling_ladder():
    """Return an immutable copy of the predeclared conductivity ladder."""
    return tuple(COUPLING_LADDER)


@contextmanager
def cell_conductivity(value):
    """Temporarily set both battery-module locations that feed cell diffusion."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("cell conductivity must be finite and positive")
    old_kb = B.KB
    old_region = B.K_BY_REGION["cell"]
    B.KB = value
    B.K_BY_REGION["cell"] = value
    try:
        yield
    finally:
        B.KB = old_kb
        B.K_BY_REGION["cell"] = old_region


def protocol_payload():
    return {
        "scientific_question": (
            "does subtle coolant-side SUPG detuning detectability depend "
            "systematically on conjugate thermal coupling"
        ),
        "fixed_endpoint": FIXED_ENDPOINT,
        "coupling_parameter": "cell effective conductivity KB",
        "coupling_units": "W/(m K)",
        "coupling_ladder": [spec._asdict() for spec in COUPLING_LADDER],
        "baseline_conductivity": BASELINE_CONDUCTIVITY,
        "n_ic": N_IC,
        "ic_seeds": list(IC_SEEDS),
        "working_mesh": {"n_cell_x": WORK_N_CELL_X, "seed": WORK_SEED},
        "reference_mesh": {
            "n_cell_x": REFERENCE_N_CELL_X, "seed": REFERENCE_SEED,
        },
        "matched_nominal_reference_per_ladder_point": True,
        "working_arms": [
            {"arm": arm, "scheme": scheme, "alpha": alpha}
            for arm, scheme, alpha in ARMS
        ],
        "q_base": Q_BASE,
        "ubar": UBAR,
        "nominal_alpha": NOMINAL_ALPHA,
        "target_alpha": TARGET_ALPHA,
        "sigma": SIGMA,
        "noise": "independent Gaussian RMS-relative grid noise",
        "signature_terms": list(A.LIB),
        "folds": FOLDS,
        "n_cal": N_CAL,
        "cal_rank": CAL_RANK,
        "target_fpr": TARGET_FPR,
        "negative_control": NEGATIVE_ARM,
        "positive_control": "Galerkin replacement versus nominal SUPG",
        "positive_control_min_tpr": POSITIVE_CONTROL_MIN_TPR,
        "negative_control_max_tpr": NEGATIVE_CONTROL_MAX_TPR,
        "max_measured_fpr": MAX_MEASURED_FPR,
        "systematic_min_tpr_range": SYSTEMATIC_MIN_TPR_RANGE,
        "systematic_min_abs_spearman": SYSTEMATIC_MIN_ABS_SPEARMAN,
        "max_energy_error": MAX_ENERGY_ERROR,
        "frozen_baseline_sensitivity": FROZEN_BASELINE_SENSITIVITY,
        "frozen_baseline_decision": FROZEN_BASELINE_DECISION,
        "go_criterion": GO_CRITERION,
        "report_criterion": REPORT_CRITERION,
    }


def protocol_hash():
    encoded = json.dumps(
        protocol_payload(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_metadata():
    return {
        "cache_schema": CACHE_SCHEMA,
        "owner": os.path.basename(__file__),
        "protocol_hash": protocol_hash(),
        **protocol_payload(),
    }


def pairing_plan(repeats=1, seed=0):
    """Expose the exact 21-train/19-calibration/8-test paired-IC plan."""
    return CC.fold_plan(
        N_IC, repeats=int(repeats), folds=FOLDS, n_cal=N_CAL, seed=int(seed)
    )


def solve_counts():
    references = len(COUPLING_LADDER) * N_IC
    working = len(COUPLING_LADDER) * len(ARMS) * N_IC
    return {"reference": references, "working": working,
            "total": references + working}


def _interface_edge_count(elems, regions):
    edges = {}
    for element, region in zip(np.asarray(elems, int), np.asarray(regions)):
        for edge in ((element[0], element[1]), (element[1], element[2]),
                     (element[2], element[0])):
            edges.setdefault(tuple(sorted(edge)), set()).add(str(region))
    return sum({"cell", "plate"}.issubset(names) for names in edges.values())


def verify_conjugate_transmission(pts, elems):
    """Verify that the ladder changes the cell side of a real cell-plate interface.

    This checks the coefficient consumed by ``assemble_module`` rather than only
    the public KB label. The experiment stops before population solves if the
    chosen parameter is disconnected from the conjugate diffusion operator.
    """
    observed = []
    noncell = None
    interface_edges = None
    for spec in COUPLING_LADDER:
        with cell_conductivity(spec.conductivity):
            geom = B.module_mesh_geometry(pts, elems)
        if interface_edges is None:
            interface_edges = _interface_edge_count(elems, geom["regions"])
        cell_values = geom["k"][geom["cell"]]
        if cell_values.size == 0 or not np.allclose(cell_values, spec.conductivity):
            raise RuntimeError(
                "chosen coupling parameter does not vary cell diffusion in the current model"
            )
        fixed_values = geom["k"][~geom["cell"]]
        if noncell is None:
            noncell = fixed_values
        elif not np.array_equal(noncell, fixed_values):
            raise RuntimeError("coupling ladder altered a non-cell material coefficient")
        observed.append(float(np.mean(cell_values / B.RHOCP_F)))
    if interface_edges is None or interface_edges <= 0:
        raise RuntimeError("current model has no conforming cell-plate transmission interface")
    if not np.all(np.diff(observed) > 0.0):
        raise RuntimeError(
            "chosen coupling parameter does not vary conjugate transmission monotonically"
        )
    return {"cell_kappa": np.asarray(observed), "interface_edges": interface_edges}


def _expected_shapes():
    n_ladder = len(COUPLING_LADDER)
    n_arms = len(ARMS)
    grid = (A.GRID_OBS, A.GRID_OBS)
    return {
        "reference_grids": (n_ladder, N_IC) + grid,
        "working_grids": (n_ladder, n_arms, N_IC) + grid,
        "energy_errors": (n_ladder, n_arms + 1, N_IC),
        "linear_residuals": (n_ladder, n_arms + 1, N_IC),
        "cell_kappa": (n_ladder,),
    }


def _validate_clean_fields(fields):
    expected = _expected_shapes()
    if set(fields) != set(expected):
        raise ValueError(f"clean fields must contain exactly {sorted(expected)}")
    checked = {}
    for name, shape in expected.items():
        value = np.asarray(fields[name], dtype=float)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains a non-finite value")
        checked[name] = value
    if not np.all(np.diff(checked["cell_kappa"]) > 0.0):
        raise ValueError("cached cell transmission coefficients must increase")
    return checked


def save_clean_cache(path, fields):
    fields = _validate_clean_fields(fields)
    path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    metadata_json = json.dumps(cache_metadata(), sort_keys=True, separators=(",", ":"))
    handle = tempfile.NamedTemporaryFile(
        dir=directory, prefix=".battery_coupling_", suffix=".npz", delete=False
    )
    temporary = handle.name
    handle.close()
    try:
        np.savez_compressed(temporary, metadata_json=np.array(metadata_json), **fields)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_clean_cache(path):
    with np.load(os.fspath(path), allow_pickle=False) as archive:
        required = {"metadata_json", *_expected_shapes().keys()}
        if set(archive.files) != required:
            raise ValueError(f"cache members must be exactly {sorted(required)}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != cache_metadata():
            raise ValueError("clean-field cache metadata does not match the frozen protocol")
        fields = {name: np.array(archive[name], copy=True)
                  for name in _expected_shapes()}
    return _validate_clean_fields(fields)


def _solve_grid(scheme, pts, elems, tags, geom, ic, alpha):
    temperature, regions, meta = B.assemble_module(
        scheme, pts, elems, tags=tags, q_base=Q_BASE, Ubar=UBAR, ic=ic,
        alpha=float(alpha), geom=geom, return_meta=True, energy_closure=True,
    )
    output = B.thermal_outputs_module(
        pts, temperature, regions, tags=tags, Ubar=meta["Ubar"],
        total_generation=meta["total_generation"], T_in=B.T_IN,
    )
    grid = B.to_module_grid(pts, temperature, window=B.SIGNATURE_WINDOW)
    return grid, float(output["energy_err"]), float(meta["linear_residual"])


def solve_clean_fields():
    ics = [B.make_module_ic(seed) for seed in IC_SEEDS]
    work = B.make_module_mesh(n_cell_x=WORK_N_CELL_X, seed=WORK_SEED)
    fine = B.make_module_mesh(n_cell_x=REFERENCE_N_CELL_X, seed=REFERENCE_SEED)
    transmission = verify_conjugate_transmission(work[0], work[1])
    shapes = _expected_shapes()
    fields = {name: np.empty(shape, dtype=float) for name, shape in shapes.items()}
    fields["cell_kappa"][:] = transmission["cell_kappa"]

    for ladder_index, spec in enumerate(COUPLING_LADDER):
        with cell_conductivity(spec.conductivity):
            work_geom = B.module_mesh_geometry(work[0], work[1])
            fine_geom = B.module_mesh_geometry(fine[0], fine[1])
            for ic_index, ic in enumerate(ics):
                ref, energy, residual = _solve_grid(
                    "supg", fine[0], fine[1], fine[2], fine_geom, ic, NOMINAL_ALPHA
                )
                fields["reference_grids"][ladder_index, ic_index] = ref
                fields["energy_errors"][ladder_index, 0, ic_index] = energy
                fields["linear_residuals"][ladder_index, 0, ic_index] = residual
                for arm_index, (_, scheme, alpha) in enumerate(ARMS):
                    grid, energy, residual = _solve_grid(
                        scheme, work[0], work[1], work[2], work_geom, ic, alpha
                    )
                    fields["working_grids"][ladder_index, arm_index, ic_index] = grid
                    fields["energy_errors"][ladder_index, arm_index + 1, ic_index] = energy
                    fields["linear_residuals"][ladder_index, arm_index + 1, ic_index] = residual
    return _validate_clean_fields(fields)


def load_or_solve_clean_fields(cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        return load_clean_cache(cache_path), "loaded"
    fields = solve_clean_fields()
    if cache_path is not None:
        save_clean_cache(cache_path, fields)
        return fields, "saved"
    return fields, "disabled"


def _noise(grid, seed):
    grid = np.asarray(grid, dtype=float)
    rng = np.random.default_rng(int(seed))
    rms = float(np.sqrt(np.mean(grid ** 2)))
    return grid + SIGMA * rms * rng.standard_normal(grid.shape)


def _noise_seed(ladder_index, arm_index, ic_index):
    return 618_000 + 10_000 * int(ladder_index) + 100 * int(arm_index) + int(ic_index)


def extract_signatures(fields):
    fields = _validate_clean_fields(fields)
    populations = {}
    for ladder_index, spec in enumerate(COUPLING_LADDER):
        reference = fields["reference_grids"][ladder_index]
        by_arm = {}
        for arm_index, (arm, _, _) in enumerate(ARMS):
            signatures = np.empty((N_IC, len(A.LIB)), dtype=float)
            for ic_index in range(N_IC):
                observed = _noise(
                    fields["working_grids"][ladder_index, arm_index, ic_index],
                    _noise_seed(ladder_index, arm_index, ic_index),
                )
                signatures[ic_index] = B.sig_from_grid(observed, reference[ic_index])
            by_arm[arm] = signatures
        negative = np.empty((N_IC, len(A.LIB)), dtype=float)
        nominal_index = 0
        for ic_index in range(N_IC):
            observed = _noise(
                fields["working_grids"][ladder_index, nominal_index, ic_index],
                _noise_seed(ladder_index, len(ARMS), ic_index),
            )
            negative[ic_index] = B.sig_from_grid(observed, reference[ic_index])
        by_arm[NEGATIVE_ARM] = negative
        populations[spec.coupling_id] = by_arm
    return populations


def paired_detection(nominal, changed, seed):
    nominal = np.asarray(nominal, dtype=float)
    changed = np.asarray(changed, dtype=float)
    if nominal.shape != changed.shape or nominal.shape[0] != N_IC:
        raise ValueError("nominal and changed signatures must pair the same 48 ICs")
    return CC.split_conformal_detection(
        nominal, changed, seed=int(seed), repeats=CC.REPEATS, folds=FOLDS,
        n_cal=N_CAL, cal_rank=CAL_RANK, n_boot=CC.N_BOOT,
    )


def evaluate(populations):
    results = []
    comparisons = (
        ("target", "supg", TARGET_ALPHA),
        (NEGATIVE_ARM, "supg", NOMINAL_ALPHA),
        ("galerkin_positive", "galerkin", 0.0),
    )
    for ladder_index, spec in enumerate(COUPLING_LADDER):
        by_arm = populations[spec.coupling_id]
        nominal = by_arm["nominal"]
        for arm_index, (arm, scheme, alpha) in enumerate(comparisons):
            detection = paired_detection(
                nominal, by_arm[arm],
                seed=729_000 + 10_000 * ladder_index + 100 * arm_index,
            )
            results.append({
                "spec": spec, "arm": arm, "scheme": scheme,
                "alpha": alpha, "detection": detection,
            })
    return results


def solver_health(fields):
    fields = _validate_clean_fields(fields)
    max_energy = float(np.max(np.abs(fields["energy_errors"])))
    max_residual = float(np.max(np.abs(fields["linear_residuals"])))
    expected_kappa = np.array(
        [spec.conductivity / B.RHOCP_F for spec in COUPLING_LADDER]
    )
    transmission_ok = bool(np.allclose(fields["cell_kappa"], expected_kappa))
    return {
        "max_energy_error": max_energy,
        "max_linear_residual": max_residual,
        "transmission_ok": transmission_ok,
        "pass": bool(max_energy <= MAX_ENERGY_ERROR and np.isfinite(max_residual)
                     and transmission_ok),
    }


def assess(results, health):
    target = [row for row in results if row["arm"] == "target"]
    negative = [row for row in results if row["arm"] == NEGATIVE_ARM]
    positive = [row for row in results if row["arm"] == "galerkin_positive"]
    if len(target) != len(COUPLING_LADDER):
        raise ValueError("assessment needs one target endpoint per ladder point")
    target.sort(key=lambda row: row["spec"].conductivity)
    tprs = np.array([row["detection"]["tpr"] for row in target], dtype=float)
    conductivities = np.array([row["spec"].conductivity for row in target], dtype=float)
    endpoint_range = float(np.ptp(tprs))
    rho = float(spearmanr(conductivities, tprs).statistic)
    if not np.isfinite(rho):
        rho = 0.0
    contrast_values = target[-1]["detection"]["detect"] - target[0]["detection"]["detect"]
    contrast = float(np.mean(contrast_values))
    contrast_ci = tuple(float(value) for value in CC.cluster_bootstrap_ci(
        contrast_values, seed=831_000
    ))
    contrast_excludes_zero = contrast_ci[0] > 0.0 or contrast_ci[1] < 0.0
    negative_pass = bool(negative and all(
        row["detection"]["tpr"] <= NEGATIVE_CONTROL_MAX_TPR for row in negative
    ))
    positive_pass = bool(positive and all(
        row["detection"]["tpr"] >= POSITIVE_CONTROL_MIN_TPR for row in positive
    ))
    fpr_pass = bool(results and all(
        row["detection"]["fpr"] <= MAX_MEASURED_FPR for row in results
    ))
    go = bool(health["pass"] and negative_pass and positive_pass and fpr_pass)
    systematic = bool(
        go and endpoint_range >= SYSTEMATIC_MIN_TPR_RANGE
        and abs(rho) >= SYSTEMATIC_MIN_ABS_SPEARMAN and contrast_excludes_zero
    )
    if not go:
        status = "CHECK_CONTROLS_OR_SOLVER"
    elif systematic:
        status = "GO_REPORT_COUPLING_DEPENDENT"
    else:
        status = "GO_REPORT_NO_SYSTEMATIC_DEPENDENCE_RESOLVED"
    return {
        "go": go, "systematic_dependence": systematic, "report_status": status,
        "endpoint_tpr_range": endpoint_range, "spearman_rho": rho,
        "extreme_contrast": contrast, "contrast_ci": contrast_ci,
        "negative_control_pass": negative_pass,
        "positive_control_pass": positive_pass, "fpr_pass": fpr_pass,
        "solver_health_pass": bool(health["pass"]),
    }


def csv_rows(results, assessment, health):
    digest = protocol_hash()
    common = {
        "n_ic": N_IC, "sigma": f"{SIGMA:.2f}", "folds": FOLDS,
        "n_cal": N_CAL, "cal_rank": CAL_RANK,
        "fpr_target": f"{TARGET_FPR:.2f}",
        "max_energy_error": f"{health['max_energy_error']:.10e}",
        "max_linear_residual": f"{health['max_linear_residual']:.10e}",
        "negative_control_pass": int(assessment["negative_control_pass"]),
        "positive_control_pass": int(assessment["positive_control_pass"]),
        "solver_health_pass": int(assessment["solver_health_pass"]),
        "systematic_dependence": int(assessment["systematic_dependence"]),
        "report_status": assessment["report_status"],
        "frozen_baseline_sensitivity": f"{FROZEN_BASELINE_SENSITIVITY:.6f}",
        "frozen_baseline_decision": FROZEN_BASELINE_DECISION,
        "protocol_hash": digest,
    }
    rows = []
    for row in results:
        spec = row["spec"]
        detection = row["detection"]
        rows.append({
            "row_type": "endpoint" if row["arm"] == "target" else "control",
            "coupling_id": spec.coupling_id,
            "cell_conductivity_w_mk": f"{spec.conductivity:.1f}",
            "coupling_ratio_to_baseline": f"{spec.conductivity / BASELINE_CONDUCTIVITY:.4f}",
            "arm": row["arm"], "scheme": row["scheme"],
            "alpha": f"{row['alpha']:.1f}",
            "tpr": f"{detection['tpr']:.6f}",
            "tpr_lo": f"{detection['tpr_ci'][0]:.6f}",
            "tpr_hi": f"{detection['tpr_ci'][1]:.6f}",
            "fpr_measured": f"{detection['fpr']:.6f}",
            "fpr_lo": f"{detection['fpr_ci'][0]:.6f}",
            "fpr_hi": f"{detection['fpr_ci'][1]:.6f}",
            **common,
        })
    rows.append({
        "row_type": "summary", "arm": "target_domain_map",
        "endpoint_tpr_range": f"{assessment['endpoint_tpr_range']:.6f}",
        "spearman_rho": f"{assessment['spearman_rho']:.6f}",
        "extreme_contrast": f"{assessment['extreme_contrast']:.6f}",
        "contrast_lo": f"{assessment['contrast_ci'][0]:.6f}",
        "contrast_hi": f"{assessment['contrast_ci'][1]:.6f}",
        **common,
    })
    return [{field: row.get(field, "") for field in CSV_FIELDS} for row in rows]


def write_csv(path, rows):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_predeclaration(cache_path):
    print("BATTERY COUPLING DOMAIN MAP: PREDECLARED PROTOCOL")
    print(f"endpoint: {FIXED_ENDPOINT}")
    print(f"ladder [W/(m K)]: {[spec.conductivity for spec in COUPLING_LADDER]}")
    print(f"paired ICs={N_IC}; split={FOLDS} folds, {N_CAL} calibration, 21 train, 8 test")
    print(f"negative control: {NEGATIVE_ARM}; positive control: Galerkin replacement")
    print(f"GO: {GO_CRITERION}")
    print(f"REPORT: {REPORT_CRITERION}")
    print(f"frozen baseline miss: sensitivity={FROZEN_BASELINE_SENSITIVITY:.6f}, "
          f"decision={FROZEN_BASELINE_DECISION}")
    print(f"optional cache: {cache_path if cache_path is not None else 'disabled'}")
    print(f"protocol hash: {protocol_hash()}")


def run(write=True, cache_path=None):
    started = time.perf_counter()
    if CC.choose_folds(N_IC) != FOLDS or (CC.N_CAL, CC.CAL_RANK) != (N_CAL, CAL_RANK):
        raise RuntimeError("cross-conformal 48-IC split is not the predeclared 6-fold/19-calibration design")
    print_predeclaration(cache_path)
    fields, cache_state = load_or_solve_clean_fields(cache_path)
    health = solver_health(fields)
    results = evaluate(extract_signatures(fields))
    assessment = assess(results, health)
    rows = csv_rows(results, assessment, health)
    csv_path = write_csv(CSV_PATH, rows) if write else None
    print(f"clean-field cache: {cache_state}")
    print(f"solver health: {health}")
    for row in results:
        print(f"{row['spec'].coupling_id:>12s} {row['arm']:<25s} "
              f"TPR={row['detection']['tpr']:.3f} FPR={row['detection']['fpr']:.3f}")
    print(f"assessment: {assessment['report_status']}")
    print("original baseline miss remains frozen and is not relabeled")
    if csv_path is not None:
        print(f"CSV -> {csv_path}")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    return {"results": results, "assessment": assessment, "health": health,
            "rows": rows, "csv": csv_path, "runtime_seconds": elapsed}


def build_parser():
    parser = argparse.ArgumentParser(description="battery coupling domain-map experiment")
    parser.add_argument("--cache", nargs="?", const=CACHE_PATH, default=None,
                        help="load or create the optional script-owned NPZ cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the experiment without writing its CSV")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(write=not args.dry_run, cache_path=args.cache)


if __name__ == "__main__":
    main()
