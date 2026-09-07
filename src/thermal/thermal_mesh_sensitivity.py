#!/usr/bin/env python3
"""Direct solver-mesh sensitivity of the Pe=100 thermal signature detector.

The varied quantity is the working FEM mesh.  Every mesh is observed on the same
64x64 grid and compared with the same clean 180x60 nominal-SUPG reference.  Thus
``solver_nx``/``solver_ny`` in the output describe the FEM discretization while
``observation_grid_n`` describes only the sampled field used for signature
extraction; the two are never treated as interchangeable.

The protocol is frozen in module constants before any solve: 60 paired ICs,
Pe=100, alpha=0.5..1.5, the reviewer sweep's base5 signature, one 1%-RMS white
noise realization per field, and the shared measured split-conformal detector.
Weakening and strengthening are reported separately.  A GO requires finite
sampled limits for every mesh and direction and permits at most one sampled
alpha step of degradation relative to the validated 60x20 anchor.  Improvements
cannot offset a degradation, and every curve, censored limit, and degradation is
retained in the CSV.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from typing import NamedTuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import heated_channel as HC
import reviewer_sensitivity as RS
import cross_conformal as CC


class MeshSpec(NamedTuple):
    mesh_id: str
    nx: int
    ny: int
    seed: int


PE = 100.0
N_IC = 60
IC_SEEDS = tuple(range(1000, 1000 + N_IC))
OBSERVATION_GRID_N = 64
LIBRARY_NAME = "base5"
LIBRARY_TERMS = tuple(RS.LIBRARIES[LIBRARY_NAME])
NOISE_MODEL = "white"
SIGMA = 0.01
ALPHAS = np.round(np.arange(0.5, 1.51, 0.1), 1)
ALPHAS_WEAK = np.round(np.arange(1.0, 0.49, -0.1), 1)
ALPHAS_STRONG = np.round(np.arange(1.0, 1.51, 0.1), 1)
DIRECTIONS = (("weaken", ALPHAS_WEAK), ("strengthen", ALPHAS_STRONG))
MESH_PLAN = (
    MeshSpec("coarse_45x15", 45, 15, 9015),
    MeshSpec("anchor_60x20", 60, 20, 2026),
    MeshSpec("refined_90x30", 90, 30, 9030),
)
ANCHOR_MESH_ID = "anchor_60x20"
REFERENCE_MESH = MeshSpec("fixed_reference_180x60", 180, 60, 7001)
TARGET_TPR = 0.95
TARGET_FPR = CC.TARGET_FPR
ALPHA_STEP = 0.1
MAX_ALLOWED_DEGRADATION = ALPHA_STEP
GO_CRITERION = (
    "GO only if every mesh/direction sampled limit is finite and no non-anchor "
    "limit is more than 0.1 delta_alpha worse than the anchor in that direction"
)
REPORTING_RULE = (
    "report both directions and every sampled curve; preserve censored and "
    "degraded cells separately; improvements never offset degradations"
)
CACHE_SCHEMA = "thermal-mesh-sensitivity-clean-v1"
CSV_PATH = os.path.join(_ROOT, "results", "tables", "thermal_mesh_sensitivity.csv")
CACHE_PATH = os.path.join(_ROOT, "results", "cache", "thermal_mesh_sensitivity_clean.npz")
CSV_FIELDS = (
    "type", "mesh_id", "solver_nx", "solver_ny", "solver_seed", "solver_nodes",
    "solver_elements", "observation_grid_n", "reference_nx", "reference_ny",
    "reference_seed", "direction", "alpha", "delta_alpha", "n_ic",
    "noise_model", "sigma", "library", "n_terms", "tpr", "tpr_lo", "tpr_hi",
    "fpr_measured", "fpr_lo", "fpr_hi", "fpr_target",
    "detection_limit_delta_alpha", "limit_lo", "limit_hi",
    "limit_censored_fraction", "anchor_limit_delta_alpha", "limit_shift_from_anchor",
    "degradation_delta_alpha", "mesh_status", "go_cell", "protocol_hash",
)


def _mesh_dict(mesh):
    return {"mesh_id": mesh.mesh_id, "nx": mesh.nx, "ny": mesh.ny, "seed": mesh.seed}


def protocol_payload():
    return {
        "pe": PE,
        "n_ic": N_IC,
        "ic_seeds": list(IC_SEEDS),
        "alphas": ALPHAS.tolist(),
        "directions": {name: values.tolist() for name, values in DIRECTIONS},
        "working_meshes": [_mesh_dict(mesh) for mesh in MESH_PLAN],
        "anchor_mesh_id": ANCHOR_MESH_ID,
        "reference_mesh": _mesh_dict(REFERENCE_MESH),
        "reference_alpha": 1.0,
        "observation_grid_n": OBSERVATION_GRID_N,
        "library": LIBRARY_NAME,
        "library_terms": list(LIBRARY_TERMS),
        "noise_model": NOISE_MODEL,
        "sigma": SIGMA,
        "target_tpr": TARGET_TPR,
        "target_fpr": TARGET_FPR,
        "max_allowed_degradation": MAX_ALLOWED_DEGRADATION,
        "go_criterion": GO_CRITERION,
        "reporting_rule": REPORTING_RULE,
    }


def protocol_hash():
    encoded = json.dumps(protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_metadata():
    return {"cache_schema": CACHE_SCHEMA, "protocol_hash": protocol_hash(),
            **protocol_payload()}


def mesh_plan():
    return tuple(MESH_PLAN)


def directions():
    return tuple((name, values.copy()) for name, values in DIRECTIONS)


def solve_counts():
    return {"reference": N_IC, "working": len(MESH_PLAN) * len(ALPHAS) * N_IC,
            "total": N_IC + len(MESH_PLAN) * len(ALPHAS) * N_IC}


def _expected_shapes():
    return ((N_IC, OBSERVATION_GRID_N, OBSERVATION_GRID_N),
            (len(MESH_PLAN), len(ALPHAS), N_IC,
             OBSERVATION_GRID_N, OBSERVATION_GRID_N))


def _validate_clean_fields(fields):
    if set(fields) != {"reference_grids", "working_grids"}:
        raise ValueError("clean fields must contain reference_grids and working_grids")
    reference = np.asarray(fields["reference_grids"], dtype=float)
    working = np.asarray(fields["working_grids"], dtype=float)
    expected_reference, expected_working = _expected_shapes()
    if reference.shape != expected_reference or working.shape != expected_working:
        raise ValueError(f"clean-field shapes must be {expected_reference} and {expected_working}")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(working)):
        raise ValueError("clean fields contain non-finite values")
    return {"reference_grids": reference, "working_grids": working}


def save_clean_cache(path, fields):
    fields = _validate_clean_fields(fields)
    path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    metadata_json = json.dumps(cache_metadata(), sort_keys=True, separators=(",", ":"))
    handle = tempfile.NamedTemporaryFile(dir=directory, prefix=".thermal_mesh_",
                                         suffix=".npz", delete=False)
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
        required = {"metadata_json", "reference_grids", "working_grids"}
        if set(archive.files) != required:
            raise ValueError(f"cache members must be exactly {sorted(required)}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != cache_metadata():
            raise ValueError("clean-field cache metadata does not match the frozen protocol")
        fields = {name: np.array(archive[name], copy=True)
                  for name in ("reference_grids", "working_grids")}
    return _validate_clean_fields(fields)


def solve_clean_fields():
    ics = [HC.make_thermal_ic(seed) for seed in IC_SEEDS]
    reference_mesh = HC.make_channel_mesh(REFERENCE_MESH.nx, REFERENCE_MESH.ny,
                                          REFERENCE_MESH.seed)
    reference_grids = np.asarray(HC.reference_channel_grids(
        ics, fine_mesh=reference_mesh, Pe=PE), dtype=float)
    working_grids = np.empty(_expected_shapes()[1], dtype=float)
    a_th = HC.thermal_diffusivity(PE)
    for mesh_index, mesh in enumerate(MESH_PLAN):
        pts, elems, tags = HC.make_channel_mesh(mesh.nx, mesh.ny, mesh.seed)
        geom = HC.channel_mesh_geometry(pts, elems, a_th)
        for alpha_index, alpha in enumerate(ALPHAS):
            for ic_index, ic in enumerate(ics):
                temperature = HC.assemble_channel(
                    "supg", pts, elems, tags, PE, ic=ic, alpha=float(alpha), geom=geom)
                working_grids[mesh_index, alpha_index, ic_index] = RS.to_normalized_grid(
                    pts, temperature, OBSERVATION_GRID_N)
    return _validate_clean_fields({"reference_grids": reference_grids,
                                   "working_grids": working_grids})


def load_or_solve_clean_fields(cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        return load_clean_cache(cache_path), "loaded"
    fields = solve_clean_fields()
    if cache_path is not None:
        save_clean_cache(cache_path, fields)
        return fields, "saved"
    return fields, "disabled"


def noise_seed(alpha_index, ic_index):
    return 864_000 + 100 * int(alpha_index) + int(ic_index)


def cell_seed(mesh_index, direction_index, alpha_index):
    return 975_000 + 10_000 * int(mesh_index) + 1_000 * int(direction_index) + int(alpha_index)


def extract_signatures(fields):
    fields = _validate_clean_fields(fields)
    reference = fields["reference_grids"]
    working = fields["working_grids"]
    signatures = {}
    for mesh_index, mesh in enumerate(MESH_PLAN):
        by_alpha = {}
        for alpha_index, alpha in enumerate(ALPHAS):
            population = np.empty((N_IC, len(LIBRARY_TERMS)), dtype=float)
            for ic_index in range(N_IC):
                observed = RS.add_structured_noise(
                    working[mesh_index, alpha_index, ic_index], NOISE_MODEL,
                    noise_seed(alpha_index, ic_index), SIGMA)
                population[ic_index] = RS.signature(
                    observed, reference[ic_index], LIBRARY_TERMS)
            by_alpha[float(alpha)] = population
        signatures[mesh.mesh_id] = by_alpha
    return signatures


def paired_detection(nominal, changed, seed):
    nominal = np.asarray(nominal, dtype=float)
    changed = np.asarray(changed, dtype=float)
    if nominal.shape != changed.shape or nominal.shape[0] != N_IC:
        raise ValueError("nominal and changed signatures must be paired over the same 60 ICs")
    return CC.split_conformal_detection(nominal, changed, seed=int(seed))


def direction_result(populations, alpha_values, mesh_index, direction_index):
    alpha_values = np.asarray(alpha_values, dtype=float)
    changed_alphas = alpha_values[~np.isclose(alpha_values, 1.0)]
    deltas = np.abs(changed_alphas - 1.0)
    nominal = populations[1.0]
    cells = [paired_detection(
        nominal, populations[float(alpha)],
        cell_seed(mesh_index, direction_index, alpha_index))
        for alpha_index, alpha in enumerate(changed_alphas)]
    tpr = np.array([cell["tpr"] for cell in cells])
    indicators = np.vstack([cell["detect"] for cell in cells])
    limit = CC.sampled_limit(deltas, tpr, TARGET_TPR)
    interval = CC.bootstrap_limit(
        deltas, indicators, TARGET_TPR,
        seed=cell_seed(mesh_index, direction_index, 900))
    return {
        "alphas": changed_alphas, "deltas": deltas, "tpr": tpr,
        "tpr_ci": np.array([cell["tpr_ci"] for cell in cells]),
        "fpr": np.array([cell["fpr"] for cell in cells]),
        "fpr_ci": np.array([cell["fpr_ci"] for cell in cells]),
        "limit": limit, "limit_ci": interval["limit_ci"],
        "censored_fraction": interval["censored_fraction"], "n_ic": N_IC,
    }


def evaluate(signatures):
    results = []
    for mesh_index, mesh in enumerate(MESH_PLAN):
        for direction_index, (direction, alpha_values) in enumerate(DIRECTIONS):
            result = direction_result(signatures[mesh.mesh_id], alpha_values,
                                      mesh_index, direction_index)
            result.update({"mesh": mesh, "direction": direction})
            results.append(result)
    return results


def classify_limits(results):
    anchors = {result["direction"]: result["limit"] for result in results
               if result["mesh"].mesh_id == ANCHOR_MESH_ID}
    if set(anchors) != {name for name, _ in DIRECTIONS}:
        raise ValueError("results need one anchor limit in each direction")
    classified = []
    for result in results:
        anchor = anchors[result["direction"]]
        limit = result["limit"]
        if result["mesh"].mesh_id == ANCHOR_MESH_ID:
            status = "anchor" if limit is not None else "anchor_censored"
            shift = 0.0 if limit is not None else None
            go_cell = limit is not None
        elif limit is None:
            status, shift, go_cell = "censored", None, False
        elif anchor is None:
            status, shift, go_cell = "anchor_censored", None, False
        else:
            shift = float(limit - anchor)
            if shift > MAX_ALLOWED_DEGRADATION + 1.0e-12:
                status, go_cell = "degraded", False
            elif shift < -1.0e-12:
                status, go_cell = "improved", True
            else:
                status, go_cell = "preserved", True
        classified.append({**result, "anchor_limit": anchor, "limit_shift": shift,
                           "degradation": None if shift is None else max(shift, 0.0),
                           "mesh_status": status, "go_cell": go_cell})
    return classified


def verdict(classified):
    return bool(classified) and all(result["go_cell"] for result in classified)


def _limit_text(value):
    return CC.limit_text(value, float(np.max(np.abs(ALPHAS - 1.0))))


def csv_rows(classified, mesh_sizes=None):
    mesh_sizes = {} if mesh_sizes is None else dict(mesh_sizes)
    rows = []
    digest = protocol_hash()
    for result in classified:
        mesh = result["mesh"]
        nodes, elements = mesh_sizes.get(mesh.mesh_id, ("", ""))
        common = {
            "mesh_id": mesh.mesh_id, "solver_nx": mesh.nx, "solver_ny": mesh.ny,
            "solver_seed": mesh.seed, "solver_nodes": nodes, "solver_elements": elements,
            "observation_grid_n": OBSERVATION_GRID_N,
            "reference_nx": REFERENCE_MESH.nx, "reference_ny": REFERENCE_MESH.ny,
            "reference_seed": REFERENCE_MESH.seed, "direction": result["direction"],
            "n_ic": N_IC, "noise_model": NOISE_MODEL, "sigma": f"{SIGMA:.2f}",
            "library": LIBRARY_NAME, "n_terms": len(LIBRARY_TERMS),
            "fpr_target": f"{TARGET_FPR:.2f}", "protocol_hash": digest,
        }
        rows.append({"type": "nominal", "alpha": "1.0", "delta_alpha": "0.0",
                     **common})
        for index, (alpha, delta) in enumerate(zip(result["alphas"], result["deltas"])):
            rows.append({
                "type": "curve", "alpha": f"{alpha:.1f}",
                "delta_alpha": f"{delta:.1f}", "tpr": f"{result['tpr'][index]:.6f}",
                "tpr_lo": f"{result['tpr_ci'][index][0]:.6f}",
                "tpr_hi": f"{result['tpr_ci'][index][1]:.6f}",
                "fpr_measured": f"{result['fpr'][index]:.6f}",
                "fpr_lo": f"{result['fpr_ci'][index][0]:.6f}",
                "fpr_hi": f"{result['fpr_ci'][index][1]:.6f}", **common})
        lo, hi = result["limit_ci"]
        rows.append({
            "type": "limit", "detection_limit_delta_alpha": _limit_text(result["limit"]),
            "limit_lo": _limit_text(lo), "limit_hi": _limit_text(hi),
            "limit_censored_fraction": f"{result['censored_fraction']:.4f}",
            "anchor_limit_delta_alpha": _limit_text(result["anchor_limit"]),
            "limit_shift_from_anchor": ("" if result["limit_shift"] is None
                                        else f"{result['limit_shift']:.1f}"),
            "degradation_delta_alpha": ("" if result["degradation"] is None
                                         else f"{result['degradation']:.1f}"),
            "mesh_status": result["mesh_status"],
            "go_cell": "1" if result["go_cell"] else "0", **common})
    return rows


def write_csv(rows, path=CSV_PATH):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mesh_sizes():
    sizes = {}
    for mesh in MESH_PLAN:
        pts, elems, _ = HC.make_channel_mesh(mesh.nx, mesh.ny, mesh.seed)
        sizes[mesh.mesh_id] = (len(pts), len(elems))
    return sizes


def print_protocol(cache_path):
    counts = solve_counts()
    print("=" * 96)
    print("THERMAL SOLVER-MESH SENSITIVITY (FROZEN BEFORE RESULTS)")
    print("=" * 96)
    print(f"Pe={PE:.0f}; IC seeds={IC_SEEDS[0]}..{IC_SEEDS[-1]}; "
          f"alpha={ALPHAS.tolist()}; sigma={SIGMA:.2f} {NOISE_MODEL} noise")
    print("Working FEM meshes: " + ", ".join(
        f"{mesh.mesh_id}={mesh.nx}x{mesh.ny} seed {mesh.seed}" for mesh in MESH_PLAN))
    print(f"Fixed nominal baseline FEM mesh: {REFERENCE_MESH.nx}x{REFERENCE_MESH.ny} "
          f"seed {REFERENCE_MESH.seed}, alpha=1.0")
    print(f"Observation grid: n={OBSERVATION_GRID_N}x{OBSERVATION_GRID_N} for every FEM mesh; "
          f"signature library={LIBRARY_NAME} ({len(LIBRARY_TERMS)} terms)")
    print(f"Detector: measured split-conformal FPR on untouched test ICs; target/bound={TARGET_FPR:.2f}; "
          f"target TPR={TARGET_TPR:.2f}")
    print(f"GO criterion: {GO_CRITERION}.")
    print(f"Reporting rule: {REPORTING_RULE}.")
    print(f"Clean-field cache: {cache_path if cache_path is not None else 'disabled'}")
    print(f"Solve count without cache: {counts['reference']} fixed-reference + "
          f"{counts['working']} working = {counts['total']}")
    print(f"Protocol SHA256: {protocol_hash()}")


def print_results(classified):
    print("\nmesh                 direction   limit  anchor  shift  status       measured FPR [min,max]")
    for result in classified:
        shift = "--" if result["limit_shift"] is None else f"{result['limit_shift']:+.1f}"
        print(f"{result['mesh'].mesh_id:<20s} {result['direction']:>10s} "
              f"{_limit_text(result['limit']):>7s} {_limit_text(result['anchor_limit']):>7s} "
              f"{shift:>6s} {result['mesh_status']:<12s} "
              f"[{result['fpr'].min():.3f},{result['fpr'].max():.3f}]")
    print(f"\n[{'GO' if verdict(classified) else 'CHECK'} / VERDICT] {GO_CRITERION}")


def run(write=True, cache_path=None):
    started = time.perf_counter()
    print_protocol(cache_path)
    fields, cache_state = load_or_solve_clean_fields(cache_path)
    print(f"\nClean fields: cache {cache_state}")
    signatures = extract_signatures(fields)
    classified = classify_limits(evaluate(signatures))
    rows = csv_rows(classified, _mesh_sizes())
    output = write_csv(rows) if write else None
    print_results(classified)
    if output is not None:
        print(f"CSV -> {output}")
    else:
        print("CSV disabled")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.1f}")
    return {"results": classified, "rows": rows, "csv": output,
            "cache_state": cache_state, "runtime_seconds": elapsed,
            "go": verdict(classified)}


def build_parser():
    parser = argparse.ArgumentParser(description="direct thermal solver-mesh sensitivity")
    parser.add_argument("--cache", nargs="?", const=CACHE_PATH, default=None,
                        help="load or create the optional clean-field NPZ cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="execute the experiment without writing the CSV")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(write=not args.dry_run, cache_path=args.cache)


if __name__ == "__main__":
    main()
