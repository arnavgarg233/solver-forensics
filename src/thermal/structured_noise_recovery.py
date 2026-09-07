#!/usr/bin/env python3
"""Confirmatory recovery experiment for structured thermal observation noise.

The protocol is fixed before any outcome is produced. It uses the Pe=100 heated
channel, the full 60-IC population, the 64 by 64 normalized observation grid,
the base5 derivative library, 1%-RMS noise, and five independent acquisitions
for every (noise model, alpha, IC) case. The alpha grid is 1.0 to 0.5 and 1.0 to
1.5 in steps of 0.1, reported separately as weakening and strengthening.

The existing decision-averaging baseline is preserved exactly: extract one
signature from each noisy field, run the shared split-conformal detector once
per acquisition, then average its indicators within each IC. The recovery arm
uses the same five noisy fields and averages them within a case before any
finite-difference derivative or signature is computed. No field is averaged
across ICs, alpha values, noise models, or reference cases.

The endpoint is the smallest sampled positive |delta alpha| whose point TPR is
at least 0.95, with right-censoring above 0.5. False alarms use the unchanged
strict split-conformal endpoint in cross_conformal.py, including its exact 0.05
per-IC bound and measured test-IC FPR.

The predeclared GO criteria are:
  1. White-noise positive control: in both directions, both limits resolve and
     field averaging is no worse than decision averaging.
  2. Structured recovery: for low-pass, gradient-weighted, and multiplicative
     noise in both directions, the field-averaged limit resolves and is
     strictly smaller than the decision-averaged baseline limit. A censored
     baseline is treated as larger than every resolved sampled limit.
  3. Overall GO requires both criteria above.

An optional NPZ cache stores only the clean 64 by 64 reference and working
fields owned by this script. Importing the module writes nothing. Running the
script always executes the full fixed protocol and writes the CSV only after
all outcomes and criteria have been computed.
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

import cross_conformal as CC
import heated_channel as HC
import reviewer_sensitivity as RS

PE = 100.0
GRID_N = 64
LIBRARY = "base5"
LIBRARY_NAMES = RS.LIBRARIES[LIBRARY]
SIGMA = 0.01
N_IC = 60
N_ACQUISITIONS = 5
ALPHAS_WEAK = np.array((1.0, 0.9, 0.8, 0.7, 0.6, 0.5), dtype=float)
ALPHAS_STRONG = np.array((1.0, 1.1, 1.2, 1.3, 1.4, 1.5), dtype=float)
ALL_ALPHAS = np.array(sorted(set(ALPHAS_WEAK) | set(ALPHAS_STRONG)), dtype=float)
NOISE_MODELS = ("white", "lowpass", "gradient", "multiplicative")
STRUCTURED_MODELS = ("lowpass", "gradient", "multiplicative")
METHODS = ("decision_average", "field_average")
TARGET_TPR = 0.95
TARGET_FPR = CC.TARGET_FPR

CSV_PATH = os.path.join(_ROOT, "results", "tables", "structured_noise_recovery.csv")
DEFAULT_CACHE_PATH = os.path.join(
    _ROOT, "results", "cache", "structured_noise_recovery_fields.npz")

GO_CRITERIA = {
    "white_positive_control": (
        "Both white-noise direction limits resolve for both methods, and each "
        "field-averaged limit is no larger than its decision-averaged baseline."),
    "structured_recovery": (
        "All six structured-noise field-averaged direction limits resolve and "
        "are strictly smaller than their decision-averaged baselines; a censored "
        "baseline is larger than every resolved sampled limit."),
    "overall_go": "White positive control and structured recovery both pass.",
}

CSV_FIELDS = (
    "type", "noise", "control_role", "direction", "method", "alpha",
    "delta_alpha", "n_ic", "n_acquisitions", "detector_replicates", "grid_n",
    "library", "sigma", "target_tpr", "fpr_target", "tpr", "tpr_lo", "tpr_hi",
    "fpr_measured", "fpr_lo", "fpr_hi", "detection_limit_delta_alpha",
    "limit_lo", "limit_hi", "limit_censored_fraction", "criterion",
    "criterion_definition", "pass",
)

CACHE_SCHEMA_VERSION = 1
CACHE_KEYS = (
    "schema_version", "pe", "grid_n", "library", "sigma", "n_ic",
    "n_acquisitions", "alphas", "ic_seeds", "reference_grids", "working_grids",
)


def protocol():
    """Return a fresh copy of the fixed confirmatory protocol."""
    return {
        "pe": PE,
        "grid_n": GRID_N,
        "library": LIBRARY,
        "library_names": tuple(LIBRARY_NAMES),
        "sigma": SIGMA,
        "n_ic": N_IC,
        "n_acquisitions": N_ACQUISITIONS,
        "noise_models": tuple(NOISE_MODELS),
        "alphas_weak": ALPHAS_WEAK.copy(),
        "alphas_strong": ALPHAS_STRONG.copy(),
        "alphas": ALL_ALPHAS.copy(),
        "target_tpr": TARGET_TPR,
        "target_fpr": TARGET_FPR,
    }


def directions(cfg):
    """Direction labels and their independently evaluated alpha grids."""
    return (("weaken", cfg["alphas_weak"]),
            ("strengthen", cfg["alphas_strong"]))


def average_acquisitions(fields):
    """Average acquisitions within one case, before derivative extraction."""
    values = np.asarray(fields, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("acquisitions must have shape (n_acquisitions, n, n)")
    if values.shape[1] != values.shape[2]:
        raise ValueError("each acquisition must be a square field")
    if not np.all(np.isfinite(values)):
        raise ValueError("acquisitions contain a non-finite value")
    return np.mean(values, axis=0, dtype=np.float64)


def noisy_acquisitions(clean, model, alpha_index, ic_index, sigma=SIGMA,
                       n_acquisitions=N_ACQUISITIONS, grid_n=GRID_N):
    """Return the existing deterministic acquisitions for exactly one case."""
    clean = np.asarray(clean, dtype=float)
    if clean.ndim != 2 or clean.shape[0] != clean.shape[1]:
        raise ValueError("clean must be one square observation field")
    if model not in RS.NOISE_MODELS:
        raise ValueError(f"unknown noise model: {model}")
    if int(n_acquisitions) < 1:
        raise ValueError("n_acquisitions must be positive")
    model_index = RS.NOISE_MODELS.index(model)
    return np.stack([
        RS.add_structured_noise(
            clean, model,
            RS.noise_seed(grid_n, model_index, alpha_index, ic_index, replicate),
            sigma)
        for replicate in range(int(n_acquisitions))
    ])


def _validate_clean_arrays(reference_grids, working_grids, cfg):
    reference = np.asarray(reference_grids, dtype=float)
    working = np.asarray(working_grids, dtype=float)
    expected_ref = (int(cfg["n_ic"]), int(cfg["grid_n"]), int(cfg["grid_n"]))
    expected_work = (len(cfg["alphas"]),) + expected_ref
    if reference.shape != expected_ref:
        raise ValueError(f"reference_grids has shape {reference.shape}, expected {expected_ref}")
    if working.shape != expected_work:
        raise ValueError(f"working_grids has shape {working.shape}, expected {expected_work}")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(working)):
        raise ValueError("clean field arrays contain a non-finite value")
    return reference, working


def signature_arms(reference_grids, working_grids, cfg=None):
    """Build baseline and recovery signatures from the same noisy fields.

    The returned mapping is ``arm[method][noise][alpha]``. Baseline values are a
    list of five ``(n_ic, 5)`` arrays. Recovery values are a one-element list
    containing the signatures extracted after within-case field averaging. The
    one-element list lets both arms use ``RS.detection_curve`` unchanged.
    """
    cfg = protocol() if cfg is None else cfg
    reference, working = _validate_clean_arrays(reference_grids, working_grids, cfg)
    n_ic = int(cfg["n_ic"])
    n_acq = int(cfg["n_acquisitions"])
    names = tuple(cfg["library_names"])
    _, _, h = RS.normalized_axes(cfg["grid_n"])
    arms = {method: {model: {} for model in cfg["noise_models"]}
            for method in METHODS}

    for model in cfg["noise_models"]:
        for alpha_index, alpha in enumerate(cfg["alphas"]):
            baseline = np.empty((n_acq, n_ic, len(names)), dtype=float)
            recovered = np.empty((n_ic, len(names)), dtype=float)
            for ic_index in range(n_ic):
                acquisitions = noisy_acquisitions(
                    working[alpha_index, ic_index], model, alpha_index, ic_index,
                    sigma=cfg["sigma"], n_acquisitions=n_acq,
                    grid_n=cfg["grid_n"])
                for replicate in range(n_acq):
                    baseline[replicate, ic_index] = RS.signature(
                        acquisitions[replicate], reference[ic_index], names, h)
                recovered[ic_index] = RS.signature(
                    average_acquisitions(acquisitions), reference[ic_index], names, h)
            arms["decision_average"][model][float(alpha)] = list(baseline)
            arms["field_average"][model][float(alpha)] = [recovered]
    return arms


def evaluate(arms, cfg=None):
    """Apply the unchanged direction-resolved split-conformal endpoint."""
    cfg = protocol() if cfg is None else cfg
    results = []
    library_index = tuple(RS.LIBRARIES).index(cfg["library"])
    for model in cfg["noise_models"]:
        model_index = RS.NOISE_MODELS.index(model)
        for direction_index, (direction, alphas) in enumerate(directions(cfg)):
            seed = RS.cell_seed(cfg["grid_n"], library_index, model_index,
                                direction_index)
            for method in METHODS:
                curve = RS.detection_curve(arms[method][model], alphas, seed)
                curve.update({"method": method, "noise": model,
                              "direction": direction})
                results.append(curve)
    return results


def _result_index(results):
    index = {(r["method"], r["noise"], r["direction"]): r for r in results}
    expected = len(METHODS) * len(NOISE_MODELS) * 2
    if len(index) != expected:
        raise ValueError(f"expected {expected} unique method/noise/direction results")
    return index


def assess_go(results):
    """Apply the frozen positive-control and structured-recovery criteria."""
    index = _result_index(results)
    white_checks = []
    for direction in ("weaken", "strengthen"):
        baseline = index[("decision_average", "white", direction)]["limit"]
        recovered = index[("field_average", "white", direction)]["limit"]
        white_checks.append(baseline is not None and recovered is not None
                            and recovered <= baseline)

    structured_checks = []
    for model in STRUCTURED_MODELS:
        for direction in ("weaken", "strengthen"):
            baseline = index[("decision_average", model, direction)]["limit"]
            recovered = index[("field_average", model, direction)]["limit"]
            structured_checks.append(
                recovered is not None and (baseline is None or recovered < baseline))

    assessment = {
        "white_positive_control": {
            "pass": bool(all(white_checks)),
            "definition": GO_CRITERIA["white_positive_control"],
        },
        "structured_recovery": {
            "pass": bool(all(structured_checks)),
            "definition": GO_CRITERIA["structured_recovery"],
        },
    }
    assessment["overall_go"] = {
        "pass": bool(assessment["white_positive_control"]["pass"]
                     and assessment["structured_recovery"]["pass"]),
        "definition": GO_CRITERIA["overall_go"],
    }
    return assessment


def _cache_metadata(cfg):
    return {
        "schema_version": np.array(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "pe": np.array(cfg["pe"], dtype=float),
        "grid_n": np.array(cfg["grid_n"], dtype=np.int64),
        "library": np.array(cfg["library"]),
        "sigma": np.array(cfg["sigma"], dtype=float),
        "n_ic": np.array(cfg["n_ic"], dtype=np.int64),
        "n_acquisitions": np.array(cfg["n_acquisitions"], dtype=np.int64),
        "alphas": np.asarray(cfg["alphas"], dtype=float),
        "ic_seeds": np.arange(1000, 1000 + int(cfg["n_ic"]), dtype=np.int64),
    }


def save_field_cache(path, reference_grids, working_grids, cfg=None):
    """Write the script-owned clean-field NPZ cache atomically."""
    cfg = protocol() if cfg is None else cfg
    reference, working = _validate_clean_arrays(reference_grids, working_grids, cfg)
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp.npz"
    payload = _cache_metadata(cfg)
    payload.update({"reference_grids": reference, "working_grids": working})
    np.savez(temporary, **payload)
    os.replace(temporary, path)
    return path


def _scalar(archive, key):
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"cache metadata {key} must be scalar")
    return value.item()


def load_field_cache(path, cfg=None):
    """Load a cache only when its complete protocol contract matches exactly."""
    cfg = protocol() if cfg is None else cfg
    with np.load(os.fspath(path), allow_pickle=False) as archive:
        if set(archive.files) != set(CACHE_KEYS):
            raise ValueError("field cache keys do not match the recovery cache schema")
        expected = _cache_metadata(cfg)
        scalar_keys = ("schema_version", "pe", "grid_n", "library", "sigma",
                       "n_ic", "n_acquisitions")
        for key in scalar_keys:
            actual = _scalar(archive, key)
            wanted = np.asarray(expected[key]).item()
            if actual != wanted:
                raise ValueError(f"field cache {key}={actual!r}, expected {wanted!r}")
        for key in ("alphas", "ic_seeds"):
            if not np.array_equal(np.asarray(archive[key]), expected[key]):
                raise ValueError(f"field cache {key} does not match the protocol")
        reference = np.array(archive["reference_grids"], dtype=float, copy=True)
        working = np.array(archive["working_grids"], dtype=float, copy=True)
    return _validate_clean_arrays(reference, working, cfg)


def solve_clean_grids(cfg=None):
    """Run the validated solves once and interpolate the clean fields to n=64."""
    cfg = protocol() if cfg is None else cfg
    fields = RS.solve_clean_fields(cfg)
    reference = np.stack([
        RS.to_normalized_grid(fields["ref_pts"], values, cfg["grid_n"])
        for values in fields["ref_nodal"]
    ])
    working = np.stack([
        np.stack([
            RS.to_normalized_grid(fields["pts"], values, cfg["grid_n"])
            for values in fields["work_nodal"][float(alpha)]
        ])
        for alpha in cfg["alphas"]
    ])
    return _validate_clean_arrays(reference, working, cfg)


def load_or_solve_clean_grids(cache_path=None, cfg=None):
    """Use the optional owned cache, or run and optionally cache full solves."""
    cfg = protocol() if cfg is None else cfg
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Clean fields: loading cache {os.path.abspath(cache_path)}")
        return load_field_cache(cache_path, cfg)
    fields = solve_clean_grids(cfg)
    if cache_path is not None:
        saved = save_field_cache(cache_path, *fields, cfg)
        print(f"Clean fields: cache written {saved}")
    return fields


def _max_delta(cfg):
    return float(np.max(np.abs(np.asarray(cfg["alphas"], dtype=float) - 1.0)))


def output_rows(results, assessment, cfg=None):
    """Create the fixed CSV schema without writing it."""
    cfg = protocol() if cfg is None else cfg
    rows = []
    max_delta = _max_delta(cfg)
    for result in results:
        common = {
            "noise": result["noise"],
            "control_role": ("positive_control" if result["noise"] == "white"
                             else "confirmatory"),
            "direction": result["direction"],
            "method": result["method"],
            "n_ic": result["n_ic"],
            "n_acquisitions": cfg["n_acquisitions"],
            "detector_replicates": result["n_noise"],
            "grid_n": cfg["grid_n"],
            "library": cfg["library"],
            "sigma": f"{cfg['sigma']:.2f}",
            "target_tpr": f"{cfg['target_tpr']:.2f}",
            "fpr_target": f"{cfg['target_fpr']:.2f}",
        }
        for j, (alpha, delta) in enumerate(zip(result["alphas"], result["deltas"])):
            rows.append({
                "type": "curve", "alpha": f"{alpha:.1f}",
                "delta_alpha": f"{delta:.1f}",
                "tpr": f"{result['tpr'][j]:.6f}",
                "tpr_lo": f"{result['tpr_ci'][j][0]:.6f}",
                "tpr_hi": f"{result['tpr_ci'][j][1]:.6f}",
                "fpr_measured": f"{result['fpr'][j]:.6f}",
                "fpr_lo": f"{result['fpr_ci'][j][0]:.6f}",
                "fpr_hi": f"{result['fpr_ci'][j][1]:.6f}",
                **common,
            })
        lo, hi = result["limit_ci"]
        rows.append({
            "type": "limit",
            "detection_limit_delta_alpha": CC.limit_text(result["limit"], max_delta),
            "limit_lo": CC.limit_text(lo, max_delta),
            "limit_hi": CC.limit_text(hi, max_delta),
            "limit_censored_fraction": f"{result['censored_fraction']:.4f}",
            **common,
        })
    for name in GO_CRITERIA:
        rows.append({
            "type": "criterion", "criterion": name,
            "criterion_definition": assessment[name]["definition"],
            "pass": "1" if assessment[name]["pass"] else "0",
            "n_ic": cfg["n_ic"], "n_acquisitions": cfg["n_acquisitions"],
            "grid_n": cfg["grid_n"], "library": cfg["library"],
            "sigma": f"{cfg['sigma']:.2f}",
            "target_tpr": f"{cfg['target_tpr']:.2f}",
            "fpr_target": f"{cfg['target_fpr']:.2f}",
        })
    return rows


def write_results(rows, path=CSV_PATH):
    """Write completed results. The full driver is the only production caller."""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_predeclaration(cfg):
    """Print the fixed protocol and GO criteria before any outcomes are computed."""
    print("=" * 104)
    print("CONFIRMATORY STRUCTURED-NOISE RECOVERY")
    print("=" * 104)
    print(f"Protocol: Pe={cfg['pe']:.0f}; n={cfg['grid_n']}; {cfg['library']}; "
          f"{cfg['n_ic']} ICs; {cfg['n_acquisitions']} acquisitions; "
          f"sigma={cfg['sigma']:.2f} field-relative RMS")
    print(f"Alpha weakening: {[float(v) for v in cfg['alphas_weak']]}")
    print(f"Alpha strengthening: {[float(v) for v in cfg['alphas_strong']]}")
    print(f"Endpoint: smallest sampled positive |delta alpha| at TPR >= "
          f"{cfg['target_tpr']:.2f}; measured test-IC FPR with exact per-IC bound "
          f"{cfg['target_fpr']:.2f}")
    print("Baseline: signatures and decisions per acquisition, then within-IC decision average")
    print("Recovery: within-case field average, then derivatives, signature, and decision")
    print("Predeclared GO criteria:")
    for name, definition in GO_CRITERIA.items():
        print(f"  {name}: {definition}")


def _print_outcomes(results, assessment, cfg):
    max_delta = _max_delta(cfg)
    print("\nDIRECTION-RESOLVED SAMPLED LIMITS")
    for result in results:
        print(f"  {result['noise']:<14s} {result['direction']:<10s} "
              f"{result['method']:<17s} "
              f"{CC.limit_text(result['limit'], max_delta):>4s}")
    print("\nPREDECLARED CRITERIA")
    for name in GO_CRITERIA:
        print(f"  {name:<24s} {'PASS' if assessment[name]['pass'] else 'FAIL'}")


def run(cache_path=None):
    """Run the full fixed experiment and write results only after completion."""
    started = time.perf_counter()
    cfg = protocol()
    if cfg["n_ic"] != HC.N_IC:
        raise RuntimeError(f"expected {HC.N_IC} validated ICs, got {cfg['n_ic']}")
    print_predeclaration(cfg)
    reference, working = load_or_solve_clean_grids(cache_path, cfg)
    arms = signature_arms(reference, working, cfg)
    results = evaluate(arms, cfg)
    assessment = assess_go(results)
    rows = output_rows(results, assessment, cfg)
    _print_outcomes(results, assessment, cfg)
    path = write_results(rows, CSV_PATH)
    elapsed = time.perf_counter() - started
    print(f"\nresults -> {path}")
    print(f"RUNTIME_SECONDS: {elapsed:.1f}")
    return {"config": cfg, "results": results, "assessment": assessment,
            "rows": rows, "path": path, "runtime_seconds": elapsed}


def build_parser():
    parser = argparse.ArgumentParser(
        description="full confirmatory recovery experiment for structured thermal noise")
    parser.add_argument(
        "--cache", nargs="?", const=DEFAULT_CACHE_PATH, default=None,
        help="reuse or create the script-owned clean-field NPZ cache; optionally give a path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(cache_path=args.cache)


if __name__ == "__main__":
    main()
