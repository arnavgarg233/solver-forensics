#!/usr/bin/env python3
"""Widened-range measurement of the structured-noise detection limit.

This is a separate experiment from ``structured_noise_recovery``. That one asked
whether field averaging recovers structured-noise detection; it FAILED its
predeclared criteria, including its white-noise positive control, and its result
is preserved unchanged. Nothing here reinterprets it.

The question here is different and purely descriptive. In the published sweep the
alpha grid stopped at |delta alpha| = 0.5, so every structured-noise cell was
reported as right-censored above 0.5. That bound is a property of where we stopped
looking, not of the detector. This experiment widens the grid to |delta alpha| <=
0.9 weakening and <= 1.0 strengthening and measures where, if anywhere, the
structured-noise limits actually fall.

The detector is the frozen decision-averaging split-conformal rule that passed its
own white-noise control. Field averaging is not used. Everything else, the Pe=100
channel, the 60-IC population, the 64x64 grid, the base5 library, 1% RMS noise and
five acquisitions per case, is unchanged.

Predeclared criteria, fixed before any outcome:
  1. solver_health: every clean solve at every sampled alpha is finite.
  2. positive_control: both white-noise limits resolve and each lies within one
     sampled step (0.1) of the published 0.4, confirming the widened grid did not
     disturb the baseline.
  3. resolution: at least one structured (noise model, direction) cell resolves a
     finite limit inside the widened range.
  Criteria 1 and 2 decide whether the measurement is trustworthy. Criterion 3
  decides whether a number can be quoted; if it fails, the outcome is a tighter
  censoring bound, which is reported as such and is not a failure of the paper.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))

import reviewer_sensitivity as RS
import structured_noise_recovery as SNR

PROTOCOL_ID = "structured-noise-extended-range-v1"
METHOD = "decision_average"
PUBLISHED_WHITE_LIMIT = 0.4
STEP = 0.1
ALPHAS_WEAK = np.round(np.arange(1.0, 0.05, -0.1), 10)
ALPHAS_STRONG = np.round(np.arange(1.0, 2.05, 0.1), 10)
CSV_PATH = os.path.join(_ROOT, "results", "tables", "structured_noise_extended.csv")
CACHE_PATH = os.path.join(_ROOT, "results", "cache", "structured_noise_extended_fields.npz")

CRITERIA = {
    "solver_health": "every clean solve at every sampled alpha is finite",
    "positive_control": (
        "both white-noise limits resolve and each lies within one sampled step "
        f"({STEP}) of the published {PUBLISHED_WHITE_LIMIT}"),
    "resolution": (
        "at least one structured noise model and direction resolves a finite "
        "limit inside the widened range"),
}


def config():
    cfg = SNR.protocol()
    cfg["alphas_weak"] = ALPHAS_WEAK.copy()
    cfg["alphas_strong"] = ALPHAS_STRONG.copy()
    cfg["alphas"] = np.array(sorted(set(ALPHAS_WEAK) | set(ALPHAS_STRONG)), dtype=float)
    return cfg


def protocol_hash(cfg):
    payload = {
        "protocol_id": PROTOCOL_ID, "method": METHOD,
        "alphas_weak": [float(a) for a in cfg["alphas_weak"]],
        "alphas_strong": [float(a) for a in cfg["alphas_strong"]],
        "noise_models": list(cfg["noise_models"]), "n_ic": int(cfg["n_ic"]),
        "grid_n": int(cfg["grid_n"]), "library": cfg["library"],
        "sigma": float(cfg["sigma"]), "target_tpr": float(cfg["target_tpr"]),
        "criteria": CRITERIA,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assess(results, health_ok, cfg):
    idx = {(r["noise"], r["direction"]): r for r in results if r["method"] == METHOD}
    white = [idx[("white", d)]["limit"] for d in ("weaken", "strengthen")]
    pc = all(l is not None and abs(l - PUBLISHED_WHITE_LIMIT) <= STEP + 1e-9 for l in white)
    structured = [idx[(m, d)]["limit"] for m in SNR.STRUCTURED_MODELS
                  for d in ("weaken", "strengthen")]
    res = any(l is not None for l in structured)
    return {
        "solver_health": {"pass": bool(health_ok), "definition": CRITERIA["solver_health"]},
        "positive_control": {"pass": bool(pc), "definition": CRITERIA["positive_control"]},
        "resolution": {"pass": bool(res), "definition": CRITERIA["resolution"]},
    }


def run(cache_path=None):
    cfg = config()
    phash = protocol_hash(cfg)
    max_weak = float(np.max(np.abs(cfg["alphas_weak"] - 1.0)))
    max_strong = float(np.max(np.abs(cfg["alphas_strong"] - 1.0)))
    print("=" * 96)
    print("STRUCTURED-NOISE LIMIT ON A WIDENED ALPHA RANGE (FROZEN BEFORE RESULTS)")
    print("=" * 96)
    print(f"Protocol: {PROTOCOL_ID}   hash {phash}")
    print(f"Detector: {METHOD} (the arm that passed its own white-noise control)")
    print(f"Weakening alphas   : {[round(float(a),2) for a in cfg['alphas_weak']]}  max |da| {max_weak:.1f}")
    print(f"Strengthening alphas: {[round(float(a),2) for a in cfg['alphas_strong']]}  max |da| {max_strong:.1f}")
    print(f"Published range stopped at |da| = 0.5; every structured cell was censored there.")
    for k, v in CRITERIA.items():
        print(f"  [{k}] {v}")
    t0 = time.time()
    reference, working = SNR.load_or_solve_clean_grids(cache_path, cfg)
    health_ok = bool(np.all(np.isfinite(reference)) and np.all(np.isfinite(working)))
    print(f"solver health: all clean fields finite = {health_ok}")
    arms = SNR.signature_arms(reference, working, cfg)
    results = SNR.evaluate(arms, cfg)
    assessment = assess(results, health_ok, cfg)

    print("\nDIRECTION-RESOLVED LIMITS (decision averaging)")
    rows = []
    for r in results:
        if r["method"] != METHOD:
            continue
        cap = max_weak if r["direction"] == "weaken" else max_strong
        lim = r["limit"]
        shown = f"{lim:.1f}" if lim is not None else f">{cap:.1f}"
        was = "0.4" if r["noise"] == "white" else ">0.5"
        print(f"  {r['noise']:15s} {r['direction']:11s} published {was:>5s}   widened {shown:>5s}")
        rows.append({"protocol_id": PROTOCOL_ID, "protocol_hash": phash, "method": METHOD,
                     "noise": r["noise"], "direction": r["direction"],
                     "published_limit": was, "widened_limit": shown,
                     "limit_resolved": lim is not None,
                     "limit_delta_alpha": "" if lim is None else f"{lim:.4f}",
                     "max_sampled_delta_alpha": f"{cap:.1f}",
                     "n_ic": cfg["n_ic"], "sigma": cfg["sigma"], "grid_n": cfg["grid_n"],
                     "library": cfg["library"], "target_tpr": cfg["target_tpr"]})
    print("\nPREDECLARED CRITERIA")
    for k, v in assessment.items():
        print(f"  {k:18s} {'PASS' if v['pass'] else 'FAIL'}")
    for r in rows:
        r["solver_health_pass"] = assessment["solver_health"]["pass"]
        r["positive_control_pass"] = assessment["positive_control"]["pass"]
        r["resolution_pass"] = assessment["resolution"]["pass"]
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nresults -> {CSV_PATH}")
    print(f"RUNTIME_SECONDS: {time.time()-t0:.1f}")
    return {"rows": rows, "assessment": assessment, "protocol_hash": phash}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", nargs="?", const=CACHE_PATH, default=None)
    a = ap.parse_args(argv)
    run(cache_path=a.cache)


if __name__ == "__main__":
    main()
