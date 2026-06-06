import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A                              # verified FEM + signature machinery
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
from itertools import combinations


N_IC = 40
N_REPEATS = 5
SIGMA = 0.01
WORKING_BASE = (28, 2026)
REFERENCE_BASE = (96, 7)
IC_BASES = (1000, 2000, 3000, 4000, 5000)
WORKING_GRID_SPECS = ((24, 11), (26, 12), (28, 13), (30, 14), (32, 15))
REFERENCE_RESOLUTIONS = (80, 88, 96, 104, 112)
MEASUREMENTS = ("M1_PRESENCE", "M2_SILENT_TAU")
PAIR_SCHEMES = {
    "M1_PRESENCE": ("galerkin", "supg"),
    "M2_SILENT_TAU": ("supg", "supg_half"),
}


def sig_from_grid(Us, Ur):
    R = Us - Ur
    Dlib, sl = A._fd_library(Us)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm > 0 else c


def ic_block(base):
    return [A.make_ic(base + i) for i in range(N_IC)]


def working_mesh(n_side, seed):
    pts, elems, on_bnd = A.make_mesh(n_side, seed)
    geom = A.mesh_geometry(pts, elems)
    return dict(pts=pts, elems=elems, on_bnd=on_bnd, geom=geom)


def clean_solver_grids(mesh, ics):
    """Solve each working configuration once, before any noise is added."""
    pts, elems, on_bnd, geom = (mesh[k] for k in ("pts", "elems", "on_bnd", "geom"))
    clean = {"galerkin": [], "supg": [], "supg_half": []}
    for ic in ics:
        u, _ = A.assemble("galerkin", pts, elems, on_bnd, ic, geom=geom)
        clean["galerkin"].append(A._to_grid(pts, u))
        u, _ = A.assemble("supg", pts, elems, on_bnd, ic, tau_scale=1.0, geom=geom)
        clean["supg"].append(A._to_grid(pts, u))
        u, _ = A.assemble("supg", pts, elems, on_bnd, ic, tau_scale=0.5, geom=geom)
        clean["supg_half"].append(A._to_grid(pts, u))
    return {name: np.asarray(grids) for name, grids in clean.items()}


def reference_grids(ics, n_side, seed=7):
    ref_pts, ref_elems, ref_bnd = A.make_mesh(n_side, seed)
    return A.reference_grids(ics, ref_pts, ref_elems, ref_bnd)


def split_clean(clean, n_blocks):
    return [
        {name: grids[i * N_IC:(i + 1) * N_IC] for name, grids in clean.items()}
        for i in range(n_blocks)
    ]


def split_refs(refs, n_blocks):
    return [refs[i * N_IC:(i + 1) * N_IC] for i in range(n_blocks)]


def signatures_for(clean, refs, noise_seed):
    """Recover signatures after adding noise to cached solver grids."""
    out = {}
    for scheme_index, scheme in enumerate(("galerkin", "supg", "supg_half")):
        rng = np.random.default_rng(noise_seed + 1009 * scheme_index)
        sigs = []
        for k, Us in enumerate(clean[scheme]):
            rms = np.sqrt(np.mean(Us**2))
            Us_noisy = Us + SIGMA * rms * rng.standard_normal(Us.shape)
            sigs.append(sig_from_grid(Us_noisy, refs[k]))
        out[scheme] = np.asarray(sigs)
    return out


def pair_metrics(signatures, measurement, perm_seed):
    left, right = PAIR_SCHEMES[measurement]
    n = len(signatures[left])
    X = A.feats(np.vstack([signatures[left], signatures[right]]))
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    groups = np.r_[np.arange(n), np.arange(n)]
    accuracy = float(A.cv_acc(X, y, groups))
    floor = float(A.perm_floor(X, y, groups, perm_seed, reps=30))
    return accuracy, floor


def mean_direction(C):
    direction = np.mean(C, axis=0)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


def pairwise_abs_cos(directions):
    values = [abs(float(np.dot(directions[i], directions[j])))
              for i, j in combinations(range(len(directions)), 2)]
    return float(np.mean(np.clip(values, 0.0, 1.0)))


def evaluate_repeat(clean, refs, noise_seed, perm_seed):
    signatures = signatures_for(clean, refs, noise_seed)
    metrics = {}
    for measurement_index, measurement in enumerate(MEASUREMENTS):
        accuracy, floor = pair_metrics(signatures, measurement, perm_seed + measurement_index)
        metrics[measurement] = dict(accuracy=accuracy, floor=floor)
    return metrics, mean_direction(signatures["supg"])


def summarize_factor(factor, repeats):
    summary = {}
    for measurement in MEASUREMENTS:
        accuracies = np.asarray([r["metrics"][measurement]["accuracy"] for r in repeats])
        floors = np.asarray([r["metrics"][measurement]["floor"] for r in repeats])
        mean_acc = float(np.mean(accuracies))
        sd_acc = float(np.std(accuracies))
        cv_acc = sd_acc / mean_acc if mean_acc != 0 else float("nan")
        directions = np.asarray([r["supg_direction"] for r in repeats])
        summary[measurement] = dict(
            n_repeats=len(repeats),
            mean_acc=mean_acc,
            sd_acc=sd_acc,
            cv_acc=float(cv_acc),
            sig_dir_cos_mean=pairwise_abs_cos(directions),
            floor_mean=float(np.mean(floors)),
            floor_sd=float(np.std(floors)),
        )
    return summary


def evaluate_factor(factor, scenarios, factor_index):
    print(f"\n[factor] {factor}: evaluating {N_REPEATS} repeats")
    repeats = []
    for repeat_index, (clean, refs, noise_seed) in enumerate(scenarios):
        metrics, direction = evaluate_repeat(
            clean,
            refs,
            noise_seed,
            300000 + factor_index * 10000 + repeat_index * 100,
        )
        repeats.append(dict(metrics=metrics, supg_direction=direction))
        print(f"  repeat {repeat_index + 1}/{N_REPEATS}: "
              f"M1={metrics['M1_PRESENCE']['accuracy']:.3f} "
              f"M2={metrics['M2_SILENT_TAU']['accuracy']:.3f}")
    return summarize_factor(factor, repeats)


def write_csv(results):
    csv_path = os.path.join(TAB, "repeatability.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["factor", "measurement", "n_repeats", "mean_acc", "sd_acc",
                         "cv_acc", "sig_dir_cos_mean"])
        for factor in results:
            for measurement in MEASUREMENTS:
                row = results[factor][measurement]
                writer.writerow([
                    factor,
                    measurement,
                    row["n_repeats"],
                    f"{row['mean_acc']:.6f}",
                    f"{row['sd_acc']:.6f}",
                    f"{row['cv_acc']:.6f}",
                    f"{row['sig_dir_cos_mean']:.6f}",
                ])
    return csv_path


def main():
    print("=" * 88)
    print("MEASUREMENT REPEATABILITY / REPRODUCIBILITY")
    print("measurand: solver configuration | signal: modified-equation signature direction")
    print("=" * 88)
    print(f"M1=PRESENCE (galerkin vs supg tau=1.0), M2=SILENT_TAU (supg tau=1.0 vs 0.5)")
    print(f"N_IC={N_IC}, repeats={N_REPEATS}, sigma={SIGMA:.2f}, "
          f"base working mesh={WORKING_BASE}, base reference mesh={REFERENCE_BASE}")

    print("\n[setup] building the base working mesh and its geometry once")
    all_blocks = [ic_block(base) for base in IC_BASES]
    all_ics = [ic for block in all_blocks for ic in block]
    base_mesh = working_mesh(*WORKING_BASE)
    all_clean = clean_solver_grids(base_mesh, all_ics)
    clean_blocks = split_clean(all_clean, N_REPEATS)

    print("[setup] solving each fine reference IC once on the base reference mesh")
    all_refs = reference_grids(all_ics, *REFERENCE_BASE)
    ref_blocks = split_refs(all_refs, N_REPEATS)

    results = {}
    initial_scenarios = [
        (clean_blocks[i], ref_blocks[i], 41001) for i in range(N_REPEATS)
    ]
    results["initial_conditions"] = evaluate_factor(
        "initial_conditions", initial_scenarios, 0)

    noise_seeds = tuple(42000 + i for i in range(N_REPEATS))
    noise_scenarios = [(clean_blocks[0], ref_blocks[0], seed) for seed in noise_seeds]
    results["noise_realizations"] = evaluate_factor(
        "noise_realizations", noise_scenarios, 1)

    print("\n[setup] building the five working-grid configurations")
    grid_scenarios = []
    for n_side, seed in WORKING_GRID_SPECS:
        print(f"  working mesh n_side={n_side}, seed={seed}")
        mesh = working_mesh(n_side, seed)
        clean = clean_solver_grids(mesh, all_blocks[0])
        grid_scenarios.append((clean, ref_blocks[0], 43001))
    results["working_grid"] = evaluate_factor(
        "working_grid", grid_scenarios, 2)

    print("\n[setup] solving the five reference-resolution configurations")
    ref_scenarios = []
    for n_side in REFERENCE_RESOLUTIONS:
        print(f"  reference mesh n_side={n_side}, seed=7")
        refs = ref_blocks[0] if n_side == REFERENCE_BASE[0] else reference_grids(all_blocks[0], n_side, 7)
        ref_scenarios.append((clean_blocks[0], refs, 44001))
    results["reference_resolution"] = evaluate_factor(
        "reference_resolution", ref_scenarios, 3)

    print("\n" + "=" * 88)
    print("REPEATABILITY RESULTS (GroupKFold-by-IC; mean +/- spread over R repeats)")
    print("=" * 88)
    print(f"{'factor':<24} {'measurement':<18} {'R':>3} {'mean_acc':>10} "
          f"{'sd_acc':>9} {'cv_acc':>9} {'sig_dir_cos':>13}")
    for factor in ("initial_conditions", "noise_realizations", "working_grid", "reference_resolution"):
        for measurement in MEASUREMENTS:
            row = results[factor][measurement]
            print(f"{factor:<24} {measurement:<18} {row['n_repeats']:>3d} "
                  f"{row['mean_acc']:>10.3f} {row['sd_acc']:>9.3f} "
                  f"{row['cv_acc']:>9.3f} {row['sig_dir_cos_mean']:>13.3f}")

    print("\nchance/permutation-floor line: chance=0.500; permutation floors are repeat means +/- SD")
    for factor in ("initial_conditions", "noise_realizations", "working_grid", "reference_resolution"):
        m1, m2 = (results[factor][m] for m in MEASUREMENTS)
        print(f"  {factor:<24} M1={m1['floor_mean']:.3f}+/-{m1['floor_sd']:.3f} "
              f"M2={m2['floor_mean']:.3f}+/-{m2['floor_sd']:.3f}")

    violations = []
    for factor in ("initial_conditions", "noise_realizations", "working_grid", "reference_resolution"):
        for measurement in MEASUREMENTS:
            row = results[factor][measurement]
            reasons = []
            if row["sd_acc"] >= 0.05:
                reasons.append(f"sd_acc={row['sd_acc']:.3f} >= 0.050")
            if row["sig_dir_cos_mean"] <= 0.98:
                reasons.append(f"sig_dir_cos_mean={row['sig_dir_cos_mean']:.3f} <= 0.980")
            if reasons:
                violations.append((factor, measurement, reasons))

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if not violations:
        print("GO / VERDICT: measurement repeatable")
    else:
        print("NO-GO / VERDICT: measurement not repeatable")
        for factor, measurement, reasons in violations:
            print(f"  FLAG {factor}/{measurement}: " + "; ".join(reasons))

    csv_path = write_csv(results)
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()
