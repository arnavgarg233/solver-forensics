import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A                              # verified FEM + signature machinery
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv


N_IC = A.N_IC
SIGMA = A.SIGMA_MAIN
MARGIN = 0.15
PERM_REPS = 30
IC_INDEX = np.arange(N_IC)


def decide(grid_controlled, acc, floor, margin=0.15) -> str:
    discriminable = (acc - floor) > margin
    if not discriminable:            return "NO-CHANGE"      # nothing detected above the floor
    if grid_controlled:              return "GO"             # detection is admissible
    return "INDETERMINATE"                                    # detected, but grid confound not excluded -> abstain


def sig_from_grid(Us, Ur):
    R = Us - Ur
    Dlib, sl = A._fd_library(Us)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm > 0 else c


def _solver_grids(scheme, pts, elems, on_bnd, ics, geom, tau_scale=1.0):
    """Solve one clean working configuration per IC and interpolate it once."""
    grids = []
    for ic in ics:
        u, _ = A.assemble(scheme, pts, elems, on_bnd, ic,
                          tau_scale=tau_scale, geom=geom)
        grids.append(A._to_grid(pts, u))
    return grids


def _noisy_signatures(clean_grids, ref_grids, sigma, seed):
    """Re-add deterministic observation noise without re-solving the FEM."""
    rng = np.random.default_rng(seed)
    sigs = []
    for Us, Ur in zip(clean_grids, ref_grids):
        if sigma > 0:
            rms = np.sqrt(np.mean(Us**2)); Us = Us + sigma * rms * rng.standard_normal(Us.shape)
        sigs.append(sig_from_grid(Us, Ur))
    return np.asarray(sigs)


def _pair_result(scenario, description, left_grids, right_grids, ref_grids,
                 grid_controlled, noise_seed_left, noise_seed_right, floor_seed):
    left = _noisy_signatures(left_grids, ref_grids, SIGMA, noise_seed_left)
    right = _noisy_signatures(right_grids, ref_grids, SIGMA, noise_seed_right)
    X = A.feats(np.vstack([left, right]))
    y = np.r_[np.zeros(N_IC, dtype=int), np.ones(N_IC, dtype=int)]
    groups = np.r_[IC_INDEX, IC_INDEX]
    acc = float(A.cv_acc(X, y, groups))
    floor = float(A.perm_floor(X, y, groups, floor_seed, reps=PERM_REPS))
    return dict(scenario=scenario, description=description, acc=acc, floor=floor,
                gap=acc - floor, grid_controlled=grid_controlled,
                confound_discriminable=(acc - floor) > MARGIN,
                decision=decide(grid_controlled, acc, floor, margin=MARGIN))


def _pooled_rel_l2(grids, ref_grids):
    numerator = 0.0
    denominator = 0.0
    for Us, Ur in zip(grids, ref_grids):
        numerator += float(np.sum((Us - Ur) ** 2))
        denominator += float(np.sum(Ur ** 2))
    return float(np.sqrt(numerator / (denominator + 1e-12)))


def _refinement_errors(meshes, ics, ref_grids):
    errors = []
    for n_side, (pts, elems, on_bnd, geom, clean_grids) in meshes.items():
        if clean_grids is None:
            clean_grids = _solver_grids("supg", pts, elems, on_bnd, ics, geom, tau_scale=1.0)
        errors.append((n_side, _pooled_rel_l2(clean_grids, ref_grids)))
    return errors


def _print_result_table(results, active_errors, active_monotone, active_decision):
    print("\nRESULT TABLE (GroupKFold-by-IC accuracy and label-permutation floor)")
    print(f"{'scenario':<10} {'naive_acc':>10} {'perm_floor':>11} {'gap':>9} "
          f"{'grid_ctrl':>11} {'confound?':>11} {'decision':>17}")
    print("-" * 86)
    for result in results:
        print(f"{result['scenario']:<10} {result['acc']:>10.3f} {result['floor']:>11.3f} "
              f"{result['gap']:>+9.3f} {str(result['grid_controlled']):>11} "
              f"{str(result['confound_discriminable']):>11} {result['decision']:>17}")
    error_text = ", ".join(f"n={n}:{err:.3e}" for n, err in active_errors)
    print(f"{'3_active':<10} {'n/a':>10} {'n/a':>11} {'n/a':>9} {'active':>11} "
          f"{'n/a':>11} {active_decision:>17}")
    print(f"  refinement errors (clean SUPG vs shared fine reference): {error_text}")
    print(f"  monotone_decreasing={active_monotone}")


def main():
    print("=" * 96)
    print("MEASUREMENT / ABSTENTION RULE | 2D SUPG modified-equation signatures")
    print("measurand = solver configuration; signal = numerical modified-equation signature")
    print("=" * 96)
    print(f"N_IC={N_IC}  sigma={SIGMA}  margin={MARGIN:.2f}  "
          "GroupKFold(5) grouped by IC  |  no figures")

    ics = [A.make_ic(1000 + i) for i in range(N_IC)]

    ref_pts, ref_elems, ref_bnd = A.make_mesh(96, seed=7)
    print(f"\nreference mesh: n_side=96, seed=7, nodes={len(ref_pts)}, tris={len(ref_elems)}")
    print(f"solving fine SUPG reference once for {N_IC} ICs and reusing it ...")
    ref_grids = A.reference_grids(ics, ref_pts, ref_elems, ref_bnd)
    print("  fine reference complete")

    print("\nSCENARIO 1 | controlled grid, REAL scheme change: Galerkin vs SUPG")
    pts28, elems28, bnd28 = A.make_mesh(28, seed=2026)
    geom28 = A.mesh_geometry(pts28, elems28)
    print(f"  working mesh: n_side=28, seed=2026, nodes={len(pts28)}, tris={len(elems28)}")
    print("  solving clean Galerkin and SUPG grids once per IC ...")
    gal28 = _solver_grids("galerkin", pts28, elems28, bnd28, ics, geom28)
    supg28 = _solver_grids("supg", pts28, elems28, bnd28, ics, geom28, tau_scale=1.0)
    result1 = _pair_result(
        "1", "controlled grid: galerkin vs supg(tau_scale=1.0) on n_side=28 seed=2026",
        gal28, supg28, ref_grids, True, 1101, 1102, 1103)

    print("\nSCENARIO 2 | uncontrolled grid, NO scheme change: Galerkin vs Galerkin")
    pts22, elems22, bnd22 = A.make_mesh(22, seed=3001)
    geom22 = A.mesh_geometry(pts22, elems22)
    pts40, elems40, bnd40 = A.make_mesh(40, seed=3002)
    geom40 = A.mesh_geometry(pts40, elems40)
    print(f"  coarse mesh: n_side=22, seed=3001, nodes={len(pts22)}, tris={len(elems22)}")
    print(f"  fine mesh:   n_side=40, seed=3002, nodes={len(pts40)}, tris={len(elems40)}")
    print("  solving clean Galerkin grids once per IC on each mesh ...")
    gal22 = _solver_grids("galerkin", pts22, elems22, bnd22, ics, geom22)
    gal40 = _solver_grids("galerkin", pts40, elems40, bnd40, ics, geom40)
    result2 = _pair_result(
        "2", "uncontrolled grid: galerkin n_side=22 seed=3001 vs galerkin n_side=40 seed=3002",
        gal22, gal40, ref_grids, False, 2101, 2102, 2103)

    print("\nSCENARIO 3 | active repair: same SUPG scheme across a resolution ladder")
    print("  query = common-grid rel-L2 error against the same fine reference; clean fields")
    print("  reusing the n_side=28 SUPG solve and the n_side=40 mesh geometry from Scenario 2")
    pts20, elems20, bnd20 = A.make_mesh(20, seed=2020)
    geom20 = A.mesh_geometry(pts20, elems20)
    meshes = {
        20: (pts20, elems20, bnd20, geom20, None),
        28: (pts28, elems28, bnd28, geom28, supg28),
        40: (pts40, elems40, bnd40, geom40, None),
    }
    active_errors = _refinement_errors(meshes, ics, ref_grids)
    error_values = [err for _, err in active_errors]
    active_monotone = all(a > b for a, b in zip(error_values, error_values[1:]))
    active_decision = "GO-after-query" if active_monotone else "INDETERMINATE"
    print("  solving clean SUPG grids for the remaining ladder points ...")
    for n_side, error in active_errors:
        print(f"  n_side={n_side:>2d}  pooled rel-L2-vs-fine={error:.3e}")
    print(f"  monotone_decreasing={active_monotone} -> {active_decision}")

    _print_result_table([result1, result2], active_errors, active_monotone, active_decision)
    print(f"\nchance/permutation-floor: binary chance=0.500; each passive floor is the "
          f"median of {PERM_REPS} seeded label permutations under GroupKFold(5).")

    rows = [
        [result1["scenario"], result1["description"], f"{result1['acc']:.4f}",
         f"{result1['floor']:.4f}", str(result1["grid_controlled"]),
         str(result1["confound_discriminable"]), result1["decision"]],
        [result2["scenario"], result2["description"], f"{result2['acc']:.4f}",
         f"{result2['floor']:.4f}", str(result2["grid_controlled"]),
         str(result2["confound_discriminable"]), result2["decision"]],
        ["3_active_query",
         "supg(tau_scale=1.0) n_side=20,28,40; rel-L2="
         + ";".join(f"{n}:{err:.4e}" for n, err in active_errors)
         + f"; monotone_decreasing={active_monotone}", "", "", "active-query", "", active_decision],
    ]
    csv_path = os.path.join(TAB, "abstention.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "description", "naive_acc", "perm_floor",
                         "grid_controlled", "confound_discriminable", "decision"])
        writer.writerows(rows)
    print(f"\nmetrics -> {csv_path}")

    expected = (result1["decision"] == "GO" and
                result2["decision"] == "INDETERMINATE" and
                active_decision == "GO-after-query")
    print("\n" + "=" * 96)
    print("VERDICT | honest passive detection with an abstention rule")
    print("=" * 96)
    print(f"  scenario 1: {result1['decision']} (controlled grid; real scheme change)")
    print(f"  scenario 2: {result2['decision']} (uncontrolled grid; pure resolution change)")
    active_verdict = "GO-after-active-query" if active_monotone else active_decision
    print(f"  scenario 3: {active_verdict} (active grid-invariant convergence query; "
          f"CSV decision={active_decision})")
    print("  the passive detector abstains exactly where a single snapshot cannot exclude the grid confound")
    print(f"\nGO / VERDICT: {'PASS' if expected else 'CHECK'} — "
          "GO, INDETERMINATE, and GO-after-active-query are the required outcomes.")


if __name__ == "__main__":
    main()
