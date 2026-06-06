import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A                              # verified FEM + signature machinery
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

"""Measurement-journal error-rate audit for the 2D solver configuration measurand.

The numerical solver configuration is the measurand and the modified-equation
signature is the measurement signal.  This script measures two error rates:

* false positive: two independently noisy observations of the same SUPG(1.0)
  configuration are incorrectly declared to be different;
* false attribution: a held-out classifier assigns a signature to the wrong one
  of five solver configurations.

The fine SUPG reference fields and clean working-mesh solver fields are cached.
Noise is added only after those solves, so the sigma sweep does not repeat FEM
assembly or solve work.
"""

import csv
from sklearn.model_selection import cross_val_predict, GroupKFold


SIGMAS = (0.0, 0.01, 0.05)
N_FPR_IC = 60
N_ID_IC = 80
N_FPR_TRIALS = 200
PERM_REPS = 15
WORK_N_SIDE = 28
WORK_SEED = 2026
REF_N_SIDE = 96
REF_SEED = 7
FIXED_SEED = 41017

SCHEME_CONFIGS = (
    ("galerkin", "galerkin", 1.0),
    ("supg(1.0)", "supg", 1.0),
    ("supg_half(0.5)", "supg", 0.5),
    ("supg_2tau(2.0)", "supg", 2.0),
    ("artvisc", "artvisc", 1.0),
)
SCHEME_NAMES = tuple(row[0] for row in SCHEME_CONFIGS)


def sig_from_grid(Us, Ur):
    R = Us - Ur
    Dlib, sl = A._fd_library(Us)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm > 0 else c


def _noisy_signature(clean_grid, ref_grid, sigma, rng):
    Us = clean_grid
    if sigma > 0:
        rms = np.sqrt(np.mean(Us**2))
        Us = Us + sigma * rms * rng.standard_normal(Us.shape)
    return sig_from_grid(Us, ref_grid)


def _build_clean_grids(pts, elems, on_bnd, geom, ics, nu_art):
    """Solve each of the five configurations once per IC and grid the result."""
    clean = {name: [] for name in SCHEME_NAMES}
    for name, scheme, tau_scale in SCHEME_CONFIGS:
        for ic in ics:
            if scheme == "artvisc":
                u, _ = A.assemble("artvisc", pts, elems, on_bnd, ic,
                                   nu_art=nu_art, geom=geom)
            else:
                u, _ = A.assemble(scheme, pts, elems, on_bnd, ic,
                                  tau_scale=tau_scale, geom=geom)
            clean[name].append(A._to_grid(pts, u))
        clean[name] = np.asarray(clean[name])
    return clean


def _signatures_for_scheme(clean, refs, sigma, seed, n_ic):
    rng = np.random.default_rng(seed)
    return np.asarray([_noisy_signature(clean[i], refs[i], sigma, rng)
                       for i in range(n_ic)])


def _fpr_trial(clean, refs, left, right, sigma, trial):
    """Build one same-scheme trial with independent RNG streams for each half."""
    left_rng = np.random.default_rng(FIXED_SEED + 100000 + 1000 * trial)
    right_rng = np.random.default_rng(FIXED_SEED + 200000 + 1000 * trial)
    left_C = np.asarray([_noisy_signature(clean[i], refs[i], sigma, left_rng)
                         for i in left])
    right_C = np.asarray([_noisy_signature(clean[i], refs[i], sigma, right_rng)
                          for i in right])
    X = A.feats(np.vstack([left_C, right_C]))
    y = np.r_[np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)]
    groups = np.r_[left, right]
    return X, y, groups


def _false_positive_rates(clean, refs):
    """Freeze-then-test false-alarm protocol.

    The permutation floor is estimated on a dedicated CALIBRATION split whose ICs/noise
    are never reused as a test trial (trial index -1, its own RNG streams), then FROZEN.
    The frozen decision rule (acc - floor > 0.15 AND acc >= 0.75 -- pre-specified
    admissibility constants, not fitted here) is evaluated on N_FPR_TRIALS INDEPENDENT
    test trials. With zero observed events the reported point estimate is 0, but the
    honest summary is "0/N, one-sided ~95% upper bound 3/N" (rule of three).
    """
    clean = clean[:N_FPR_IC]
    refs = refs[:N_FPR_IC]
    rates = {}
    floors = {}
    accuracies = {}
    events = {}
    uppers = {}
    half = N_FPR_IC // 2
    for sigma_index, sigma in enumerate(SIGMAS):
        split_rng = np.random.default_rng(FIXED_SEED + 300000 + sigma_index)
        # --- calibration split (disjoint from the test trials) -> frozen floor ---
        cal_perm = split_rng.permutation(N_FPR_IC)
        Xc, yc, gc = _fpr_trial(clean, refs, cal_perm[:half], cal_perm[half:], sigma, -1)
        floor = float(A.perm_floor(Xc, yc, gc,
                                   seed=FIXED_SEED + 400000 + sigma_index,
                                   reps=PERM_REPS))
        # --- test trials, frozen rule ---
        accs = []
        n_events = 0
        for trial in range(N_FPR_TRIALS):
            permutation = split_rng.permutation(N_FPR_IC)
            X, y, groups = _fpr_trial(clean, refs, permutation[:half],
                                      permutation[half:], sigma, trial)
            acc = float(A.cv_acc(X, y, groups))
            accs.append(acc)
            if (acc - floor > 0.15) and (acc >= 0.75):
                n_events += 1
        rates[sigma] = n_events / N_FPR_TRIALS
        events[sigma] = n_events
        uppers[sigma] = 3.0 / N_FPR_TRIALS if n_events == 0 else None
        floors[sigma] = floor
        accuracies[sigma] = float(np.mean(accs))
    return rates, floors, accuracies, events, uppers


def _identification(clean, refs, sigma, sigma_index):
    features = []
    for class_index, name in enumerate(SCHEME_NAMES):
        C = _signatures_for_scheme(clean[name], refs, sigma,
                                   FIXED_SEED + 500000 + 10000 * sigma_index + class_index,
                                   N_ID_IC)
        features.append(C)
    X = A.feats(np.vstack(features))
    y = np.concatenate([np.full(N_ID_IC, class_index, dtype=int)
                        for class_index in range(len(SCHEME_NAMES))])
    groups = np.concatenate([np.arange(N_ID_IC) for _ in SCHEME_NAMES])
    pred = cross_val_predict(A._clf(), X, y, groups=groups, cv=GroupKFold(5))
    rate = float(np.mean(pred != y))
    floor = float(A.perm_floor(X, y, groups,
                               seed=FIXED_SEED + 600000 + sigma_index,
                               reps=PERM_REPS))
    return rate, floor, pred, y


def _confusion(pred, y):
    matrix = np.zeros((len(SCHEME_NAMES), len(SCHEME_NAMES)), dtype=int)
    for true_label, pred_label in zip(y, pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _print_matrix(matrix):
    print("\nconfusion matrix at sigma=0.01 (rows=true, cols=pred):")
    print(f"{'true / pred':<20}" + "".join(f"{name:>18}" for name in SCHEME_NAMES))
    for row_index, name in enumerate(SCHEME_NAMES):
        cells = "".join(f"{matrix[row_index, col_index]:>18d}"
                        for col_index in range(len(SCHEME_NAMES)))
        print(f"{name:<20}{cells}")


def _dominant_pairs(matrix):
    pairs = []
    for i in range(len(SCHEME_NAMES)):
        for j in range(i + 1, len(SCHEME_NAMES)):
            pairs.append((int(matrix[i, j] + matrix[j, i]),
                          SCHEME_NAMES[i], SCHEME_NAMES[j]))
    return sorted(pairs, reverse=True)


def _write_csv(path, fpr_rates, id_rates, confusion, fpr_events, fpr_uppers):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "metric", "sigma", "rate", "n_trials",
                         "true_scheme", "pred_scheme", "count", "events",
                         "upper95_onesided"])
        for sigma in SIGMAS:
            upper = fpr_uppers[sigma]
            writer.writerow(["metric", "false_positive_rate", f"{sigma:.2f}",
                             f"{fpr_rates[sigma]:.6f}", N_FPR_TRIALS, "", "",
                             "", fpr_events[sigma],
                             "" if upper is None else f"{upper:.6f}"])
        for sigma in SIGMAS:
            writer.writerow(["metric", "false_attribution_rate", f"{sigma:.2f}",
                             f"{id_rates[sigma]:.6f}", 1, "", "", ""])
        for true_index, true_name in enumerate(SCHEME_NAMES):
            for pred_index, pred_name in enumerate(SCHEME_NAMES):
                writer.writerow(["confusion", "", "0.01", "", "",
                                 true_name, pred_name,
                                 int(confusion[true_index, pred_index])])


def main():
    print("=" * 96)
    print("MEASUREMENT ERROR-RATE AUDIT: solver configuration as measurand")
    print("modified-equation signature as measurement signal")
    print("=" * 96)
    print(f"working mesh: n_side={WORK_N_SIDE}, seed={WORK_SEED}; "
          f"fine SUPG reference: n_side={REF_N_SIDE}, seed={REF_SEED}")
    print(f"ICs: FPR={N_FPR_IC}, identification={N_ID_IC}; sigmas={SIGMAS}")
    print(f"FPR protocol: {N_FPR_TRIALS} random disjoint-half trials per sigma; "
          f"permutation floor={PERM_REPS} reps on one representative split per sigma")

    ics = [A.make_ic(1000 + i) for i in range(N_ID_IC)]
    pts, elems, on_bnd = A.make_mesh(WORK_N_SIDE, seed=WORK_SEED)
    geom = A.mesh_geometry(pts, elems)
    ref_pts, ref_elems, ref_bnd = A.make_mesh(REF_N_SIDE, seed=REF_SEED)
    nu_art = A.added_diffusion_supg(pts, elems, tau_scale=1.0, geom=geom)

    print(f"working mesh: {len(pts)} nodes, {len(elems)} triangles; "
          f"matched artvisc nu_art={nu_art:.3e}")
    print(f"[solving fine SUPG reference once for {N_ID_IC} ICs] ...")
    ref_grids = A.reference_grids(ics, ref_pts, ref_elems, ref_bnd)
    print("[solving each clean working-mesh configuration once per IC] ...")
    clean = _build_clean_grids(pts, elems, on_bnd, geom, ics, nu_art)
    print("[reusing clean grids for the noise sweep] ...")

    fpr_rates, fpr_floors, fpr_mean_acc, fpr_events, fpr_uppers = \
        _false_positive_rates(clean["supg(1.0)"], ref_grids)
    id_rates = {}
    id_floors = {}
    id_predictions = {}
    id_labels = None
    for sigma_index, sigma in enumerate(SIGMAS):
        rate, floor, pred, labels = _identification(
            clean, ref_grids, sigma, sigma_index)
        id_rates[sigma] = rate
        id_floors[sigma] = floor
        id_predictions[sigma] = pred
        id_labels = labels

    confusion = _confusion(id_predictions[0.01], id_labels)
    csv_path = os.path.join(TAB, "error_rates.csv")
    _write_csv(csv_path, fpr_rates, id_rates, confusion, fpr_events, fpr_uppers)

    print("\n" + "=" * 96)
    print("RESULT TABLE")
    print("=" * 96)
    print(f"{'metric':<25}{'sigma':>8}{'rate':>12}{'events/N':>12}{'95% upper':>12}{'mean acc':>12}")
    print("-" * 81)
    for sigma in SIGMAS:
        upper = fpr_uppers[sigma]
        upper_txt = "-" if upper is None else f"{upper:.4f}"
        print(f"{'false_positive_rate':<25}{sigma:>8.2f}{fpr_rates[sigma]:>12.4f}"
              f"{f'{fpr_events[sigma]}/{N_FPR_TRIALS}':>12}{upper_txt:>12}{fpr_mean_acc[sigma]:>12.3f}")
    for sigma in SIGMAS:
        print(f"{'false_attribution_rate':<25}{sigma:>8.2f}{id_rates[sigma]:>12.3f}"
              f"{1:>12d}{1.0 - id_rates[sigma]:>12.3f}")

    print("\nchance/permutation-floor line:")
    print("  binary chance=0.500; five-way identification chance=0.200")
    for sigma in SIGMAS:
        print(f"  sigma={sigma:.2f}: FPR representative floor={fpr_floors[sigma]:.3f}; "
              f"5-way ID floor={id_floors[sigma]:.3f}")
    print("  FPR declaration rule: (accuracy - floor > 0.15) and accuracy >= 0.75")
    print(f"\nCSV -> {csv_path}")

    _print_matrix(confusion)
    pairs = _dominant_pairs(confusion)
    nonzero_pairs = [pair for pair in pairs if pair[0] > 0]
    if nonzero_pairs:
        dominant_text = "; ".join(
            f"{left} <-> {right}: {count}"
            for count, left, right in nonzero_pairs[:3])
    else:
        dominant_text = "none"
    print(f"dominant bidirectional confusions: {dominant_text}")
    print("interpretation: the leading attribution errors are the measured "
          "collinear/consistency limits, especially nominal SUPG versus its "
          "tau-detuned neighbours when those pairs dominate.")

    max_fpr = max(fpr_rates.values())
    verdict = "GO" if max_fpr <= 0.05 else "CAUTION"
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for sigma in SIGMAS:
        upper = fpr_uppers[sigma]
        if fpr_events[sigma] == 0:
            fp_txt = (f"no false alarms observed ({fpr_events[sigma]}/{N_FPR_TRIALS}); "
                      f"one-sided ~95% upper bound {upper:.4f} (rule of three)")
        else:
            fp_txt = (f"false-alarm rate {fpr_rates[sigma]:.4f} "
                      f"({fpr_events[sigma]}/{N_FPR_TRIALS})")
        print(f"  sigma={sigma:.2f}: {fp_txt}; false-attribution="
              f"{id_rates[sigma]:.3f} (= 1 - ID accuracy)")
    print(f"[{verdict}] false-alarm reliability is "
          f"{'small' if verdict == 'GO' else 'not uniformly small'} "
          f"(worst observed {max_fpr:.4f}, one-sided 95% upper bound {3.0/N_FPR_TRIALS:.4f} "
          f"when zero events); attribution uncertainty is reported by the "
          "false-attribution rates and confusion matrix.")


if __name__ == "__main__":
    main()
