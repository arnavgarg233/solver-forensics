"""Repeated split-conformal detection for paired per-unit signature populations.

WHY THIS MODULE EXISTS
The repository's earlier ``supervised_sensitivity`` cross-fitted the classifier
correctly but then set its decision threshold from the 95th percentile of ALL
evaluated nominal out-of-fold scores.  The nominal scores it thresholded were
the same scores it went on to test, so a 5% false-alarm rate was IMPOSED by
construction and never measured, and the alpha=1 self-comparison was a
degenerate nominal-against-itself contrast.

A first replacement calibrated a test fold on 39 out-of-fold nominal scores of
other ICs.  That removed the imposed level but was only APPROXIMATELY valid:
those calibration scores came from other folds' models, which had been fitted on
the current test ICs, so calibration and test scores were not exchangeable under
one fixed score function.  No exactness claim for that 39/38 design survives.

THE ESTIMATOR (strict disjoint split)
The independent unit is the initial condition (IC); the nominal and changed rows
of one IC are a pair and are never separated.  Each of ``repeats`` independent IC
permutations is cut into ``folds`` test folds, and for every test fold the ICs
outside it are partitioned deterministically into ``N_CAL`` = 19 calibration ICs
and the remaining classifier-training ICs.  Training, calibration and test IC
sets are therefore PAIRWISE DISJOINT, and:

  1. one StandardScaler + LogisticRegression classifier is fitted on the paired
     nominal/changed rows of the training ICs only (>= ``MIN_TRAIN_IC`` = 20);
  2. that same fitted model scores the 19 calibration NOMINAL ICs and the
     untouched test ICs, so all these scores share one fixed score function;
  3. the alarm threshold is the ``CAL_RANK`` = 19th order statistic, i.e. the
     maximum, of the 19 calibration nominal scores.  Conditional on the fitted
     model the calibration and test nominal scores are exchangeable, so
     P(false alarm) = (19 + 1 - 19)/(19 + 1) = 1/20 <= 5% exactly, and strict
     ``>`` thresholding keeps ties on the conservative side;
  4. changed detections and nominal false alarms are then counted on the test
     ICs, which entered neither the fit nor the calibration set.

For n=60 with 5 folds the split is 29 train / 19 calibration / 12 test ICs; for
n=48 with 6 folds it is 21 / 19 / 8.

Per-IC indicators are averaged over repeats, so TPR and the MEASURED false-alarm
rate are means of ``n`` independent per-IC quantities.  Their intervals come
from an IC-cluster percentile bootstrap over those per-IC values, so repeats,
folds and any nested replicates never inflate the effective sample size.

Remaining caveat: the test ICs of one fold share a single threshold, so their
alarm indicators are dependent within the fold.  Each is marginally bounded by
5%; the between-fold spread is carried by the reported intervals.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_CAL = 19                 # calibration ICs per test fold, disjoint from train
CAL_RANK = 19              # 1-based order statistic (the maximum) used as threshold
MIN_TRAIN_IC = 20          # smallest admissible training set, in ICs
PREFERRED_FOLDS = 5        # 5 folds at n=60; raised when 5 would train on too few
REPEATS = 20               # independent split repeats
N_BOOT = 2000              # IC-cluster bootstrap replicates
CI_COVERAGE = 0.95
TARGET_FPR = 0.05


def _classifier():
    """The repository's audit classifier (mirrors supg_2d_engineering._clf)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def calibration_level(n_cal=N_CAL, cal_rank=CAL_RANK):
    """Exact per-IC false-alarm level of the (n_cal, cal_rank) order statistic.

    Rejects any pair that does not give exactly ``TARGET_FPR``, so a calibration
    set can never be silently loosened to a level the guarantee does not hold at.
    """
    n_cal, cal_rank = int(n_cal), int(cal_rank)
    if not 1 <= cal_rank <= n_cal:
        raise ValueError(f"rank {cal_rank} is outside 1..{n_cal}")
    level = (n_cal + 1 - cal_rank) / (n_cal + 1)
    if abs(level - TARGET_FPR) > 1.0e-12:
        raise ValueError(f"calibration ({n_cal}, {cal_rank}) gives a "
                         f"{level:.4f} level, not the required {TARGET_FPR:.2f}")
    return float(level)


def choose_folds(n, n_cal=N_CAL, min_train=MIN_TRAIN_IC, preferred=PREFERRED_FOLDS):
    """Fewest folds from ``preferred`` upwards that support a three-way split.

    A fold count is admissible when the ICs outside the largest test fold can
    supply both the ``n_cal`` calibration ICs and a training set of at least
    ``min_train`` ICs, the two being disjoint.  This returns 5 for n=60 (29/19/12)
    and 6 for n=48 (21/19/8).
    """
    n = int(n)
    need = int(n_cal) + int(min_train)
    for folds in range(int(preferred), n + 1):
        if n - int(np.ceil(n / folds)) >= need:
            return folds
    raise ValueError(f"n={n} ICs cannot leave {need} ICs ({n_cal} calibration + "
                     f"{min_train} training) outside a test fold; an exact "
                     f"{TARGET_FPR:.0%} level needs n >= {need + 1}")


def fold_plan(n, repeats=REPEATS, folds=None, n_cal=N_CAL, seed=0,
              min_train=MIN_TRAIN_IC):
    """Deterministic pairwise-disjoint train/calibration/test ICs per fold.

    Each repeat permutes the ICs once and cuts the permutation into folds, so an
    IC is a test IC exactly once per repeat and its paired nominal and changed
    rows always travel together.  The ICs outside the test fold are split in
    permutation order into the first ``n_cal`` calibration ICs and the remaining
    training ICs, so the three sets are disjoint and their union is all n ICs.
    """
    n, n_cal, repeats = int(n), int(n_cal), int(repeats)
    folds = choose_folds(n, n_cal, min_train) if folds is None else int(folds)
    if n - int(np.ceil(n / folds)) < n_cal + int(min_train):
        raise ValueError(f"{folds} folds on n={n} cannot leave {n_cal} calibration "
                         f"ICs plus {min_train} training ICs outside a test fold")
    plan = []
    for repeat in range(repeats):
        order = np.random.default_rng([int(seed), repeat]).permutation(n)
        cut = []
        for chunk in np.array_split(order, folds):
            test = np.sort(chunk)
            outside = order[~np.isin(order, test)]
            cut.append({"test": test, "cal": np.sort(outside[:n_cal]),
                        "train": np.sort(outside[n_cal:])})
        plan.append(cut)
    return plan


def _features(values, name):
    """Per-IC feature matrix; a 1-D detector becomes a single column."""
    X = np.asarray(values, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"{name} must be (n_ic,) or (n_ic, n_feature)")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains a non-finite value")
    return X


def split_conformal_detection(nominal, changed, seed=0, repeats=REPEATS,
                              folds=None, n_cal=N_CAL, cal_rank=CAL_RANK,
                              n_boot=N_BOOT, coverage=CI_COVERAGE):
    """Measured TPR and false-alarm rate for one paired nominal/changed contrast.

    One classifier per (repeat, fold) is fitted on the training ICs only and then
    scores that fold's calibration nominal ICs and its untouched test ICs, so a
    single fixed score function produces every score the fold's decision uses.
    Returns ``tpr`` and ``fpr`` with IC-cluster bootstrap intervals, the per-IC
    repeat-averaged ``detect`` and ``alarm`` indicators those means come from,
    and the protocol actually used.  ``fpr`` is measured on test ICs; only
    ``fpr_bound`` is a guarantee.
    """
    X0 = _features(nominal, "nominal")
    X1 = _features(changed, "changed")
    if X0.shape != X1.shape:
        raise ValueError("nominal and changed populations must have equal n and "
                         f"feature count, got {X0.shape} and {X1.shape}")
    n = X0.shape[0]
    bound = calibration_level(n_cal, cal_rank)
    plan = fold_plan(n, repeats, folds, n_cal, seed)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n), np.ones(n)]
    detect = np.zeros(n)
    alarm = np.zeros(n)

    for cut in plan:
        for fold in cut:
            train, cal, test = fold["train"], fold["cal"], fold["test"]
            rows = np.r_[train, train + n]
            model = _classifier().fit(X[rows], y[rows])
            score = model.predict_proba(
                X[np.r_[cal, test, test + n]])[:, 1]
            threshold = np.sort(score[:cal.size])[int(cal_rank) - 1]
            detect[test] += score[cal.size + test.size:] > threshold
            alarm[test] += score[cal.size:cal.size + test.size] > threshold
    detect /= len(plan)
    alarm /= len(plan)

    ci = cluster_bootstrap_ci(np.column_stack([detect, alarm]), n_boot, seed,
                              coverage)
    return {"n": n, "repeats": len(plan), "folds": len(plan[0]),
            "n_cal": int(n_cal), "cal_rank": int(cal_rank), "fpr_bound": bound,
            "tpr": float(np.mean(detect)), "tpr_ci": tuple(float(v) for v in ci[0]),
            "fpr": float(np.mean(alarm)), "fpr_ci": tuple(float(v) for v in ci[1]),
            "detect": detect, "alarm": alarm}


def cluster_bootstrap_ci(values, n_boot=N_BOOT, seed=0, coverage=CI_COVERAGE):
    """Percentile interval for per-IC means under IC-cluster resampling.

    ``values`` is ``(n_ic,)`` or ``(n_ic, n_quantity)``; columns share one
    resample of the ICs, so several quantities keep their joint IC structure.
    The number of ICs is the effective sample size.
    """
    V = np.asarray(values, dtype=float)
    flat = V.ndim == 1
    V = V[:, None] if flat else V
    if V.ndim != 2 or V.shape[0] < 2:
        raise ValueError("the cluster bootstrap needs at least two ICs")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, V.shape[0], size=(int(n_boot), V.shape[0]))
    means = V[draws].mean(axis=1)
    edges = 100.0 * np.array([0.5 * (1.0 - coverage), 0.5 * (1.0 + coverage)])
    ci = np.percentile(means, edges, axis=0).T
    return ci[0] if flat else ci


def sampled_limit(deltas, tprs, target):
    """Smallest SAMPLED positive delta whose point TPR reaches ``target``.

    Returns ``None`` when no sampled delta reaches it, i.e. the limit is
    right-censored above the swept range.  Nothing is interpolated, so an
    unmeasured gap between two sampled deltas is never reported as a limit, and
    a delta=0 entry is a nominal reference rather than a candidate limit.
    """
    d = np.asarray(deltas, dtype=float)
    t = np.asarray(tprs, dtype=float)
    if d.shape != t.shape:
        raise ValueError("deltas and TPRs must have the same shape")
    hit = (d > 0.0) & (t >= float(target))
    return float(np.min(d[hit])) if np.any(hit) else None


def _censored_percentile(ordered, q):
    """Percentile of right-censored limits sorted ascending with inf last."""
    position = (len(ordered) - 1) * float(q) / 100.0
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if not np.isfinite(ordered[hi]):
        return None
    if ordered[lo] == ordered[hi]:
        return float(ordered[lo])
    weight = position - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def bootstrap_limit(deltas, indicators, target, n_boot=N_BOOT, seed=0,
                    coverage=CI_COVERAGE):
    """IC-cluster bootstrap interval for the sampled-grid detection limit.

    ``indicators`` is ``(n_delta, n_ic)``: the per-IC repeat-averaged detection
    indicators of ONE complete direction curve.  A replicate resamples the ICs
    once and reuses that resample at every delta, so each IC's cross-delta
    pairing is preserved and the whole curve moves together.  Replicates whose
    curve never reaches ``target`` are right-censored; an interval bound that
    falls on a censored replicate is reported as ``None`` together with the
    censored fraction.
    """
    d = np.asarray(deltas, dtype=float)
    M = np.asarray(indicators, dtype=float)
    if M.ndim != 2 or M.shape[0] != d.size:
        raise ValueError("indicators must be (n_delta, n_ic) matching deltas")
    if M.shape[1] < 2:
        raise ValueError("the cluster bootstrap needs at least two ICs")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, M.shape[1], size=(int(n_boot), M.shape[1]))
    curves = M[:, draws].mean(axis=2)                 # (n_delta, n_boot)
    hit = curves >= float(target)
    limits = np.full(int(n_boot), np.inf)
    for j in np.argsort(np.where(d > 0.0, d, np.inf)):
        if d[j] > 0.0:
            limits = np.where(np.isinf(limits) & hit[j], d[j], limits)
    ordered = np.sort(limits)
    lo = 100.0 * 0.5 * (1.0 - coverage)
    return {"limit_ci": (_censored_percentile(ordered, lo),
                         _censored_percentile(ordered, 100.0 - lo)),
            "censored_fraction": float(np.mean(~np.isfinite(limits))),
            "n_boot": int(n_boot)}


def limit_text(value, max_delta):
    """Sampled-grid limit as text at the resolution of the sampled grid itself.

    The detuning grid is spaced by 0.1, so one decimal is the honest precision;
    right-censored limits print as '>max_delta'.
    """
    return f">{float(max_delta):.1f}" if value is None else f"{float(value):.1f}"
