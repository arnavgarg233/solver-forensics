import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A                              # verified FEM + signature machinery
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
from sklearn.model_selection import GroupKFold, cross_val_predict


# The measurand is the continuous SUPG strength alpha=tau_scale.  Everything
# below consumes the verified anchor's FEM, interpolation, and FD library.
# The detuning is intrinsically ASYMMETRIC, so the detection limit is reported
# separately for WEAKENING (alpha<1, toward Galerkin) and STRENGTHENING (alpha>1,
# over-stabilization).  delta_alpha = |alpha - 1| in both directions.
ALPHAS_WEAK = np.round(np.array([1.0 - 0.1 * k for k in range(11)]), 3)     # 1.0 -> 0.0
ALPHAS_STRONG = np.round(np.array([1.0 + 0.1 * k for k in range(11)]), 3)   # 1.0 -> 2.0
DIRECTIONS = (("weaken", ALPHAS_WEAK), ("strengthen", ALPHAS_STRONG))
ALL_ALPHAS = np.round(
    np.array(sorted(set(np.r_[ALPHAS_WEAK, ALPHAS_STRONG].tolist()))), 3)   # 0.0 .. 2.0
SIGMAS = (0.0, 0.01, 0.05)
N_IC = 80
B_BOOT = 200
TARGET_SENSITIVITY = 0.95
TARGET_FPR = 0.05


def sig_from_grid(Us, Ur):
    R = Us - Ur
    Dlib, sl = A._fd_library(Us)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm > 0 else c


def _detection_limit(deltas, sensitivities):
    """First 95% crossing, linearly interpolated on the alpha-grid interval."""
    hit = np.flatnonzero((deltas > 0.0) & (sensitivities >= TARGET_SENSITIVITY))
    if len(hit) == 0:
        return None
    j = int(hit[0])
    if j == 0:
        return float(deltas[0])
    x0, y0 = float(deltas[j - 1]), float(sensitivities[j - 1])
    x1, y1 = float(deltas[j]), float(sensitivities[j])
    if y1 <= y0:
        return x1
    return x0 + (TARGET_SENSITIVITY - y0) * (x1 - x0) / (y1 - y0)


def _supervised_sensitivity(nominal, detuned):
    """Sensitivity of the method's own discriminative detector at a calibrated FPR.

    Two-class HELD-OUT detection: class 0 = nominal (alpha=1) signatures, class 1 =
    detuned signatures, grouped by initial condition (GroupKFold keeps an IC's nominal
    and detuned samples in the same held-out fold, so no IC leaks across the split). The
    detector score is the cross-validated P(detuned) from the SAME StandardScaler +
    LogisticRegression the audit uses -- i.e. the signal the method actually exploits,
    not a raw angle. The decision threshold is calibrated on the held-out NOMINAL scores
    to a TARGET_FPR false-positive rate; sensitivity = fraction of held-out DETUNED scores
    above it. At delta_alpha=0 (nominal vs nominal) this returns ~TARGET_FPR by
    construction (the FPR check); at large detuning it saturates to 1.
    """
    X = A.feats(np.vstack([nominal, detuned]))
    y = np.r_[np.zeros(len(nominal)), np.ones(len(detuned))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(detuned))]
    k = min(5, len(nominal))
    proba = cross_val_predict(A._clf(), X, y, groups=g,
                              cv=GroupKFold(k), method="predict_proba")[:, 1]
    threshold = float(np.percentile(proba[y == 0], 100.0 * (1.0 - TARGET_FPR)))
    return float(np.mean(proba[y == 1] > threshold))


def _evaluate_detector(S_alpha, deltas):
    """Sensitivity vs |detuning| for one noise level, plus the 95%-sensitivity limit.

    S_alpha is ordered with the nominal (delta_alpha=0) config FIRST.
    """
    nominal = S_alpha[0]
    sensitivities = np.array(
        [_supervised_sensitivity(nominal, S_alpha[j]) for j in range(len(S_alpha))],
        dtype=float,
    )
    return TARGET_FPR, sensitivities, _detection_limit(deltas, sensitivities)


def _censored_percentile(values, q):
    """Percentile for limits right-censored above delta_alpha=1."""
    ordered = sorted(np.inf if value is None else value for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if np.isinf(ordered[lo]) or np.isinf(ordered[hi]):
        return None
    weight = position - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def _bootstrap_ci(S_alpha, deltas, seed):
    """Resample ICs; return (ci_lo, ci_hi, n_censored). A resample is right-censored
    when the 95% sensitivity is never reached within the tested detuning range."""
    rng = np.random.default_rng(seed)
    limits = []
    for _ in range(B_BOOT):
        draw = rng.integers(0, N_IC, size=N_IC)
        _, _, limit = _evaluate_detector(S_alpha[:, draw, :], deltas)
        limits.append(limit)
    return (_censored_percentile(limits, 2.5), _censored_percentile(limits, 97.5),
            sum(limit is None for limit in limits))


def _limit_text(value):
    return ">1.0" if value is None else f"{value:.3f}"


def _build_clean_grids(pts, elems, on_bnd, ics, geom):
    """Solve every alpha in ALL_ALPHAS once per IC; noise is added only later.
    Returns {alpha: (N_IC, GRID, GRID)} covering both detuning directions."""
    clean = {}
    for alpha in ALL_ALPHAS:
        grids = np.empty((N_IC, A.GRID_OBS, A.GRID_OBS))
        for i, ic in enumerate(ics):
            u, _ = A.assemble("supg", pts, elems, on_bnd, ic,
                              tau_scale=float(alpha), geom=geom)
            grids[i] = A._to_grid(pts, u)
        clean[float(alpha)] = grids
    print(f"  cached {len(ALL_ALPHAS)} alphas x {N_IC} clean solver grids")
    return clean


def _signature_sweeps(clean, ref_grids):
    """Apply one fixed RNG stream per (sigma, alpha), then recover signatures.
    Returns {sigma: {alpha: (N_IC, len(LIB))}}."""
    sweeps = {}
    for sigma_index, sigma in enumerate(SIGMAS):
        per_alpha = {}
        for alpha_index, alpha in enumerate(ALL_ALPHAS):
            rng = np.random.default_rng(73000 + 100 * sigma_index + alpha_index)
            S = np.empty((N_IC, len(A.LIB)))
            for i in range(N_IC):
                Us = clean[float(alpha)][i]
                rms = np.sqrt(np.mean(Us**2))
                Us_noisy = Us + sigma * rms * rng.standard_normal(Us.shape)
                S[i] = sig_from_grid(Us_noisy, ref_grids[i])
            per_alpha[float(alpha)] = S
        sweeps[sigma] = per_alpha
        print(f"  sigma={sigma:.2f}: recovered {len(ALL_ALPHAS) * N_IC} signatures")
    return sweeps


def _stack_direction(per_alpha, alphas):
    """Stack signatures in a direction's alpha order (nominal alpha=1 first)."""
    return np.stack([per_alpha[float(a)] for a in alphas], axis=0)


def _write_csv(rows):
    path = os.path.join(TAB, "detection_limit.csv")
    fields = ["type", "direction", "sigma", "alpha", "delta_alpha", "sensitivity",
              "detection_limit_delta_alpha", "dl_ci_lo", "dl_ci_hi", "fpr_at_alpha1",
              "frac_censored"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    assert A.N_IC == N_IC, "the requested protocol requires the anchor's 80 IC ensemble"
    print("=" * 88)
    print("MEASUREMENT :: DETECTION LIMIT OF SUPG STABILIZATION DETUNING")
    print("Measurand: alpha=tau_scale; signal: unit modified-equation signature direction")
    print("=" * 88)
    print(f"Protocol: {N_IC} ICs | weaken alphas={list(ALPHAS_WEAK)} | strengthen alphas={list(ALPHAS_STRONG)}")
    print(f"          calibrated FPR target={TARGET_FPR:.2f} | B={B_BOOT} | detection limit reported per direction")

    ics = [A.make_ic(1000 + i) for i in range(N_IC)]
    pts, elems, on_bnd = A.make_mesh(28, seed=2026)
    geom = A.mesh_geometry(pts, elems)
    ref_pts, ref_elems, ref_bnd = A.make_mesh(96, seed=7)

    print("\nREFERENCE: fine SUPG fields (once per IC; independent of alpha and noise)")
    ref_grids = A.reference_grids(ics, ref_pts, ref_elems, ref_bnd)
    print(f"  cached {len(ref_grids)} fine reference grids")

    print("\nWORKING SOLVER: clean SUPG grids (one solve per IC and alpha)")
    clean = _build_clean_grids(pts, elems, on_bnd, ics, geom)

    print("\nSIGNATURE ACQUISITION: add observation noise after the clean solves")
    sweeps = _signature_sweeps(clean, ref_grids)

    summaries = []
    csv_rows = []
    for dir_index, (direction, alphas) in enumerate(DIRECTIONS):
        deltas = np.abs(alphas - 1.0)
        print(f"\nRESULT TABLE [{direction}]: calibrated-detector sensitivity vs detuning")
        print(f"  {'sigma':>5} {'alpha':>6} {'delta_alpha':>12} {'sensitivity':>12}")
        print("  " + "-" * 42)
        for sigma_index, sigma in enumerate(SIGMAS):
            S_dir = _stack_direction(sweeps[sigma], alphas)
            _, sensitivity, limit = _evaluate_detector(S_dir, deltas)
            ci_lo, ci_hi, n_censored = _bootstrap_ci(
                S_dir, deltas, seed=88000 + 1000 * dir_index + sigma_index)
            frac_censored = n_censored / B_BOOT
            monotone = bool(np.all(np.diff(sensitivity) >= -1e-12))
            for alpha, delta, value in zip(alphas, deltas, sensitivity):
                print(f"  {sigma:5.2f} {alpha:6.1f} {delta:12.1f} {value:12.3f}")
                csv_rows.append(dict(type="curve", direction=direction, sigma=f"{sigma:.2f}",
                                     alpha=f"{alpha:.1f}", delta_alpha=f"{delta:.1f}",
                                     sensitivity=f"{value:.6f}"))
            summaries.append(dict(direction=direction, sigma=sigma, sensitivity=sensitivity,
                                  limit=limit, ci_lo=ci_lo, ci_hi=ci_hi,
                                  fpr=float(sensitivity[0]), extreme=float(sensitivity[-1]),
                                  monotone=monotone, frac_censored=frac_censored))
            csv_rows.append(dict(type="limit", direction=direction, sigma=f"{sigma:.2f}",
                                 detection_limit_delta_alpha=_limit_text(limit),
                                 dl_ci_lo=_limit_text(ci_lo), dl_ci_hi=_limit_text(ci_hi),
                                 fpr_at_alpha1=f"{sensitivity[0]:.6f}",
                                 frac_censored=f"{frac_censored:.3f}"))

    print("\nLIMIT TABLE: 95% sensitivity crossing at a calibrated 5% false-positive rate")
    print("  (a right-censored resample never reaches 95% within the tested range; a large")
    print("   censored fraction means the point limit and CI are lower bounds, not precise values)")
    print(f"  {'direction':>11} {'sigma':>5} {'FPR@a=1':>9} {'sens@extreme':>13} {'DL delta_a':>11} {'95% CI':>20} {'censored':>9}")
    print("  " + "-" * 92)
    for result in summaries:
        ci_text = f"[{_limit_text(result['ci_lo'])}, {_limit_text(result['ci_hi'])}]"
        limit_text = _limit_text(result['limit'])
        if result["frac_censored"] > 0.05 and result["limit"] is not None:
            limit_text = "~" + limit_text  # censored: treat as a lower bound
        print(f"  {result['direction']:>11} {result['sigma']:5.2f} {result['fpr']:9.3f} "
              f"{result['extreme']:13.3f} {limit_text:>11} {ci_text:>20} "
              f"{result['frac_censored']*100:8.1f}%")

    print("\nCHANCE / FALSE-POSITIVE CONTEXT: held-out detection test whose decision threshold")
    print(f"  is calibrated to a {TARGET_FPR:.2f} false-positive rate on the nominal (alpha=1) scores;")
    print("  sensitivity is the true-positive rate of the audit's own classifier at that operating point.")

    csv_path = _write_csv(csv_rows)
    print(f"\nmetrics -> {csv_path}")

    verdict_ok = all(result["extreme"] >= TARGET_SENSITIVITY and result["fpr"] <= 0.10
                     for result in summaries)
    print("\n" + "=" * 92)
    print(f"[{'GO' if verdict_ok else 'CHECK'} / VERDICT] detection-limit measurement (asymmetric)")
    print("=" * 92)
    for result in summaries:
        tag = f"{result['direction']:>11}, sigma={result['sigma']:.2f}"
        if result["limit"] is None:
            print(f"  {tag}: not reached within the tested range "
                  f"({result['frac_censored']*100:.0f}% of bootstraps also censored); "
                  f"FPR={result['fpr']:.2f}, sensitivity at extreme={result['extreme']:.2f}.")
        elif result["frac_censored"] > 0.05:
            print(f"  {tag}: detuning resolvable near delta_alpha ~= {result['limit']:.3f} "
                  f"but RIGHT-CENSORED ({result['frac_censored']*100:.0f}% of bootstraps never "
                  f"reach 95%); report as a lower bound, CI [{_limit_text(result['ci_lo'])}, "
                  f"{_limit_text(result['ci_hi'])}].")
        else:
            print(f"  {tag}: resolves detuning down to delta_alpha ~= {result['limit']:.3f} "
                  f"(95% CI [{_limit_text(result['ci_lo'])}, {_limit_text(result['ci_hi'])}]); "
                  f"FPR={result['fpr']:.2f}, sensitivity at extreme={result['extreme']:.2f}.")


if __name__ == "__main__":
    main()
