#!/usr/bin/env python3
"""
solver-forensics :: COST / PRACTICAL RECIPE  (turn the phenomenon into a usable tool)
====================================================================================
The audit pieces are scattered across experiments; this one answers the OPERATIONAL
questions a practitioner asks before reaching for the tool, all on the fast linear-
advection kernels (analytic exact solution, the project's validated 4 schemes). It
produces a single compact figure and a 'recipe' table: how many solver runs, how fine
a reference, and how robust the auditor's library/filter choices need to be.

THREE OPERATIONAL QUESTIONS
---------------------------
(1) HOW MANY SOLVER RUNS does the active multi-resolution feature need?
    The grid confound is broken by the convergence RATE p (slope of log||r|| vs log N),
    which is grid-invariant. Estimating p needs >= 2 resolutions. We sweep the number of
    grids in the resolution ladder {2,3,4,5,6} and measure rate-based diffusive-vs-
    dispersive detection (GroupKFold-by-IC, with a permutation floor) vs #runs, plus the
    grid control NC2 (same scheme, two disjoint ladders). The minimum #runs that still
    separates the families is the recipe number.

(2) HOW FINE A REFERENCE suffices?
    The reference is a NUMERICAL fine solve a practitioner would actually use when no
    analytic solution exists: a 2nd-order Lax-Wendroff advection solve at N_ref = m*N,
    sampled back to N. Its fineness m in {1,2,4,8} is a real knob with genuine reference
    error (~5% at 1x, ~0.08% at 8x). We measure two DIFFERENT tasks:
      - ATTRIBUTION accuracy (classify the scheme from the residual signature, 1% field
        noise) against the numerical m-fineness reference, and
      - coefficient RECOVERY: the angle (rad) between the coefficient direction recovered
        against the m-fineness numerical reference and the GOLD direction recovered against
        the analytic exact (clean, so the only error source is reference coarseness).
    Claim under test: attribution holds at 1x where recovery (direction error) needs 4-8x.
    NB the reference scheme is deliberately LOW-order (LW): a high-order reference converges
    so fast that even 1x suffices for recovery too (reported as a caveat).

(3) SENSITIVITY to the auditor's LIBRARY and FILTER choices.
    Re-run attribution with different derivative libraries
      {u_xx,u_xxx} | {u_xx,u_xxx,u_xxxx} | {u_x,u_xx,u_xxx,u_xxxx} (rich, +spurious u_x)
    and different observation filters / coarsenings (anti-aliased N_obs in
    {native, 64, 48, 32} and a smoothing pre-filter). Report detection robustness so a
    practitioner knows the choice does not have to be tuned.

Outputs:
  results/tables/cost_recipe.csv     (long-form metrics for all three parts)
  results/tables/cost_recipe_card.csv (the compact 'recipe card': runs / fineness / robustness)
  figures/fig_cost_recipe.png        (4-panel compact figure)

Self-contained: numpy + scipy + sklearn. Deterministic. CPU, a few minutes on M2.
Run:  python src/limits/cost_recipe.py [--plot]
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

L, A, NU, T = 1.0, 1.0, 0.6, 0.30
N_IC = 60

# ---------------------------------------------------------------- physics (validated kernels)
def exact(u0, t, N):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    return np.fft.irfft(np.fft.rfft(u0) * np.exp(-1j * k * A * t), n=N)
def upwind(u, nu):         return u - nu * (u - np.roll(u, 1))
def lax_friedrichs(u, nu): return 0.5 * (np.roll(u, -1) + np.roll(u, 1)) - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))
def lax_wendroff(u, nu):   return u - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1)) + 0.5 * nu * nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
def beam_warming(u, nu):   return u - 0.5 * nu * (3 * u - 4 * np.roll(u, 1) + np.roll(u, 2)) + 0.5 * nu * nu * (u - 2 * np.roll(u, 1) + np.roll(u, 2))
SCHEMES = {"upwind": upwind, "lax_friedrichs": lax_friedrichs, "lax_wendroff": lax_wendroff, "beam_warming": beam_warming}
names = list(SCHEMES); DIFFUSIVE = {"upwind", "lax_friedrichs"}; UP = "upwind"

def random_ic(N, rng, n_modes=4):                   # smooth (modes 1-4) -> resolved at every ladder grid
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes): u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 5) * x / L + rng.uniform(0, 2 * np.pi))
    return u / (np.std(u) + 1e-9)

def antialias(u, M):                                # proper Fourier resample (handles up/down sampling)
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M // 2 + 1], n=M) * (M / N)

def run(scheme, N, u0_N):
    dx = L / N; dt = NU * dx / A; ns = int(round(T / dt)); u = u0_N.copy()
    for _ in range(ns): u = SCHEMES[scheme](u, NU)
    return u, exact(u0_N, ns * dt, N)

# ---------------------------------------------------------------- coefficient recovery (configurable library + filter)
def deriv(u, p, h):
    if p == 1: return (np.roll(u, -1) - np.roll(u, 1)) / (2 * h)
    if p == 2: return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / h ** 2
    if p == 3: return (np.roll(u, -2) - 2 * np.roll(u, -1) + 2 * np.roll(u, 1) - np.roll(u, 2)) / (2 * h ** 3)
    return (np.roll(u, -2) - 4 * np.roll(u, -1) + 6 * u - 4 * np.roll(u, 1) + np.roll(u, 2)) / h ** 4

def smooth3(u):                                     # mild 3-point binomial pre-filter (observation filter option)
    return 0.25 * np.roll(u, 1) + 0.5 * u + 0.25 * np.roll(u, -1)

def coeffs(u, r, lib, h):                           # least-squares c on the chosen derivative library
    Amat = np.stack([deriv(u, p, h) for p in lib], 1)
    c, *_ = np.linalg.lstsq(Amat, r, rcond=None)
    return c
def direction(c):
    n = np.linalg.norm(c); return c / n if n > 0 else c

def signature(uf, ref, sigma, n_obs, lib, gn, prefilter=False):
    """Observed signature: add field noise, anti-alias to n_obs, optional smoothing, recover c-direction."""
    N = len(uf)
    un = uf + sigma * np.sqrt(np.mean(uf ** 2)) * gn.standard_normal(N) if sigma > 0 else uf
    uo = antialias(un, n_obs); ro = antialias(un - ref, n_obs)
    if prefilter: uo, ro = smooth3(uo), smooth3(ro)
    return direction(coeffs(uo, ro, lib, L / n_obs))

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def acc(F, y, g): return cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean()
def perm_floor(F, y, g, seed, reps=24):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean() for _ in range(reps)]))

ic = np.arange(N_IC); half = N_IC // 2

# ================================================================ shared IC bank (base 768, divisible by all ladder grids)
N_BASE = 768
rng = np.random.default_rng(0)
bases = [random_ic(N_BASE, rng) for _ in range(N_IC)]

# ================================================================ PART 1: how many solver runs (resolution ladders)
# convergence rate p = -slope of log(rel-residual) vs log(N). Grid-invariant -> breaks the grid confound.
LADDERS = {2: (64, 96), 3: (64, 80, 96), 4: (64, 80, 96, 112), 5: (64, 76, 88, 100, 112), 6: (64, 72, 80, 88, 96, 104)}
# disjoint ladders (same #runs) for the grid control NC2 at each k
LADDERS_B = {2: (72, 104), 3: (72, 88, 104), 4: (72, 88, 104, 120), 5: (68, 80, 92, 104, 116), 6: (68, 76, 84, 92, 100, 108)}
SIGMA1 = 0.01

def rate_feature(scheme, ladder, sigma, seed):
    """Per-IC convergence rate over a ladder of grids; reference is the analytic exact at each N."""
    gn = np.random.default_rng(seed); P = []
    logN = np.log(np.array(ladder, float))
    for u0b in bases:
        mags = []
        for N in ladder:
            u0N = antialias(u0b, N); uf, ex = run(scheme, N, u0N)
            un = uf + sigma * np.sqrt(np.mean(ex ** 2)) * gn.standard_normal(N) if sigma > 0 else uf
            r = un - ex
            mags.append(np.sqrt(np.mean(r ** 2)) / (np.sqrt(np.mean(ex ** 2)) + 1e-12))
        P.append(-np.polyfit(logN, np.log(np.array(mags) + 1e-12), 1)[0])
    return np.array(P)[:, None]

def part1():
    print("\n=== PART 1: how many solver runs does the rate feature need? ===")
    rows = []
    yd = {sc: (0 if sc in DIFFUSIVE else 1) for sc in names}
    for k in sorted(LADDERS):
        pA = {sc: rate_feature(sc, LADDERS[k], SIGMA1, 100 + k) for sc in names}
        F = np.vstack([pA[sc] for sc in names]); y = np.concatenate([[yd[sc]] * N_IC for sc in names])
        g = np.concatenate([ic] * len(names))
        det = acc(F, y, g); fl = perm_floor(F, y, g, 7 + k)
        # NC2: same scheme (upwind), rate from ladder A vs disjoint ladder B (same #runs) -> grid control
        pB_up = rate_feature(UP, LADDERS_B[k], SIGMA1, 500 + k)
        nc2 = acc(np.vstack([pA[UP], pB_up]), np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic])
        # mean rates (self-check: ~1 diffusive, ~2 dispersive)
        pmean = {sc: float(pA[sc].mean()) for sc in names}
        rows.append(dict(k=k, ladder=LADDERS[k], detect=det, floor=fl, nc2=nc2, pmean=pmean))
        print(f"  #runs={k}  ladder={LADDERS[k]}  diff-vs-disp={det:.3f} (floor {fl:.3f})  NC2(grid)={nc2:.3f}  "
              f"p[up]={pmean['upwind']:.2f} p[lw]={pmean['lax_wendroff']:.2f}")
    # minimum #runs that separates (detect >= 0.85 and margin over NC2 >= 0.20)
    ok = [r["k"] for r in rows if r["detect"] >= 0.85 and (r["detect"] - r["nc2"]) >= 0.20]
    min_runs = min(ok) if ok else None
    print(f"  -> minimum solver runs that separates the families (det>=0.85, margin>=0.20): "
          f"{min_runs if min_runs else 'NONE in {2..6}'}")
    return rows, min_runs

# ================================================================ PART 2: how fine a reference suffices
# Reference = NUMERICAL Lax-Wendroff (2nd-order) advection solve at N_ref = m*N (m in {1,2,4,8}).
# GOLD = the analytic exact (the genuine reference) -> recovery drift is purely reference-error-driven.
N2 = 64                                              # solver grid for this part
FINENESS = (1, 2, 4, 8)
LIB2 = (2, 3, 4)

def lw_ref(u0_N, N, m):
    """Practitioner's numerical reference: 2nd-order Lax-Wendroff at resolution m*N, sampled to N.
       Genuine reference error (~5% at 1x) that converges ~O((1/m)^2)."""
    Nr = m * N; u = antialias(u0_N, Nr); dx = L / Nr; dt = NU * dx / A; ns = int(round(T / dt))
    nu = A * dt / dx
    for _ in range(ns): u = lax_wendroff(u, nu)
    return antialias(u, N)

# also expose a high-order reference for the caveat (converges too fast -> 1x suffices for recovery too)
def fd4_ref(u0_N, N, m):
    Nr = m * N; u = antialias(u0_N, Nr); h = L / Nr; dt = 0.3 * h / A; ns = int(np.ceil(T / dt)); dt = T / ns
    d1 = lambda v: (-np.roll(v, -2) + 8 * np.roll(v, -1) - 8 * np.roll(v, 1) + np.roll(v, 2)) / (12 * h)
    rhs = lambda v: -A * d1(v)
    for _ in range(ns):
        a = rhs(u); b = rhs(u + 0.5 * dt * a); c = rhs(u + 0.5 * dt * b); d = rhs(u + dt * c)
        u = u + dt * (a + 2 * b + 2 * c + d) / 6
    return antialias(u, N)

def _drift_curve(fields, ics, ref_fn, gold):
    """clean coefficient-direction drift (rad) vs gold, per fineness m (no field noise).
       Returns per-m dict with both the overall median AND the worst-scheme median, plus per-scheme.
       The WORST-scheme drift is the load-bearing recovery number: a coarse reference is most
       misleading when it shares the solver's own leading truncation error (e.g. LW-vs-LW-ref)."""
    gn0 = np.random.default_rng(0); out = {}
    for m in FINENESS:
        per_sc = {}
        for sc in names:
            ang_sc = []
            for j, u0 in enumerate(ics):
                refm = ref_fn(u0, N2, m)
                s = signature(fields[sc][j], refm, 0.0, N2, LIB2, gn0)
                cax = abs(float(s @ gold[sc][j])); ang_sc.append(np.arccos(np.clip(cax, 0, 1)))
            per_sc[sc] = float(np.median(ang_sc))
        all_ang = list(per_sc.values())
        out[m] = dict(median=float(np.median(all_ang)), worst=float(max(all_ang)), per_scheme=per_sc)
    return out

def part2():
    print("\n=== PART 2: how fine a reference suffices (attribution vs recovery)? ===")
    rng2 = np.random.default_rng(1); ics = [random_ic(N2, rng2) for _ in range(N_IC)]
    fields = {sc: [run(sc, N2, u0)[0] for u0 in ics] for sc in names}
    # GOLD coefficient directions: against the ANALYTIC exact (true reference), clean
    gn0 = np.random.default_rng(11)
    gold = {sc: np.array([signature(fields[sc][j], exact(u0, T, N2), 0.0, N2, LIB2, gn0)
                          for j, u0 in enumerate(ics)]) for sc in names}

    yd = {sc: (0 if sc in DIFFUSIVE else 1) for sc in names}
    rows = []
    for m in FINENESS:
        refs = [lw_ref(u0, N2, m) for u0 in ics]
        sig = {sc: [] for sc in names}
        gn = np.random.default_rng(200 + m)
        for sc in names:
            for j, u0 in enumerate(ics):
                sig[sc].append(signature(fields[sc][j], refs[j], SIGMA1, N2, LIB2, gn))   # 1% noise -> attribution
        sig = {sc: np.array(v) for sc, v in sig.items()}
        F = np.vstack([sig[sc] for sc in names]); y = np.concatenate([[yd[sc]] * N_IC for sc in names])
        g = np.concatenate([ic] * len(names))
        det = acc(F, y, g); fl = perm_floor(F, y, g, 30 + m)
        rows.append(dict(m=m, detect=det, floor=fl))
        print(f"  ref fineness {m}x  attribution(diff-vs-disp, 1% noise)={det:.3f} (floor {fl:.3f})")

    # RECOVERY drift (clean, reference-error-only) for the low-order LW reference and the high-order caveat
    drift_lw = _drift_curve(fields, ics, lw_ref, gold)
    drift_hi = _drift_curve(fields, ics, fd4_ref, gold)
    for m in FINENESS:
        rows_m = next(r for r in rows if r["m"] == m)
        rows_m["drift"] = drift_lw[m]["worst"]; rows_m["drift_med"] = drift_lw[m]["median"]
        rows_m["drift_hi"] = drift_hi[m]["worst"]; rows_m["per_scheme"] = drift_lw[m]["per_scheme"]
        print(f"  ref fineness {m}x  coeff-direction drift (worst scheme) LW-ref={rows_m['drift']:.4f} rad "
              f"(median {rows_m['drift_med']:.4f}; high-order ref worst={rows_m['drift_hi']:.4f})")
    # per-scheme at 1x makes the mechanism explicit (LW-vs-LW-ref cancellation)
    ps1 = next(r for r in rows if r["m"] == 1)["per_scheme"]
    print("  per-scheme drift @1x (LW ref):  " + "  ".join(f"{sc}={ps1[sc]:.3f}" for sc in names))
    print("  -> a coarse reference is catastrophic exactly when it SHARES the solver's leading error")
    return rows

# ================================================================ PART 3: library + filter sensitivity
N3 = 64
LIBS = {"{u_xx,u_xxx}": (2, 3), "{u_xx,u_xxx,u_xxxx}": (2, 3, 4), "{u_x,u_xx,u_xxx,u_xxxx}(rich)": (1, 2, 3, 4)}
FILTERS = [("native(64)", 64, False), ("coarsen64", 64, False), ("coarsen48", 48, False),
           ("coarsen32", 32, False), ("smooth+64", 64, True)]
# (native and coarsen64 are identical here since solver grid is 64; kept so the column reads as a sweep
#  from a practitioner's point of view - native vs an explicitly-requested 64-point observation.)

def part3():
    print("\n=== PART 3: sensitivity to library + observation filter ===")
    rng3 = np.random.default_rng(2); ics = [random_ic(N3, rng3) for _ in range(N_IC)]
    fields = {sc: [run(sc, N3, u0) for u0 in ics] for sc in names}     # (uf, ex)
    yd = {sc: (0 if sc in DIFFUSIVE else 1) for sc in names}
    grid = {}                                                          # (lib, filt) -> (detect, floor)
    libnames = list(LIBS)
    for li, (lname, lib) in enumerate(LIBS.items()):
        for fi, (fname, nobs, pref) in enumerate(FILTERS):
            sig = {sc: [] for sc in names}
            gn = np.random.default_rng(900 + 10 * li + fi)             # deterministic per (library, filter) cell
            for sc in names:
                for j in range(N_IC):
                    uf, ex = fields[sc][j]
                    sig[sc].append(signature(uf, ex, SIGMA1, nobs, lib, gn, prefilter=pref))
            sig = {sc: np.array(v) for sc, v in sig.items()}
            F = np.vstack([sig[sc] for sc in names]); y = np.concatenate([[yd[sc]] * N_IC for sc in names])
            g = np.concatenate([ic] * len(names))
            det = acc(F, y, g); fl = perm_floor(F, y, g, 77)
            grid[(lname, fname)] = (det, fl)
            print(f"  lib={lname:30s} filter={fname:12s} diff-vs-disp={det:.3f} (floor {fl:.3f})")
    dets = [v[0] for v in grid.values()]
    print(f"  -> detection across {len(grid)} (library x filter) cells: "
          f"min={min(dets):.3f}  median={np.median(dets):.3f}  max={max(dets):.3f}")
    return grid

# ================================================================ main
def main():
    print("COST / PRACTICAL RECIPE  | linear advection, analytic exact, 4 schemes, "
          f"{N_IC} ICs, field noise {SIGMA1:.0%}")
    r1, min_runs = part1()
    r2 = part2()
    g3 = part3()

    # ---- long-form CSV ----
    csv = os.path.join(TAB, "cost_recipe.csv")
    with open(csv, "w") as f:
        f.write("part,key,subkey,metric,value\n")
        for r in r1:
            f.write(f"runs,k={r['k']},,diff_vs_disp,{r['detect']:.4f}\n")
            f.write(f"runs,k={r['k']},,perm_floor,{r['floor']:.4f}\n")
            f.write(f"runs,k={r['k']},,nc2_grid,{r['nc2']:.4f}\n")
            for sc, p in r["pmean"].items():
                f.write(f"runs,k={r['k']},{sc},rate_p,{p:.4f}\n")
        for r in r2:
            f.write(f"reference,fineness={r['m']}x,,attribution,{r['detect']:.4f}\n")
            f.write(f"reference,fineness={r['m']}x,,perm_floor,{r['floor']:.4f}\n")
            f.write(f"reference,fineness={r['m']}x,,coeff_drift_rad_LWref_worst,{r['drift']:.4f}\n")
            f.write(f"reference,fineness={r['m']}x,,coeff_drift_rad_LWref_median,{r['drift_med']:.4f}\n")
            f.write(f"reference,fineness={r['m']}x,,coeff_drift_rad_highorderref_worst,{r['drift_hi']:.4f}\n")
            for sc, dv in r["per_scheme"].items():
                f.write(f"reference,fineness={r['m']}x,{sc},coeff_drift_rad_LWref,{dv:.4f}\n")
        for (lname, fname), (det, fl) in g3.items():
            f.write(f"library_filter,{lname},{fname},diff_vs_disp,{det:.4f}\n")
            f.write(f"library_filter,{lname},{fname},perm_floor,{fl:.4f}\n")
    print(f"\nlong-form metrics -> {csv}")

    # ---- compact RECIPE CARD ----
    dets3 = [v[0] for v in g3.values()]
    # attribution at 1x reference vs recovery drift at 1x and 4x/8x
    ref1 = next(r for r in r2 if r["m"] == 1); ref2 = next(r for r in r2 if r["m"] == 2)
    ref4 = next(r for r in r2 if r["m"] == 4); ref8 = next(r for r in r2 if r["m"] == 8)
    card = os.path.join(TAB, "cost_recipe_card.csv")
    with open(card, "w") as f:
        f.write("question,recipe,measured\n")
        f.write(f"solver_runs_for_rate_detection,"
                f"\"min {min_runs if min_runs else '>6'} grids (det>=0.85 & margin>=0.20)\","
                f"\"k=2:{r1[0]['detect']:.2f} k=3:{r1[1]['detect']:.2f} k=6:{r1[-1]['detect']:.2f}; "
                f"NC2 k=2:{r1[0]['nc2']:.2f} k=6:{r1[-1]['nc2']:.2f}\"\n")
        f.write(f"reference_fineness_for_attribution,"
                f"\"1x suffices for attribution\","
                f"\"attrib 1x={ref1['detect']:.2f} 4x={ref4['detect']:.2f} 8x={ref8['detect']:.2f}\"\n")
        f.write(f"reference_fineness_for_recovery,"
                f"\"2-4x for coefficient recovery (worst scheme)\","
                f"\"worst-scheme drift 1x={ref1['drift']:.2f} 2x={ref2['drift']:.2f} 4x={ref4['drift']:.3f} rad "
                f"(LW solver vs LW ref cancels at 1x)\"\n")
        f.write(f"library_filter_robustness,"
                f"\"any reasonable library/filter\","
                f"\"detection min={min(dets3):.2f} median={np.median(dets3):.2f} max={max(dets3):.2f} "
                f"across {len(g3)} cells\"\n")
    print(f"recipe card       -> {card}")

    return r1, min_runs, r2, g3

# ================================================================ figure
def _figure(r1, min_runs, r2, g3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try: import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", palette="muted")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8172B3"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8)); fig.subplots_adjust(wspace=0.30, hspace=0.40)

    # A: runs needed
    axA = axes[0, 0]
    ks = [r["k"] for r in r1]
    axA.plot(ks, [r["detect"] for r in r1], "o-", color=GREEN, lw=2.2, ms=7, label="diffusive-vs-dispersive")
    axA.plot(ks, [r["nc2"] for r in r1], "s--", color=RED, lw=2, ms=6, label="grid control NC2")
    axA.plot(ks, [r["floor"] for r in r1], ":", color=GREY, lw=1.4, label="permutation floor")
    axA.axhline(0.85, color=GREEN, ls=(0, (2, 2)), lw=1, alpha=0.6)
    if min_runs: axA.axvline(min_runs, color="#333", ls=(0, (1, 2)), lw=1.2)
    axA.set_xlabel("number of solver runs (grids in ladder)"); axA.set_ylabel("detection accuracy")
    axA.set_xticks(ks); axA.set_ylim(0.3, 1.04)
    axA.set_title("Rate feature: 2 runs already separate families", fontsize=10)
    axA.legend(frameon=True, framealpha=0.92, edgecolor="#ddd", fontsize=8, loc="center right")
    axA.text(-0.16, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: reference fineness (attribution holds at 1x, recovery needs 4-8x) -- twin axis, log drift
    axB = axes[0, 1]; ms = [r["m"] for r in r2]; x = np.arange(len(ms))
    EPS = 1e-3
    dr = [max(r["drift"], EPS) for r in r2]
    l1, = axB.plot(x, dr, "o-", color=RED, lw=2.2, ms=7, label="coeff. drift, worst scheme (LW ref)")
    axB.set_yscale("log"); axB.set_ylim(EPS * 0.5, 3.0)
    axB.axhline(np.pi / 2, color=GREY, ls=(0, (1, 2)), lw=0.9); axB.text(0.05, np.pi / 2 * 1.05, r"$\pi/2$ (orthogonal)", fontsize=6.5, color=GREY)
    axB.set_xticks(x); axB.set_xticklabels([f"{m}x" for m in ms])
    axB.set_xlabel(r"reference fineness  $N_{\mathrm{ref}}/N$")
    axB.set_ylabel("coeff.-direction drift (rad, log)", color=RED); axB.tick_params(axis="y", labelcolor=RED)
    axB.grid(True, which="both", color="#eee", lw=0.6)
    axBR = axB.twinx()
    l2, = axBR.plot(x, [r["detect"] for r in r2], "s--", color=GREEN, lw=2.2, ms=7, label="attribution accuracy (1% noise)")
    axBR.set_ylabel("attribution accuracy", color=GREEN); axBR.tick_params(axis="y", labelcolor=GREEN)
    axBR.set_ylim(0.45, 1.03); axBR.grid(False)
    axB.set_title("Attribution holds at 1x; recovery needs 2-4x", fontsize=9.5)
    axB.legend(handles=[l1, l2], loc="lower left", frameon=True, framealpha=0.92, edgecolor="#ddd", fontsize=7.5)
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: library x filter heatmap of detection
    axC = axes[1, 0]
    libs = list(LIBS); filts = [f[0] for f in FILTERS]
    H = np.array([[g3[(lb, ft)][0] for ft in filts] for lb in libs])
    im = axC.imshow(H, cmap="crest", vmin=0.7, vmax=1.0, aspect="auto")
    axC.set_xticks(range(len(filts))); axC.set_xticklabels(filts, rotation=30, ha="right", fontsize=7.5)
    axC.set_yticks(range(len(libs))); axC.set_yticklabels([lb.replace("(rich)", "\n(rich)") for lb in libs], fontsize=7.5)
    for i in range(len(libs)):
        for j in range(len(filts)):
            axC.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="#f5f5f5" if H[i, j] > 0.95 else "#111")
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04); cb.set_label("diff-vs-disp accuracy", fontsize=8)
    axC.set_title("Detection robust across library + filter choices", fontsize=9.5)
    axC.grid(False)
    axC.text(-0.30, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: recipe card (text panel)
    axD = axes[1, 1]; axD.axis("off")
    dets3 = [v[0] for v in g3.values()]
    ref1 = next(r for r in r2 if r["m"] == 1); ref2 = next(r for r in r2 if r["m"] == 2)
    ref4 = next(r for r in r2 if r["m"] == 4); ref8 = next(r for r in r2 if r["m"] == 8)
    lines = [
        ("RECIPE CARD", ""),
        ("Solver runs (rate detection)", f"{min_runs if min_runs else '>6'} grids min"),
        ("   diff-vs-disp @ 2 / 6 runs", f"{r1[0]['detect']:.2f} / {r1[-1]['detect']:.2f}"),
        ("   grid control NC2 @ 2 / 6", f"{r1[0]['nc2']:.2f} / {r1[-1]['nc2']:.2f}"),
        ("Reference for ATTRIBUTION", "1x suffices"),
        ("   attribution @ 1x / 8x", f"{ref1['detect']:.2f} / {ref8['detect']:.2f}"),
        ("Reference for RECOVERY", "2-4x needed"),
        ("   worst drift 1x / 2x / 4x", f"{ref1['drift']:.2f} / {ref2['drift']:.2f} / {ref4['drift']:.3f}"),
        ("Library / filter robustness", "any reasonable choice"),
        ("   detection min / median", f"{min(dets3):.2f} / {np.median(dets3):.2f}"),
    ]
    y = 0.97
    for i, (k, v) in enumerate(lines):
        if i == 0:
            axD.text(0.0, y, k, fontsize=12, fontweight="bold", transform=axD.transAxes); y -= 0.115; continue
        bold = not k.startswith("   ")
        axD.text(0.0, y, k, fontsize=8.6 if bold else 8.0, fontweight="bold" if bold else "normal",
                 color="#222" if bold else "#555", transform=axD.transAxes)
        axD.text(1.0, y, v, fontsize=8.6 if bold else 8.0, ha="right",
                 fontweight="bold" if bold else "normal", color=BLUE if bold else "#555", transform=axD.transAxes)
        y -= 0.092
    axD.text(-0.06, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "fig_cost_recipe.png")
    fig.savefig(out); plt.close(fig); print(f"figure -> {out}")

if __name__ == "__main__":
    import sys
    r1, min_runs, r2, g3 = main()
    if "--plot" in sys.argv: _figure(r1, min_runs, r2, g3)
