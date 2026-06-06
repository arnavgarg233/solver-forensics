"""
solver-forensics :: IRREGULAR-MESH NOVELTY GATE
====================================================================
Feasibility is settled across linear -> grid/CFL transfer -> smooth nonlinear.
The BINDING constraint is now novelty-vs-SITE, and exactly one experiment
touches it: irregular / scattered-node observation. Strong-form finite
differences NEED a grid; weak-form integration does not. If weak-form holds
where a COMPETENT strong-form baseline breaks, "weak-form" earns its name and
a place in the title. If they tie, weak-form is an implementation detail and
the paper is the application contribution (attribution / auditability).

FAIRNESS (load-bearing - do not strawman strong-form):
  A. interpolate-then-FD : periodic cubic spline of the scattered samples onto a
     uniform grid, then the SAME FD coefficient recovery. This is what a competent
     person applying SITE to scattered data does, and the SITE reviewer will demand
     it. Weak-form must beat THIS, not raw-FD-on-scattered-nodes.
  B. MLS / local-poly    : moving-least-squares degree-4 derivatives at each node.
     Strong-form's best shot on scattered data.
  C. weak-form quadrature: compact test functions, weak integrals estimated by
     periodic Voronoi quadrature directly on the raw nodes. Never differentiates.

ISOLATION: linear advection with the ANALYTIC exact solution -> no reference to
contaminate, so on a uniform grid all three methods are ~perfect and the ONLY
moving part is grid-vs-scattered. Held fixed at validated settings (CFL=0.6,
same 4 schemes, 1% FIELD-relative noise, 64-node budget). Severity is the sweep.

BANKED CAVEAT: c3-sign (LW vs BW) is the finer, lower-energy feature, so if
anything degrades on scattered nodes it degrades FIRST, for every method. The
differentiator CLAIM therefore lives on diffusive-vs-dispersive (the robust,
defensible channel); LW-vs-BW surviving is upside, not the central claim.

PRE-REGISTERED DECISION (read off diffusive-vs-dispersive at the SEVERE level):
  weak-form holds while interpolate-then-FD drops materially  -> weak-form earns its name; central; title can say so
  all three hold or all three drop together                   -> weak-form NOT central; application contribution, specialist tier
  a strong-form baseline beats weak-form                      -> drop weak-form framing; attribution-via-coefficients paper
  weak-form also fails on diff-vs-disp                        -> irregular observation outside the current auditability regime

Pure numpy + scipy + sklearn, CPU, a few minutes on M2. No accuracy hardcoded.
"""
import os
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings; warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
L, a, NU, T, N_C = 1.0, 1.0, 0.6, 0.30, 256
M = 64                      # node budget (matches the validated anti-aliased coarseness)
SIGMA = 0.01                # field-relative noise (validated level)
N_IC = 80

# ---- linear advection: analytic exact + the validated 4 schemes (verbatim, nu param) ----
def exact_advection(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
    return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*a*t), n=N)
def upwind(u, nu):         return u - nu*(u - np.roll(u,1))
def lax_friedrichs(u, nu): return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))
def lax_wendroff(u, nu):   return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))
def beam_warming(u, nu):   return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))
SCHEMES = {"upwind":upwind, "lax_friedrichs":lax_friedrichs,
           "lax_wendroff":lax_wendroff, "beam_warming":beam_warming}
names = list(SCHEMES)
DIFFUSIVE = {"upwind","lax_friedrichs"}
LW, BW = names.index("lax_wendroff"), names.index("beam_warming")

def random_ic(N, rng, n_modes=6):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        kk = rng.integers(1, 8)
        u += rng.normal()*np.sin(2*np.pi*kk*x/L + rng.uniform(0, 2*np.pi))
    if rng.random() < 0.7:
        x0, w = rng.uniform(0, L), rng.uniform(L*0.02, L*0.08)
        u += rng.normal()*np.exp(-(((x-x0+L/2) % L - L/2)**2)/(2*w*w))
    return u

dx = L/N_C; dt = NU*dx/a; NSTEPS = int(round(T/dt))
def run_scheme(name, u0):
    u = u0.copy()
    for _ in range(NSTEPS): u = SCHEMES[name](u, NU)
    return u

# ================================================================ scattered nodes
def make_nodes(jit, miss, warp, rng):
    """Severe irregular nodes: smooth monotone warp (strong spacing variation) + jitter + missingness."""
    t = np.arange(M)/M
    x = t + warp*np.sin(2*np.pi*t + rng.uniform(0, 2*np.pi))      # spacing varies ~10x at warp=0.15
    x = x + jit*(1.0/M)*(rng.random(M) - 0.5)
    x = np.sort(x % L)
    if miss > 0:
        keep = rng.random(len(x)) > miss
        if keep.sum() < 16:                                       # floor so a method can still fit
            keep[np.argsort(rng.random(len(x)))[:16]] = True
        x = x[keep]
    x = np.unique(np.round(x, 9))
    return np.sort(x % L)

def periodic_spline(x, vals):
    xs = np.concatenate([x, [x[0] + L]]); vs = np.concatenate([vals, [vals[0]]])
    return CubicSpline(xs, vs, bc_type="periodic")

# ================================================================ three coefficient estimators
def bump_polys(q=6):
    p0 = Polynomial([1.0, 0.0, -1.0])**q
    return [p0.deriv(k) for k in range(5)]
POLY = bump_polys()
CENTERS = np.linspace(0, L, 16, endpoint=False)
WIDTHS = (0.10, 0.15, 0.22)

def c_weakform(x, u, r):                                          # C: quadrature on RAW nodes
    o = np.argsort(x); x, u, r = x[o], u[o], r[o]
    xe = np.concatenate([[x[-1] - L], x, [x[0] + L]])
    w = 0.5*(xe[2:] - xe[:-2])                                    # periodic Voronoi quadrature weights
    A, b = [], []
    for cx in CENTERS:
        for wd in WIDTHS:
            s = (((x - cx + L/2) % L) - L/2)/wd
            m = np.abs(s) <= 1.0
            if m.sum() < 5: continue
            se = np.where(m, s, 0.0)
            phi = POLY[0](se)*m; d2 = POLY[2](se)/wd**2*m
            d3 = POLY[3](se)/wd**3*m; d4 = POLY[4](se)/wd**4*m
            b.append(np.sum(w*r*phi))
            A.append([np.sum(w*u*d2), -np.sum(w*u*d3), np.sum(w*u*d4)])
    if len(b) < 6: return np.zeros(3), False
    c, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return c, True

def c_interp_fd(x, u, r):                                         # A: spline to uniform grid, then FD
    try:
        ug = periodic_spline(x, u)(np.linspace(0, L, M, endpoint=False))
        rg = periodic_spline(x, r)(np.linspace(0, L, M, endpoint=False))
    except Exception:
        return np.zeros(3), False
    h = L/M
    uxx   = (np.roll(ug,-1) - 2*ug + np.roll(ug,1))/h**2
    uxxx  = (np.roll(ug,-2) - 2*np.roll(ug,-1) + 2*np.roll(ug,1) - np.roll(ug,2))/(2*h**3)
    uxxxx = (np.roll(ug,-2) - 4*np.roll(ug,-1) + 6*ug - 4*np.roll(ug,1) + np.roll(ug,2))/h**4
    A = np.stack([uxx, uxxx, uxxxx], 1)
    c, *_ = np.linalg.lstsq(A, rg, rcond=None)
    return c, True

def c_mls(x, u, r, deg=4, K=11):                                 # B: MLS local-poly derivatives on nodes
    n = len(x)
    if n < deg + 2: return np.zeros(3), False
    K = min(K, n)
    D = np.zeros((n, 3))
    for j in range(n):
        d = ((x - x[j] + L/2) % L) - L/2
        idx = np.argsort(np.abs(d))[:K]; dd = d[idx]
        hh = np.max(np.abs(dd)) + 1e-12
        wsq = np.sqrt(np.exp(-(dd/(0.6*hh))**2))
        V = np.vander(dd, deg + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V*wsq[:, None], u[idx]*wsq, rcond=None)
        D[j] = [2*coef[2], 6*coef[3], 24*coef[4]]
    c, *_ = np.linalg.lstsq(D, r, rcond=None)
    return c, True

ESTIMATORS = {"interp+FD": c_interp_fd, "MLS": c_mls, "weak-form": c_weakform}

# ================================================================ features / metrics
def coeff_features(C):
    unit = C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        r32 = np.clip(np.nan_to_num(C[:,1]/C[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
    return np.nan_to_num(np.hstack([unit, r32[:,None]]))
def cv_acc(X, lab, grp, dist):
    if dist == "diff_disp":
        y, sel = np.array([0 if names[l] in DIFFUSIVE else 1 for l in lab]), np.ones(len(lab), bool)
    else:
        y, sel = lab, np.isin(lab, [LW, BW])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X[sel], y[sel], groups=grp[sel], cv=GroupKFold(5)).mean()

# ================================================================ RUN
LEVELS = [("uniform-64",        dict(jit=0.0,  miss=0.0,  warp=0.0)),
          ("jitter-25%",        dict(jit=0.25, miss=0.0,  warp=0.0)),
          ("jitter-50%",        dict(jit=0.50, miss=0.0,  warp=0.0)),
          ("jit50+miss30%",     dict(jit=0.50, miss=0.30, warp=0.0)),
          ("warp+jit50+miss50", dict(jit=0.50, miss=0.50, warp=0.15))]   # SEVERE
print(f"linear advection, analytic exact, {N_IC} paired ICs, {M}-node budget, field noise {SIGMA:.0%}")
print("severity ladder folds jitter + missingness + spacing-warp into one axis\n")

# clean per-scheme reference directions (uniform 64, no noise, weak-form) for angle error
def fine_fields(u0):
    ex = exact_advection(u0, NSTEPS*dt, N_C)
    return {nm: (run_scheme(nm, u0), ex) for nm in names}
ref_rng = np.random.default_rng(123); refC = {nm: [] for nm in names}
xu = np.linspace(0, L, M, endpoint=False)
for _ in range(40):
    u0 = random_ic(N_C, ref_rng); ff = fine_fields(u0)
    for nm in names:
        un, ex = ff[nm]
        u_n = periodic_spline(np.linspace(0,L,N_C,endpoint=False), un)(xu)
        e_n = periodic_spline(np.linspace(0,L,N_C,endpoint=False), ex)(xu)
        c, _ = c_weakform(xu, u_n, u_n - e_n); refC[nm].append(c/(np.linalg.norm(c)+1e-12))
refdir = {nm: np.mean(refC[nm], 0) for nm in names}
refdir = {nm: v/(np.linalg.norm(v)+1e-12) for nm, v in refdir.items()}

xg = np.linspace(0, L, N_C, endpoint=False)
results = {}     # (method, level) -> dict of metrics
for lvl_i, (lname, lp) in enumerate(LEVELS):
    store = {mth: {"C": [], "lab": [], "grp": [], "ok": []} for mth in ESTIMATORS}
    ic_rng = np.random.default_rng(7)                            # SAME ICs across levels
    for ic in range(N_IC):
        u0 = random_ic(N_C, ic_rng); ff = fine_fields(u0)
        nrng = np.random.default_rng(20_000 + lvl_i*1000 + ic)  # deterministic node geometry per (level, ic)
        nodes = make_nodes(lp["jit"], lp["miss"], lp["warp"], nrng)
        ex_nodes = periodic_spline(xg, ff[names[0]][1])(nodes)  # exact is shared
        gnoise = np.random.default_rng(5_000 + ic)
        for li, nm in enumerate(names):
            un = ff[nm][0]
            u_nodes = periodic_spline(xg, un)(nodes)
            u_nodes = u_nodes + SIGMA*np.sqrt(np.mean(u_nodes**2))*gnoise.standard_normal(len(nodes))
            r_nodes = u_nodes - ex_nodes
            for mth, fn in ESTIMATORS.items():
                c, ok = fn(nodes, u_nodes, r_nodes)
                store[mth]["C"].append(c); store[mth]["lab"].append(li)
                store[mth]["grp"].append(ic); store[mth]["ok"].append(ok)
    for mth in ESTIMATORS:
        C = np.array(store[mth]["C"]); lab = np.array(store[mth]["lab"])
        grp = np.array(store[mth]["grp"]); ok = np.array(store[mth]["ok"])
        X = coeff_features(C)
        unit = C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        ang = np.array([np.degrees(np.arccos(np.clip(abs(unit[i] @ refdir[names[lab[i]]]), 0, 1)))
                        for i in range(len(lab))])
        lwbw = np.isin(lab, [LW, BW])
        c3sign = np.mean([(C[i,1] < 0) == (lab[i] == LW) for i in np.where(lwbw)[0]])
        results[(mth, lname)] = dict(dd=cv_acc(X, lab, grp, "diff_disp"),
                                     lw=cv_acc(X, lab, grp, "lw_bw"),
                                     ang=np.median(ang), ang90=np.percentile(ang, 90),
                                     out30=np.mean(ang > 30), c3=c3sign, fail=1 - ok.mean())
    print(f"  done: {lname}")

# ================================================================ tables
def row(metric, fmt="{:>7.3f}"):
    return {mth: [fmt.format(results[(mth, ln)][metric]) for ln, _ in LEVELS] for mth in ESTIMATORS}
hdr = "  " + " ".join(f"{ln:>17}" for ln, _ in LEVELS)
for title, metric in [("diffusive-vs-dispersive accuracy (THE channel the claim rides on)", "dd"),
                      ("LW-vs-BW accuracy (c3-sign - upside, expected to degrade first)", "lw"),
                      ("coefficient angle error vs clean grid, degrees (lower=better)", "ang"),
                      ("c3-sign accuracy on LW/BW", "c3")]:
    print(f"\n=== {title} ===")
    print(f"{'method':<11}|{hdr}")
    fmt = "{:>7.1f}" if metric == "ang" else "{:>7.3f}"
    r = row(metric, fmt)
    for mth in ESTIMATORS:
        print(f"{mth:<11}|  " + " ".join(f"{v:>17}" for v in r[mth]))

print("\n=== outlier/variance diagnostic - angle-err median/90th-pct, %samples>30°, solve-fail% ===")
for ln, _ in LEVELS:
    cells = "  ".join(f"{m}: {results[(m,ln)]['ang']:>4.0f}/{results[(m,ln)]['ang90']:>4.0f}° "
                      f"{results[(m,ln)]['out30']*100:>3.0f}%>30 f{results[(m,ln)]['fail']*100:>2.0f}%"
                      for m in ESTIMATORS)
    print(f"{ln:>18} | {cells}")

# ================================================================ CSV + plot
csv = os.path.join(TAB, "irregular_mesh_results.csv")
with open(csv, "w") as f:
    f.write("method,level,diff_disp,lw_bw,angle_err_deg,c3_sign_acc,fail_rate\n")
    for (mth, ln), d in results.items():
        f.write(f"{mth},{ln},{d['dd']:.4f},{d['lw']:.4f},{d['ang']:.2f},{d['c3']:.3f},{d['fail']:.3f}\n")
print(f"\nfull results -> {csv}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = np.arange(len(LEVELS)); fig, ax = plt.subplots(1, 2, figsize=(14, 5))
mark = {"interp+FD":"s--", "MLS":"^--", "weak-form":"o-"}
for j, (metric, ttl) in enumerate([("dd","diffusive-vs-dispersive (load-bearing)"),
                                    ("lw","LW-vs-BW (c3-sign, upside)")]):
    for mth in ESTIMATORS:
        ax[j].plot(xs, [results[(mth, ln)][metric] for ln, _ in LEVELS], mark[mth], label=mth)
    ax[j].axhline(0.5, color="grey", ls=":"); ax[j].axhline(0.9, color="green", ls="--", alpha=.5)
    ax[j].set_xticks(xs); ax[j].set_xticklabels([ln for ln, _ in LEVELS], rotation=30, ha="right", fontsize=7)
    ax[j].set_ylim(0.3, 1.03); ax[j].set_title(ttl); ax[j].set_ylabel("GroupKFold accuracy"); ax[j].legend(fontsize=8)
plt.tight_layout(); plot = os.path.join(FIG, "irregular_mesh_result.png"); plt.savefig(plot, dpi=130)
print(f"plot       -> {plot}")

# ================================================================ pre-registered decision
print("\n" + "="*70 + "\nDECISION  (read off diffusive-vs-dispersive at the SEVERE level)\n" + "="*70)
sev = LEVELS[-1][0]
wk, ifd, mls = (results[("weak-form", sev)]["dd"], results[("interp+FD", sev)]["dd"], results[("MLS", sev)]["dd"])
base_drop = min(ifd, mls)                                        # best strong-form baseline at severe
gap = wk - base_drop
print(f"severe level = '{sev}'")
print(f"diff-vs-disp:  weak-form={wk:.3f}   interp+FD={ifd:.3f}   MLS={mls:.3f}   (best baseline={base_drop:.3f})")
print(f"weak-form LW/BW at severe = {results[('weak-form', sev)]['lw']:.3f}   "
      f"angle-err weak={results[('weak-form', sev)]['ang']:.0f}° vs interp+FD={results[('interp+FD', sev)]['ang']:.0f}°")
if wk > max(ifd, mls) + 0.01 and base_drop < 0.85 and gap >= 0.10:
    print(f"\n[WEAK-FORM EARNS ITS NAME]  weak-form holds diff-vs-disp ({wk:.3f}) while the best competent")
    print(f"   strong-form baseline drops to {base_drop:.3f} (gap {gap:+.3f}). Central to the paper; title can say so.")
    if results[("weak-form", sev)]["lw"] > 0.85:
        print("   BONUS: LW/BW also survives on weak-form -> taxonomy holds on irregular nodes too.")
elif max(ifd, mls) > wk + 0.02:
    print(f"\n[STRONG-FORM WINS]  a competent baseline ({max(ifd,mls):.3f}) beats weak-form ({wk:.3f}).")
    print("   Drop the weak-form framing; frame as black-box attribution via modified-equation coefficients.")
elif min(wk, ifd, mls) < 0.65:
    print(f"\n[BOTH FAIL]  even diff-vs-disp collapses at severe scattering (weak={wk:.3f}, best base={base_drop:.3f}).")
    print("   Irregular observation is outside the current auditability regime; report as a measured limit.")
else:
    print(f"\n[TIE]  weak-form ({wk:.3f}) and the best baseline ({base_drop:.3f}) hold/drop together (gap {gap:+.3f}).")
    print("   Weak-form has no methodological edge here -> scope it as an implementation detail. The paper is")
    print("   the application contribution (attribution, magnitude-invariant, grid/CFL-robust, coarse+noisy);")
    print("   realistic tier stays specialist-family.")
