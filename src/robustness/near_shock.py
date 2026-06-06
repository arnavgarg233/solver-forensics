"""
solver-forensics :: NEAR-SHOCK IRREGULAR GATE  (TERMINAL)
==============================================================================
The smooth-field irregular gate showed weak-form has NO edge when u is smooth:
a cubic spline reconstructs a band-limited field near-perfectly from 32 scattered
nodes, so interpolate-then-FD never pays the interpolation cost the weak-form
hypothesis is about. That test was structurally unable to exercise the mechanism.

This is the decisive, BOUNDED, TERMINAL completion: the one regime where the
mechanism can bite - a STEEP front (thin viscous shock) sampled on SEVERELY
irregular nodes, where interpolating u smears the gradient and strong-form's
differentiation breaks, while weak-form only integrates.

PRE-REGISTERED (fixed before any number is seen):
  * DECISION METRIC: diffusive-vs-dispersive classification accuracy,
    weak-form vs interpolate-then-FD, ONLY. (LW/BW and angle = context, NOT
    decision-driving - the angle metric goes wild near a shock.)
  * WIN: weak_dd - interp_dd >= +0.10 at the SEVERE near-shock level. It trailed
    by ~0.10 on smooth fields; it must cross to a REAL lead, not parity.
  * KILL: loss OR null -> commit to (A) strong-form attribution paper. ZERO
    further rescue attempts. One bounded test, terminal either way.
  * REGIME VALIDITY (null guard): if scattered-node reconstruction error of u is
    small, the front was not steep relative to the nodes -> the test is NULL and
    the honest default is (A), NOT "inconclusive, run again."

Fairness: interpolate-then-FD (the competent SITE-on-scattered baseline that
must be beaten) + MLS (strong-form's best shot) + weak-form quadrature.

Substrate: viscous Burgers, thin shock (front ~ mu < node spacing), integrated
PAST shock formation. Coarse schemes use operator-split EXACT (spectral)
diffusion (stable on the steep field); spectral IF-RK4 at N_ref=2048 is the
truth, checked against 1024. Pure numpy+scipy+sklearn, CPU.
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
L = 1.0
MU = 0.005                  # thin viscous shock: front ~0.005 < node spacing 1/64=0.0156
N_C = 256                   # coarse "black-box solver" grid
N_REF, N_REF_CHK = 2048, 1024
TFIN = 0.45                 # past shock formation (t*~0.32) -> developed thin shock
CFL_C = CFL_REF = 0.4
M = 64                      # node budget
SIGMA = 0.01
N_IC = 70
WIN_MARGIN = 0.10
METHODS = ["interp+FD", "MLS", "weak-form"]

# ================================================================ physics
def random_ic_shock(N, rng):
    x = np.linspace(0, L, N, endpoint=False)
    return 1.0 + rng.uniform(0.4, 0.6)*np.sin(2*np.pi*(x - rng.uniform(0, L))/L)   # one clean shock

# ---- inviscid advective flux updates (the 4 schemes; diffusion handled separately) ----
def a_upwind(u, dt, dx): f = 0.5*u*u; return u - (dt/dx)*(f - np.roll(f, 1))
def a_lf(u, dt, dx):
    f = 0.5*u*u; Fp = 0.5*(f + np.roll(f,-1)) - 0.5*(dx/dt)*(np.roll(u,-1) - u)
    return u - (dt/dx)*(Fp - np.roll(Fp, 1))
def a_lw(u, dt, dx):
    f = 0.5*u*u; uh = 0.5*(u + np.roll(u,-1)) - 0.5*(dt/dx)*(np.roll(f,-1) - f)
    return u - (dt/dx)*(0.5*uh*uh - np.roll(0.5*uh*uh, 1))
def a_bw(u, dt, dx):
    f = 0.5*u*u; Fp = f + 0.5*(1 - u*dt/dx)*(f - np.roll(f, 1))
    return u - (dt/dx)*(Fp - np.roll(Fp, 1))
ADV = {"upwind":a_upwind, "lax_friedrichs":a_lf, "lax_wendroff":a_lw, "beam_warming":a_bw}
names = list(ADV)
DIFFUSIVE = {"upwind","lax_friedrichs"}
LW, BW = names.index("lax_wendroff"), names.index("beam_warming")

def run_coarse(name, u0):
    N = N_C; dx = L/N; adv = ADV[name]
    k = 2*np.pi*np.fft.rfftfreq(N, d=dx); Ehalf = np.exp(-MU*k*k*0)  # set per-dt below
    umax = np.max(np.abs(u0)) + 1e-9
    nsteps = int(np.ceil(TFIN/(CFL_C*dx/umax))); dt = TFIN/nsteps
    Ehalf = np.exp(-MU*k*k*dt/2)
    u = u0.copy()
    for _ in range(nsteps):                                  # Strang: diffuse/2, advect, diffuse/2
        u = np.fft.irfft(np.fft.rfft(u)*Ehalf, n=N)
        u = adv(u, dt, dx)
        u = np.fft.irfft(np.fft.rfft(u)*Ehalf, n=N)
    return u

def ifrk4(uh, Lhat, adv, dt, nsteps):
    E, E2 = np.exp(Lhat*dt), np.exp(Lhat*dt/2)
    for _ in range(nsteps):
        a = dt*adv(uh); b = dt*adv(E2*(uh + a/2)); c = dt*adv(E2*uh + b/2); d = dt*adv(E*uh + E2*c)
        uh = E*uh + (E*a + 2*E2*(b + c) + d)/6
    return uh
def solve_ref(u0, N):                                        # spectral IF-RK4 truth (resolves the shock)
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N); ik = 1j*k; mask = np.abs(k) <= (2/3)*np.max(np.abs(k))
    umax = np.max(np.abs(u0)) + 1e-9
    nsteps = int(np.ceil(TFIN/(CFL_REF*(L/N)/umax))); dt = TFIN/nsteps
    def adv(uh): u = np.fft.ifft(uh).real; return -0.5*ik*(np.fft.fft(u*u)*mask)
    return np.fft.ifft(ifrk4(np.fft.fft(u0), -MU*k*k, adv, dt, nsteps)).real

# ================================================================ scattered nodes + estimators
def make_nodes(jit, miss, warp, rng):
    t = np.arange(M)/M
    x = t + warp*np.sin(2*np.pi*t + rng.uniform(0, 2*np.pi)) + jit*(1.0/M)*(rng.random(M) - 0.5)
    x = np.sort(x % L)
    if miss > 0:
        keep = rng.random(len(x)) > miss
        if keep.sum() < 16: keep[np.argsort(rng.random(len(x)))[:16]] = True
        x = x[keep]
    return np.sort(np.unique(np.round(x % L, 9)))
def periodic_spline(x, v):
    return CubicSpline(np.concatenate([x, [x[0]+L]]), np.concatenate([v, [v[0]]]), bc_type="periodic")

POLY = [Polynomial([1.0, 0.0, -1.0])**6]; POLY = [POLY[0].deriv(k) for k in range(5)]
CENTERS = np.linspace(0, L, 16, endpoint=False); WIDTHS = (0.10, 0.15, 0.22)
def c_weakform(x, u, r):
    o = np.argsort(x); x, u, r = x[o], u[o], r[o]
    xe = np.concatenate([[x[-1]-L], x, [x[0]+L]]); w = 0.5*(xe[2:] - xe[:-2])
    A, b = [], []
    for cx in CENTERS:
        for wd in WIDTHS:
            s = (((x - cx + L/2) % L) - L/2)/wd; m = np.abs(s) <= 1.0
            if m.sum() < 5: continue
            se = np.where(m, s, 0.0)
            b.append(np.sum(w*r*(POLY[0](se)*m)))
            A.append([np.sum(w*u*(POLY[2](se)/wd**2*m)), -np.sum(w*u*(POLY[3](se)/wd**3*m)),
                      np.sum(w*u*(POLY[4](se)/wd**4*m))])
    if len(b) < 6: return np.zeros(3), False
    c, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None); return c, True
def c_interp_fd(x, u, r):
    xu = np.linspace(0, L, M, endpoint=False)
    try: ug, rg = periodic_spline(x, u)(xu), periodic_spline(x, r)(xu)
    except Exception: return np.zeros(3), False
    h = L/M
    A = np.stack([(np.roll(ug,-1)-2*ug+np.roll(ug,1))/h**2,
                  (np.roll(ug,-2)-2*np.roll(ug,-1)+2*np.roll(ug,1)-np.roll(ug,2))/(2*h**3),
                  (np.roll(ug,-2)-4*np.roll(ug,-1)+6*ug-4*np.roll(ug,1)+np.roll(ug,2))/h**4], 1)
    c, *_ = np.linalg.lstsq(A, rg, rcond=None); return c, True
def c_mls(x, u, r, deg=4, K=11):
    n = len(x)
    if n < deg + 2: return np.zeros(3), False
    K = min(K, n); D = np.zeros((n, 3))
    for j in range(n):
        d = ((x - x[j] + L/2) % L) - L/2; idx = np.argsort(np.abs(d))[:K]; dd = d[idx]
        wsq = np.sqrt(np.exp(-(dd/(0.6*(np.max(np.abs(dd))+1e-12)))**2))
        coef, *_ = np.linalg.lstsq(np.vander(dd, deg+1, increasing=True)*wsq[:,None], u[idx]*wsq, rcond=None)
        D[j] = [2*coef[2], 6*coef[3], 24*coef[4]]
    c, *_ = np.linalg.lstsq(D, r, rcond=None); return c, True
EST = {"interp+FD": c_interp_fd, "MLS": c_mls, "weak-form": c_weakform}

def coeff_features(C):
    unit = C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        r32 = np.clip(np.nan_to_num(C[:,1]/C[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
    return np.nan_to_num(np.hstack([unit, r32[:,None]]))
def cv_acc(X, lab, grp, dist):
    if dist == "diff_disp": y, sel = np.array([0 if names[l] in DIFFUSIVE else 1 for l in lab]), np.ones(len(lab), bool)
    else: y, sel = lab, np.isin(lab, [LW, BW])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X[sel], y[sel], groups=grp[sel], cv=GroupKFold(5)).mean()

# ================================================================ RUN
print(f"near-shock viscous Burgers (mu={MU}, front~{MU} < node spacing {1/M:.4f}), past shock t*~0.32")
print(f"coarse N={N_C}, truth=spectral N_ref={N_REF}, {N_IC} ICs, {M}-node budget, field noise {SIGMA:.0%}\n")

xg, xr = np.linspace(0, L, N_C, endpoint=False), np.linspace(0, L, N_REF, endpoint=False)
xrc = np.linspace(0, L, N_REF_CHK, endpoint=False)

# precompute per-IC: coarse fields (4 schemes) + reference (2048) + reference check (1024)
ic_rng = np.random.default_rng(7)
DATA = []   # per IC: dict(u0, coarse={name:field256}, ref2048, ref1024)
ref_drift = []
for ic in range(N_IC):
    u0 = random_ic_shock(N_C, ic_rng)
    coarse = {nm: run_coarse(nm, u0) for nm in names}
    # same continuous IC at fine res (band-limited single sine -> spline resample is exact)
    ref = solve_ref(periodic_spline(xg, u0)(xr), N_REF)
    refc = solve_ref(periodic_spline(xg, u0)(xrc), N_REF_CHK)
    DATA.append(dict(coarse=coarse, ref=ref, refc=refc))
    ref_drift.append(np.linalg.norm(periodic_spline(xr, ref)(xrc) - refc)/(np.linalg.norm(refc)+1e-12))
ref_drift = float(np.median(ref_drift))
print(f"reference convergence (1024 vs 2048, median rel L2) = {ref_drift:.4f}  "
      f"({'OK, truth resolves the shock' if ref_drift < 0.03 else 'WARNING: reference under-resolved'})")

LEVELS = [("uniform-64", dict(jit=0.0, miss=0.0, warp=0.0)),
          ("jitter-50%", dict(jit=0.50, miss=0.0, warp=0.0)),
          ("warp+jit50+miss50", dict(jit=0.50, miss=0.50, warp=0.15))]   # SEVERE = decision level

# null guard: how badly does scattered-node interpolation reconstruct the steep field u?
sev = LEVELS[-1]
recon = []
for ic in range(N_IC):
    un = DATA[ic]["coarse"]["upwind"]
    nodes = make_nodes(sev[1]["jit"], sev[1]["miss"], sev[1]["warp"], np.random.default_rng(99 + ic))
    un_nodes = periodic_spline(xg, un)(nodes)
    un_rec = periodic_spline(nodes, un_nodes)(xg)
    recon.append(np.linalg.norm(un_rec - un)/(np.linalg.norm(un)+1e-12))
recon_err = float(np.median(recon))
print(f"null guard: scattered-node reconstruction error of u at SEVERE = {recon_err:.3f}  "
      f"({'mechanism EXERCISED (interp smears the front)' if recon_err > 0.05 else 'NULL: field reconstructs cleanly -> test cannot discriminate'})\n")

results = {}
for lvl_i, (lname, lp) in enumerate(LEVELS):
    store = {m: {"C": [], "lab": [], "grp": []} for m in METHODS}
    for ic in range(N_IC):
        nodes = make_nodes(lp["jit"], lp["miss"], lp["warp"], np.random.default_rng(20000 + lvl_i*333 + ic))
        ex_nodes = periodic_spline(xr, DATA[ic]["ref"])(nodes)
        gn = np.random.default_rng(5000 + ic)
        for li, nm in enumerate(names):
            u_nodes = periodic_spline(xg, DATA[ic]["coarse"][nm])(nodes)
            u_nodes = u_nodes + SIGMA*np.sqrt(np.mean(u_nodes**2))*gn.standard_normal(len(nodes))
            r_nodes = u_nodes - ex_nodes
            for m, fn in EST.items():
                c, _ = fn(nodes, u_nodes, r_nodes)
                store[m]["C"].append(c); store[m]["lab"].append(li); store[m]["grp"].append(ic)
    for m in METHODS:
        X = coeff_features(np.array(store[m]["C"])); lab = np.array(store[m]["lab"]); grp = np.array(store[m]["grp"])
        results[(m, lname)] = dict(dd=cv_acc(X, lab, grp, "diff_disp"), lw=cv_acc(X, lab, grp, "lw_bw"))
    print(f"  done: {lname}")

# ================================================================ tables
print("\n=== DECISION METRIC: diffusive-vs-dispersive accuracy ===")
print(f"{'method':<11}|" + "".join(f"{ln:>20}" for ln, _ in LEVELS))
for m in METHODS:
    print(f"{m:<11}|" + "".join(f"{results[(m,ln)]['dd']:>20.3f}" for ln, _ in LEVELS))
print("\n--- context only (NOT decision-driving): LW-vs-BW accuracy ---")
for m in METHODS:
    print(f"{m:<11}|" + "".join(f"{results[(m,ln)]['lw']:>20.3f}" for ln, _ in LEVELS))

with open(os.path.join(TAB, "near_shock_results.csv"), "w") as f:
    f.write("method,level,diff_disp,lw_bw\n")
    for (m, ln), d in results.items(): f.write(f"{m},{ln},{d['dd']:.4f},{d['lw']:.4f}\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = np.arange(len(LEVELS)); plt.figure(figsize=(8, 5))
for m, mk in zip(METHODS, ["s--", "^--", "o-"]):
    plt.plot(xs, [results[(m, ln)]["dd"] for ln, _ in LEVELS], mk, label=m)
plt.axhline(0.5, color="grey", ls=":"); plt.xticks(xs, [ln for ln, _ in LEVELS], rotation=20, fontsize=8)
plt.ylim(0.3, 1.03); plt.ylabel("diff-vs-disp accuracy"); plt.title("Near-shock irregular: decision metric"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "near_shock_result.png"), dpi=130)

# ================================================================ TERMINAL decision
sevname = LEVELS[-1][0]
wk, ifd, mls = (results[("weak-form", sevname)]["dd"], results[("interp+FD", sevname)]["dd"], results[("MLS", sevname)]["dd"])
print("\n" + "="*72 + "\nTERMINAL DECISION  (pre-registered; diff-vs-disp, weak-form vs interp+FD)\n" + "="*72)
print(f"severe near-shock:  weak-form={wk:.3f}   interp+FD={ifd:.3f}   (MLS={mls:.3f}, context)")
print(f"margin weak-interp = {wk-ifd:+.3f}   (WIN bar = +{WIN_MARGIN:.2f})   null-guard recon_err={recon_err:.3f}")
if recon_err <= 0.05:
    print("\n[NULL -> (A)]  the steep front still reconstructed cleanly from scattered nodes, so this test could")
    print("   not discriminate the methods. Honest default is (A): strong-form attribution paper. TERMINAL.")
elif wk - ifd >= WIN_MARGIN:
    print(f"\n[WEAK-FORM EARNS A SCOPED METHODS LEG]  weak-form leads by {wk-ifd:+.3f} >= +{WIN_MARGIN:.2f} where")
    print("   interpolation demonstrably smears the front (recon_err high). Differentiator is REAL but BOUNDED to")
    print("   the steep-gradient / irregular-observation regime - state it as exactly that, not a general claim.")
    print("   Paper gets a methods leg: 'weak-form solver attribution under steep-gradient irregular observation.'")
    print("   Tier stays specialist-family; this buys a cleaner SITE boundary, not a higher venue.")
else:
    print(f"\n[LOSS -> (A)]  weak-form did NOT clearly beat interpolate-then-FD ({wk-ifd:+.3f} < +{WIN_MARGIN:.2f}) even")
    print("   with the front demonstrably smeared by interpolation. Per pre-registration: drop weak-form as the")
    print("   headline, ZERO further rescue attempts. Write the strong-form paper:")
    print("   'A modified-equation signature method for identifying hidden discretization schemes' - attribution,")
    print("   taxonomy, reference-convergence controls, measured auditability limits. TERMINAL.")
