"""
solver-forensics :: WEAK-FORM MICRO-GATE
=============================================================
Fused micro-gate that collapses three doc steps into one decision. Stays on
1D advection ONLY (no Burgers/KdV expansion yet). It decides the
magnitude-invariant ceiling and whether the signal is grid/CFL-robust BEFORE
any Phase-1 spend.

THE PROBLEM IT TARGETS
  In fft_shape_probe.py the L2-normalized |FFT| shape signal for diffusive-vs-dispersive
  was 0.95 at noise 0 but fell off a CLIFF to ~0.68 at 1% noise (and flatlined).
  The magnitude-invariant fingerprint lives below the noise floor of a probe
  that reads a raw magnitude spectrum.

THE FIX UNDER TEST
  A WEAK-FORM modified-equation feature. It NEVER differentiates the (noisy)
  field; it only INTEGRATES the field and the residual against smooth,
  compactly-supported test functions phi_j, using the identity
        ∫ (∂_x^p u) phi_j dx = (-1)^p ∫ u (∂_x^p phi_j) dx,
  and solves  ∫ r phi_j = Σ_p c_p (-1)^p ∫ u ∂_x^p phi_j  for c=[c2,c3,c4].
  Integration averages noise down; differentiation amplifies it. The
  magnitude-invariant feature is the DIRECTION of c (unit-normalized) + ratios.

THREE METHODS, same task, same GroupKFold-by-IC split:
  (1) FFT-shape   : logistic on L2-normalized |FFT| magnitude   [shape baseline]
  (2) strong-form : FD-differentiate the field, LSQ r = Σ c_p ∂^p u  [naive filter]
  (3) weak-form   : integrate against phi_j, solve for c           [UNDER TEST]

DECISION is read off FIELD-RELATIVE noise (eta ~ sigma*RMS(u_obs) added to the
field BEFORE forming the residual) on the GRID/CFL-TRANSFER row.

Pure numpy + sklearn, CPU, a few minutes on M2. No accuracy numbers hardcoded.

SOLVER REUSE: the four scheme updates, the exact periodic-advection solution,
and the IC generator are the IDENTICAL numerics from fft_shape_probe.py - parametrized
by (N, nu) ONLY because the CFL/grid-transfer experiment must vary them, which
the original module-global versions cannot. The math is not rewritten.
"""
import os
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings; warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
L, a, T = 1.0, 1.0, 0.30
N_OBS = 64                      # fixed anti-aliased observation grid (physical coarseness, all N)
GRIDS, CFLS = [128, 256, 512], [0.4, 0.6, 0.8]
BASE = (256, 0.6)               # training grid/CFL
N_IC_BASE, N_IC_TEST = 200, 60
METHODS = ["FFT-shape", "strong-form", "weak-form"]

# ---------------------------------------------------------------- physics
# (identical numerics to fft_shape_probe.py, parametrized by N, nu)
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
DIFFUSIVE  = {"upwind","lax_friedrichs"}
DISPERSIVE = {"lax_wendroff","beam_warming"}
LW, BW = names.index("lax_wendroff"), names.index("beam_warming")

def random_ic(N, rng, n_modes=6):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        kk = rng.integers(1,8)
        u += rng.normal()*np.sin(2*np.pi*kk*x/L + rng.uniform(0,2*np.pi))
    if rng.random() < 0.7:
        x0, w = rng.uniform(0,L), rng.uniform(L*0.02, L*0.08)
        u += rng.normal()*np.exp(-(((x-x0+L/2)%L - L/2)**2)/(2*w*w))
    return u

# ---------------------------------------------------------------- observation
def antialias_to(R, N_obs):
    """Ideal low-pass to the N_obs grid's Nyquist, then sample to N_obs points (periodic)."""
    N = R.shape[-1]
    if N == N_obs: return R
    F = np.fft.rfft(R, axis=-1)
    F[..., N_obs//2 + 1:] = 0.0
    Rlp = np.fft.irfft(F, n=N, axis=-1)
    return Rlp[..., ::(N // N_obs)]

# ---------------------------------------------------------------- simulate + noise
def simulate(N, nu, n_ic, seed):
    """Paired residuals: every IC run through ALL four schemes; observed at N_OBS."""
    rng = np.random.default_rng(seed)
    dx = L/N; dt = nu*dx/a; nsteps = int(round(T/dt))
    Unum, Uex, lab, grp = [], [], [], []
    for ic in range(n_ic):
        u0 = random_ic(N, rng)
        ex_obs = antialias_to(exact_advection(u0, nsteps*dt, N), N_OBS)
        for li, name in enumerate(names):
            u = u0.copy()
            for _ in range(nsteps): u = SCHEMES[name](u, nu)
            Unum.append(antialias_to(u, N_OBS)); Uex.append(ex_obs)
            lab.append(li); grp.append(seed*1_000_000 + ic)
    return (np.array(Unum), np.array(Uex), np.array(lab), np.array(grp))

def apply_noise(Unum, Uex, kind, sigma, seed):
    """field: noise on the black-box field BEFORE the residual; resid: noise on the residual only."""
    nrng = np.random.default_rng(seed)
    if kind == "field":
        rms = np.sqrt(np.mean(Unum**2, axis=1, keepdims=True))
        ufield = Unum + sigma*rms*nrng.standard_normal(Unum.shape) if sigma > 0 else Unum
        return ufield, ufield - Uex                     # uobs (noisy field), robs
    else:  # resid-relative comparator: field stays clean, noise only corrupts the residual
        clean = Unum - Uex
        rms = np.sqrt(np.mean(clean**2, axis=1, keepdims=True))
        robs = clean + sigma*rms*nrng.standard_normal(clean.shape) if sigma > 0 else clean
        return Unum, robs

# ---------------------------------------------------------------- weak-form machinery
def bump_polys(q):
    """(1-s^2)^q as an exact polynomial in s, plus its analytic derivatives p0..p4."""
    p0 = Polynomial([1.0, 0.0, -1.0])**q
    return [p0.deriv(k) for k in range(5)]

def build_weakform(q=6, n_centers=16, widths=(0.08, 0.12, 0.16)):
    """Smooth compactly-supported phi_j and their analytic ∂^p, sampled on the N_OBS grid."""
    xobs = np.linspace(0, L, N_OBS, endpoint=False)
    P = bump_polys(q)
    PHI, D2, D3, D4 = [], [], [], []
    for w in widths:
        for cx in np.linspace(0, L, n_centers, endpoint=False):
            d = ((xobs - cx + L/2) % L) - L/2          # periodic minimal-image distance
            s = d / w
            m = (np.abs(s) <= 1.0).astype(float)
            se = np.where(m > 0, s, 0.0)               # avoid large-s poly eval; masked out anyway
            PHI.append(P[0](se)*m)
            D2.append(P[2](se)/w**2 * m)
            D3.append(P[3](se)/w**3 * m)
            D4.append(P[4](se)/w**4 * m)
    return tuple(np.array(z) for z in (PHI, D2, D3, D4))

def weakform_coeffs(U, R, WF):
    """Solve ∫ r phi = Σ_p c_p (-1)^p ∫ u ∂^p phi  for c=[c2,c3,c4]; NEVER differentiates U or R."""
    PHI, D2, D3, D4 = WF; h = L/N_OBS
    B  =  (R @ PHI.T) * h                              # (M,J)  ∫ r phi
    A2 =  (U @ D2.T)  * h                              # (-1)^2 ∫ u ∂2 phi
    A3 = -(U @ D3.T)  * h                              # (-1)^3 ∫ u ∂3 phi
    A4 =  (U @ D4.T)  * h                              # (-1)^4 ∫ u ∂4 phi
    A  = np.stack([A2, A3, A4], axis=2)                # (M,J,3)
    AtA = np.einsum('mji,mjk->mik', A, A) + 1e-8*np.eye(3)
    Atb = np.einsum('mji,mj->mi', A, B)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]                   # (M,3)

def strongform_coeffs(U, R):
    """Naive matched filter: finite-difference the (noisy) field, LSQ r = Σ c_p ∂^p u."""
    h = L/N_OBS
    uxx   = (np.roll(U,-1,1) - 2*U + np.roll(U,1,1)) / h**2
    uxxx  = (np.roll(U,-2,1) - 2*np.roll(U,-1,1) + 2*np.roll(U,1,1) - np.roll(U,2,1)) / (2*h**3)
    uxxxx = (np.roll(U,-2,1) - 4*np.roll(U,-1,1) + 6*U - 4*np.roll(U,1,1) + np.roll(U,2,1)) / h**4
    A = np.stack([uxx, uxxx, uxxxx], axis=2)           # (M,N_OBS,3)
    AtA = np.einsum('mni,mnk->mik', A, A) + 1e-8*np.eye(3)
    Atb = np.einsum('mni,mn->mi', A, R)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]

def coeff_features(c, n_ratio):
    """Magnitude-invariant: unit DIRECTION of c plus bounded ratio(s). Not exact-coeff recovery."""
    unit = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        r32 = np.clip(np.nan_to_num(c[:,1]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
        r42 = np.clip(np.nan_to_num(c[:,2]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
    cols = [unit, r32[:,None]] + ([r42[:,None]] if n_ratio == 2 else [])
    return np.hstack(cols)

def fft_features(R):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return np.abs(np.fft.rfft(Rn, axis=1))

def features_all(uobs, robs, WF):
    return {"FFT-shape":   fft_features(robs),
            "strong-form": coeff_features(strongform_coeffs(uobs, robs), 2),
            "weak-form":   coeff_features(weakform_coeffs(uobs, robs, WF), 1)}

# ---------------------------------------------------------------- evaluation
def _yd(lab, dist):
    if dist == "diff_disp":
        return np.array([0 if names[l] in DIFFUSIVE else 1 for l in lab]), np.ones(len(lab), bool)
    sel = np.isin(lab, [LW, BW])
    return lab, sel                                     # binary labels {LW,BW} on the subset

def cv_same(X, lab, grp, dist):
    y, sel = _yd(lab, dist)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X[sel], y[sel], groups=grp[sel], cv=GroupKFold(5)).mean()

def transfer(Xtr, ltr, Xte, lte, dist):
    ytr, str_ = _yd(ltr, dist); yte, ste = _yd(lte, dist)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr[str_], ytr[str_])
    return clf.score(Xte[ste], yte[ste])

# ================================================================ RUN
print(f"observation: anti-aliased to N_OBS={N_OBS} points (fixed physical coarseness across all grids)")
WF = build_weakform()
print(f"weak-form: {WF[0].shape[0]} smooth test functions (centers x widths), q=6, never differentiates the field\n")

print("simulating base grid/CFL", BASE, "...")
Unum_b, Uex_b, lab_b, grp_b = simulate(*BASE, N_IC_BASE, seed=1)

rows = []   # (label, {(method,dist): acc})
SAME = [("field",0.0,"same grid/CFL, field σ=0"),
        ("field",0.01,"same grid/CFL, field σ=0.01"),
        ("field",0.05,"same grid/CFL, field σ=0.05")]
for kind, sig, label in SAME:
    uo, ro = apply_noise(Unum_b, Uex_b, kind, sig, seed=10)
    F = features_all(uo, ro, WF)
    rows.append((label, {(m, d): cv_same(F[m], lab_b, grp_b, d)
                         for m in METHODS for d in ("diff_disp","lw_bw")}))
    print(f"  done: {label}")

# ---- transfer row: train on base (field σ=0.01), test on the 8 held-out grid/CFL combos ----
print("simulating held-out grid/CFL combos for transfer ...")
uo_tr, ro_tr = apply_noise(Unum_b, Uex_b, "field", 0.01, seed=10)
Ftr = features_all(uo_tr, ro_tr, WF)
test_combos, Ute, Rte, Lte = [], [], [], []
sidx = 100
for N in GRIDS:
    for nu in CFLS:
        if (N, nu) == BASE: continue
        sidx += 1
        Un, Ux, la, _ = simulate(N, nu, N_IC_TEST, seed=sidx)
        uo, ro = apply_noise(Un, Ux, "field", 0.01, seed=sidx+500)
        test_combos.append((N, nu, uo, ro, la))
        Ute.append(uo); Rte.append(ro); Lte.append(la)
        print(f"  done: N={N}, CFL={nu}")
Ute, Rte, Lte = np.vstack(Ute), np.vstack(Rte), np.concatenate(Lte)
Fte = features_all(Ute, Rte, WF)
trans_res = {(m, d): transfer(Ftr[m], lab_b, Fte[m], Lte, d)
             for m in METHODS for d in ("diff_disp","lw_bw")}
rows.append(("TRANSFER grid/CFL, field σ=0.01  [DECISION]", trans_res))

# ---- residual-relative comparator (clean-limit, same grid) ----
uo, ro = apply_noise(Unum_b, Uex_b, "resid", 0.01, seed=10)
F = features_all(uo, ro, WF)
rows.append(("resid-rel σ=0.01 (clean-limit comparator)",
             {(m, d): cv_same(F[m], lab_b, grp_b, d) for m in METHODS for d in ("diff_disp","lw_bw")}))

# reorder to the requested layout: σ0, field σ0.01, field σ0.05, TRANSFER, resid-comparator
order = [0, 1, 2, 3, 4]
rows = [rows[i] for i in order]

# ---------------------------------------------------------------- table
print("\n" + "="*96)
print("d/d = diffusive-vs-dispersive (load-bearing) | lw/bw = same-order (expected near chance, magnitude-removed)")
print("="*96)
hdr = f"{'row':<44}|" + "|".join(f"{m:^16}" for m in METHODS)
sub = f"{'':<44}|" + "|".join(f"{'d/d':>7} {'lw/bw':>7} " for _ in METHODS)
print(hdr); print(sub); print("-"*96)
for label, res in rows:
    cells = "|".join(f"{res[(m,'diff_disp')]:>7.3f} {res[(m,'lw_bw')]:>7.3f} " for m in METHODS)
    print(f"{label:<44}|{cells}")

# per-combo transfer for the weak-form load-bearing claim (grid/CFL robustness diagnostic)
print("\nper-combo TRANSFER, weak-form diffusive-vs-dispersive (train=256/CFL0.6, field σ=0.01):")
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
ytr, str_ = _yd(lab_b, "diff_disp"); clf.fit(Ftr["weak-form"][str_], ytr[str_])
for N, nu, uo, ro, la in test_combos:
    Xc = features_all(uo, ro, WF)["weak-form"]; yc, sc = _yd(la, "diff_disp")
    print(f"   N={N:<4} CFL={nu}: {clf.score(Xc[sc], yc[sc]):.3f}")

# ---- physical sanity check: do recovered coefficients match modified-equation theory? ----
print("\nphysical check - mean weak-form unit coefficients on CLEAN base data c=[c2,c3,c4]:")
c_clean = weakform_coeffs(Unum_b, Unum_b - Uex_b, WF)
cu = c_clean / (np.linalg.norm(c_clean, axis=1, keepdims=True) + 1e-12)
for li, name in enumerate(names):
    sel = lab_b == li
    m = cu[sel].mean(0)
    print(f"   {name:<15} c2={m[0]:+.2f} c3={m[1]:+.2f} c4={m[2]:+.2f}   "
          f"c3>0 in {(c_clean[sel,1] > 0).mean()*100:4.0f}% of samples")
print("   theory: diffusive (upwind/LF) -> |c2| dominant; LW -> c3<0 (lags behind); BW -> c3>0 (leads ahead)")

# ---------------------------------------------------------------- CSV + plot
csv = os.path.join(TAB, "coefficient_attribution_results.csv")
with open(csv, "w") as f:
    f.write("row,method,distinction,accuracy\n")
    for label, res in rows:
        for m in METHODS:
            for d in ("diff_disp","lw_bw"):
                f.write(f"\"{label}\",{m},{d},{res[(m,d)]:.4f}\n")
print(f"\nfull results -> {csv}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = [r[0].replace("  [DECISION]","").replace(" (clean-limit comparator)","") for r in rows]
x = np.arange(len(rows)); wd = 0.26
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
for j, dist in enumerate(("diff_disp","lw_bw")):
    for i, m in enumerate(METHODS):
        ax[j].bar(x + (i-1)*wd, [r[1][(m,dist)] for r in rows], wd, label=m)
    ax[j].axhline(0.5, color="grey", ls=":")
    ax[j].axhline(0.9, color="green", ls="--", lw=1, alpha=0.6)
    ax[j].set_xticks(x); ax[j].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[j].set_ylim(0.3, 1.02); ax[j].set_ylabel("accuracy"); ax[j].legend(fontsize=8)
    ax[j].set_title(("diffusive vs dispersive (load-bearing)" if j==0
                     else "LW vs BW (same-order limit)"))
plt.tight_layout()
plot = os.path.join(FIG, "coefficient_attribution_result.png"); plt.savefig(plot, dpi=130)
print(f"plot         -> {plot}")

# ---------------------------------------------------------------- decision
tr = rows[3][1]
wf, sf, ft = (tr[("weak-form","diff_disp")], tr[("strong-form","diff_disp")], tr[("FFT-shape","diff_disp")])
collapse = max(rows[2][1][(m,"diff_disp")] for m in METHODS) < 0.70
print("\n" + "="*96)
print("DECISION  (read off the TRANSFER row, field-relative 1% noise)")
print("="*96)
print(f"diffusive-vs-dispersive on transfer:  weak-form={wf:.3f}  strong-form={sf:.3f}  FFT-shape={ft:.3f}")
if wf >= 0.90:
    print(f"[GO]  weak-form holds {wf:.3f} >= 0.90 across grid/CFL under field 1% noise.")
    print("      Signal is STRUCTURAL and grid-robust -> proceed to Phase 1 (add Burgers, then KdV).")
elif collapse and (wf - ft) < 0.05:
    print(f"[SCOPE DOWN]  all three methods collapse together under field noise "
          f"(σ=0.05 same-grid best = {max(rows[2][1][(m,'diff_disp')] for m in METHODS):.3f}); "
          f"weak-form gives no edge ({wf:.3f} vs FFT {ft:.3f}).")
    print("      Shape signal is below the field-noise floor -> scope to known-resolution AUDIT")
    print("      (lean on magnitude with the dx confound stated loudly) or an honest-limits paper.")
else:
    print(f"[PARTIAL]  weak-form ({wf:.3f}) beats FFT-shape ({ft:.3f}) by {wf-ft:+.3f} and "
          f"strong-form ({sf:.3f}) by {wf-sf:+.3f}, but is below the 0.90 GO bar.")
    print("      Integration recovers SOME magnitude-invariant signal the FFT probe lost, yet not")
    print("      enough for a clean grid-robust claim. Judgement call: tighten weak-form (more/larger")
    print("      test functions, denoising) and re-gate, or scope to known-resolution audit.")
lw_max = max(rows[3][1][(m,"lw_bw")] for m in METHODS)
print(f"[EXPECTED]  same-order LW-vs-BW on transfer, best method = {lw_max:.3f} "
      f"({'near chance - measured same-order limit, do not chase' if lw_max < 0.65 else 'above chance - note as a bonus, verify'})")
