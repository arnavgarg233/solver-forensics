"""
solver-forensics :: BURGERS CHECKPOINT
============================================================
The grid/CFL transfer gate cleared. The REAL next gate is leaving linear
advection, because three things only show up once the PDE is nonlinear:

  (1) ANSATZ RISK. The whole coefficient edifice assumes r ≈ Σ_p c_p ∂_x^p u
      with CONSTANT c. Burgers' truncation terms are u-dependent (numerical
      diffusion ∝ local |u|), so the constant-c ansatz is only approximate.
      Does it still recover the right dominant term + SIGN well enough to
      attribute? (we report a fit-quality R^2 to quantify the degradation.)

  (2) REFERENCE CONTAMINATION. Burgers has no closed-form exact, so the
      residual is r = (coarse scheme error) - (fine reference error). If the
      reference is not fine enough, its OWN truncation leaks into the recovered
      coefficients and we are partly measuring the reference. CONTROL: refine
      the reference (N_ref sweep) and confirm the recovered coefficients are
      STABLE. If they drift with N_ref, the result is reference-dependent.

  (3) REALISM. Real solvers solve nonlinear PDEs. The audit story has to reach
      this regime at all.

BANKED CAVEAT (from the linear gate): the c3-SIGN feature that revived the
same-order taxonomy is finer and lower-energy than the c2-vs-c3 magnitude that
carries diffusive-vs-dispersive. So it is the FIRST thing expected to degrade
under nonlinearity + reference contamination. We hold "taxonomy is back" as
"back in the clean linear case - re-verify per stressor", and this run is the
first re-verification.

Pure numpy + sklearn, CPU, a few minutes on M2. No accuracy numbers hardcoded.

Schemes are ported to FLUX form for the conservation law u_t + (u^2/2)_x = mu u_xx
and SELF-VALIDATED: a correct diffusive scheme must come back c2-dominant; LW
must come back c3<0; BW c3>0. A wrong flux shows up as a wrong signature.
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
L = 1.0
N_OBS = 64                 # anti-aliased observation grid
N_C   = 256                # coarse "black-box solver" grid
MU    = 0.0                # INVISCID, integrated pre-shock: residual = advective truncation only
TFIN  = 0.08               # safely below shock time (~0.13 for these ICs) -> solution stays smooth
AMP   = 0.4                # IC fluctuation amplitude (u in [1-AMP, 1+AMP], stays > 0)
CFL_C, CFL_REF = 0.4, 0.4
N_BASE = 2048              # ICs generated here once, then decimated (band-limited -> exact)
N_REF_GOLD = 1024          # spectral gold-truth reference resolution
CONTROL_NREFS = [256, 512, 1024, 2048]   # FD2-reference refinement sweep (256 == N_C: contaminated)
N_IC, N_IC_CTRL = 120, 50
METHODS = ["FFT-shape", "strong-form", "weak-form"]

# ================================================================ physics
def random_ic_pos(N, rng, n_modes=4):
    """Smooth periodic IC, scaled to u in [1-AMP, 1+AMP] so u>0 (upwind well-defined)."""
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        kk = rng.integers(1, 4)
        u += rng.normal()*np.sin(2*np.pi*kk*x/L + rng.uniform(0, 2*np.pi))
    u /= (np.max(np.abs(u)) + 1e-9)
    return 1.0 + AMP*u

# ---- reference solvers: integrating-factor RK4 (stiff diffusion handled exactly) ----
def ifrk4(uh, Lhat, adv, dt, nsteps):
    E, E2 = np.exp(Lhat*dt), np.exp(Lhat*dt/2)
    for _ in range(nsteps):
        a = dt*adv(uh)
        b = dt*adv(E2*(uh + a/2))
        c = dt*adv(E2*uh + b/2)
        d = dt*adv(E*uh + E2*c)
        uh = E*uh + (E*a + 2*E2*(b + c) + d)/6
    return uh

def solve_ref(u0, N, kind):
    """kind='spectral' -> near-exact truth; kind='fd2' -> 2nd-order central advection (carries O(dx^2))."""
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N); ik = 1j*k; dx = L/N
    umax = np.max(np.abs(u0)) + 1e-9
    nsteps = int(np.ceil(TFIN/(CFL_REF*dx/umax))); dt = TFIN/nsteps
    if kind == "spectral":
        Lhat = -MU*k*k
        mask = np.abs(k) <= (2/3)*np.max(np.abs(k))               # 2/3 dealias
        def adv(uh):
            u = np.fft.ifft(uh).real
            return -0.5*ik*(np.fft.fft(u*u)*mask)
    else:  # fd2
        Lhat = -MU*(2 - 2*np.cos(k*dx))/dx**2                     # FD-Laplacian symbol
        def adv(uh):
            u = np.fft.ifft(uh).real; f = 0.5*u*u
            return np.fft.fft(-(np.roll(f, -1) - np.roll(f, 1))/(2*dx))
    uh = ifrk4(np.fft.fft(u0), Lhat, adv, dt, nsteps)
    return np.fft.ifft(uh).real

# ---- coarse one-step flux-form schemes for u_t + (u^2/2)_x = mu u_xx, u>0 ----
def _diff(u, dx): return (np.roll(u, -1) - 2*u + np.roll(u, 1))/dx**2
def s_upwind(u, dt, dx):
    f = 0.5*u*u                                                   # upwind (u>0): F_{i+1/2}=f_i
    return u - (dt/dx)*(f - np.roll(f, 1))
def s_lax_friedrichs(u, dt, dx):
    f = 0.5*u*u
    Fp = 0.5*(f + np.roll(f, -1)) - 0.5*(dx/dt)*(np.roll(u, -1) - u)
    return u - (dt/dx)*(Fp - np.roll(Fp, 1))
def s_lax_wendroff(u, dt, dx):                                    # Richtmyer two-step
    f = 0.5*u*u
    uhalf = 0.5*(u + np.roll(u, -1)) - 0.5*(dt/dx)*(np.roll(f, -1) - f)
    Fp = 0.5*uhalf*uhalf
    return u - (dt/dx)*(Fp - np.roll(Fp, 1))
def s_beam_warming(u, dt, dx):                                    # 2nd-order upwind (u>0)
    f = 0.5*u*u
    Fp = f + 0.5*(1 - u*dt/dx)*(f - np.roll(f, 1))
    return u - (dt/dx)*(Fp - np.roll(Fp, 1))

SCHEMES = {"upwind":s_upwind, "lax_friedrichs":s_lax_friedrichs,
           "lax_wendroff":s_lax_wendroff, "beam_warming":s_beam_warming}
names = list(SCHEMES)
DIFFUSIVE = {"upwind","lax_friedrichs"}
LW, BW = names.index("lax_wendroff"), names.index("beam_warming")

def run_coarse(scheme, u0, N):
    dx = L/N
    dt_adv = CFL_C*dx/(np.max(np.abs(u0)) + 1e-9)
    dt = dt_adv if MU == 0 else min(dt_adv, 0.4*dx*dx/MU)
    nsteps = int(np.ceil(TFIN/dt)); dt = TFIN/nsteps
    u = u0.copy()
    for _ in range(nsteps): u = scheme(u, dt, dx)
    return u

# ================================================================ observation + features
def antialias_to(R, N_obs):
    N = R.shape[-1]
    if N == N_obs: return R
    F = np.fft.rfft(R, axis=-1); F[..., N_obs//2 + 1:] = 0.0
    return np.fft.irfft(F, n=N, axis=-1)[..., ::(N // N_obs)]

def bump_polys(q):
    p0 = Polynomial([1.0, 0.0, -1.0])**q
    return [p0.deriv(k) for k in range(5)]
def build_weakform(q=6, n_centers=16, widths=(0.08, 0.12, 0.16)):
    xobs = np.linspace(0, L, N_OBS, endpoint=False); P = bump_polys(q)
    PHI, D2, D3, D4 = [], [], [], []
    for w in widths:
        for cx in np.linspace(0, L, n_centers, endpoint=False):
            s = (((xobs - cx + L/2) % L) - L/2)/w
            m = (np.abs(s) <= 1.0).astype(float); se = np.where(m > 0, s, 0.0)
            PHI.append(P[0](se)*m); D2.append(P[2](se)/w**2*m)
            D3.append(P[3](se)/w**3*m); D4.append(P[4](se)/w**4*m)
    return tuple(np.array(z) for z in (PHI, D2, D3, D4))

def weakform_coeffs(U, R, WF, return_fit=False):
    PHI, D2, D3, D4 = WF; h = L/N_OBS
    B  =  (R @ PHI.T)*h
    A  = np.stack([(U @ D2.T)*h, -(U @ D3.T)*h, (U @ D4.T)*h], axis=2)
    AtA = np.einsum('mji,mjk->mik', A, A) + 1e-8*np.eye(3)
    c = np.linalg.solve(AtA, np.einsum('mji,mj->mi', A, B)[..., None])[..., 0]
    if not return_fit: return c
    pred = np.einsum('mji,mi->mj', A, c)                          # weak-space goodness-of-fit
    r2 = 1 - np.sum((B - pred)**2, 1)/(np.sum(B**2, 1) + 1e-12)
    return c, r2
def strongform_coeffs(U, R):
    h = L/N_OBS
    uxx   = (np.roll(U,-1,1) - 2*U + np.roll(U,1,1))/h**2
    uxxx  = (np.roll(U,-2,1) - 2*np.roll(U,-1,1) + 2*np.roll(U,1,1) - np.roll(U,2,1))/(2*h**3)
    uxxxx = (np.roll(U,-2,1) - 4*np.roll(U,-1,1) + 6*U - 4*np.roll(U,1,1) + np.roll(U,2,1))/h**4
    A = np.stack([uxx, uxxx, uxxxx], axis=2)
    AtA = np.einsum('mni,mnk->mik', A, A) + 1e-8*np.eye(3)
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', A, R)[..., None])[..., 0]
def coeff_features(c, n_ratio):
    unit = c/(np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        r32 = np.clip(np.nan_to_num(c[:,1]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
        r42 = np.clip(np.nan_to_num(c[:,2]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
    return np.nan_to_num(np.hstack([unit, r32[:,None]] + ([r42[:,None]] if n_ratio == 2 else [])))
def fft_features(R):
    Rn = R/(np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return np.nan_to_num(np.abs(np.fft.rfft(Rn, axis=1)))
def features_all(U, R, WF):
    return {"FFT-shape": fft_features(R),
            "strong-form": coeff_features(strongform_coeffs(U, R), 2),
            "weak-form": coeff_features(weakform_coeffs(U, R, WF), 1)}

def add_field_noise(U, sigma, seed):
    if sigma == 0: return U
    g = np.random.default_rng(seed)
    return U + sigma*np.sqrt(np.mean(U**2, axis=1, keepdims=True))*g.standard_normal(U.shape)

# ================================================================ data generation
def build_dataset(n_ic, seed, ref_kind, N_ref):
    """Paired residuals r_obs = coarse_obs - ref_obs (both at TFIN, anti-aliased to N_OBS)."""
    rng = np.random.default_rng(seed)
    Ucoarse, Resid, lab, grp = [], [], [], []
    for ic in range(n_ic):
        u_base = random_ic_pos(N_BASE, rng)              # same continuous IC for both solvers
        u0_c, u0_ref = u_base[::(N_BASE//N_C)], u_base[::(N_BASE//N_ref)]
        ref_obs = antialias_to(solve_ref(u0_ref, N_ref, ref_kind), N_OBS)
        for li, name in enumerate(names):
            cu = antialias_to(run_coarse(SCHEMES[name], u0_c, N_C), N_OBS)
            Ucoarse.append(cu); Resid.append(cu - ref_obs); lab.append(li); grp.append(seed*10**6 + ic)
    return (np.array(Ucoarse), np.array(Resid), np.array(lab), np.array(grp))

def _yd(lab, dist):
    if dist == "diff_disp":
        return np.array([0 if names[l] in DIFFUSIVE else 1 for l in lab]), np.ones(len(lab), bool)
    return lab, np.isin(lab, [LW, BW])
def cv_acc(X, lab, grp, dist):
    y, sel = _yd(lab, dist)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X[sel], y[sel], groups=grp[sel], cv=GroupKFold(5)).mean()

# ================================================================ RUN
WF = build_weakform()
print(f"Burgers checkpoint: inviscid (mu={MU}, integrated pre-shock), smooth u in [{1-AMP:.1f},{1+AMP:.1f}], "
      f"coarse N={N_C}, observe at N_OBS={N_OBS}")
print(f"gold reference: spectral IF-RK4 at N_ref={N_REF_GOLD}\n")

print(f"generating {N_IC} paired ICs vs spectral-gold reference ...")
Uc, R, lab, grp = build_dataset(N_IC, seed=1, ref_kind="spectral", N_ref=N_REF_GOLD)

# (1) physical signature check + ansatz fit quality ------------------------------------
c_clean, r2 = weakform_coeffs(Uc, R, WF, return_fit=True)
cu = c_clean/(np.linalg.norm(c_clean, axis=1, keepdims=True) + 1e-12)
print("physical check - mean weak-form unit c=[c2,c3,c4] on Burgers (spectral ref, clean):")
for li, name in enumerate(names):
    sel = lab == li; m = cu[sel].mean(0)
    print(f"   {name:<15} c2={m[0]:+.2f} c3={m[1]:+.2f} c4={m[2]:+.2f}   "
          f"c3>0 in {(c_clean[sel,1]>0).mean()*100:4.0f}%   ansatz R^2={r2[sel].mean():.2f}")
print("   theory: diffusive -> |c2| dominant; LW -> c3<0; BW -> c3>0   "
      f"(linear gate had R^2~0.99; watch the drop)\n")

# (2) attribution vs the linear baseline, clean and field-noisy -------------------------
print("attribution (GroupKFold-by-IC, spectral-gold reference):")
print(f"{'noise':<14}|" + "|".join(f"{m:^16}" for m in METHODS))
print(f"{'':<14}|" + "|".join(f"{'d/d':>7} {'lw/bw':>7} " for _ in METHODS))
for sigma in (0.0, 0.01):
    F = features_all(add_field_noise(Uc, sigma, 7), R, WF)
    cells = "|".join(f"{cv_acc(F[m],lab,grp,'diff_disp'):>7.3f} {cv_acc(F[m],lab,grp,'lw_bw'):>7.3f} "
                     for m in METHODS)
    print(f"field σ={sigma:<6}|{cells}")

# (3) REFERENCE-REFINEMENT STABILITY CONTROL -------------------------------------------
print("\n" + "="*70)
print("REFERENCE-REFINEMENT CONTROL (FD2 reference, refine N_ref; coeffs must converge)")
print("  N_ref=256 == coarse grid -> reference no better than solver (contaminated)")
print("="*70)
ctrl_cu, ctrl_acc = {}, {}
for N_ref in CONTROL_NREFS:                                       # pass 1: build every reference level
    Uc2, R2c, lab2, grp2 = build_dataset(N_IC_CTRL, seed=2, ref_kind="fd2", N_ref=N_ref)
    c2c, r2c = weakform_coeffs(Uc2, R2c, WF, return_fit=True)
    ctrl_cu[N_ref] = (c2c/(np.linalg.norm(c2c, axis=1, keepdims=True) + 1e-12), lab2)
    F = features_all(Uc2, R2c, WF)
    ctrl_acc[N_ref] = (cv_acc(F["weak-form"], lab2, grp2, "diff_disp"),
                       cv_acc(F["weak-form"], lab2, grp2, "lw_bw"), r2c.mean())
cuf, labf = ctrl_cu[CONTROL_NREFS[-1]]                            # pass 2: drift vs the finest reference
print(f"{'N_ref':>7} | {'d/d weak':>9} {'lw/bw weak':>11} | {'c-drift vs 2048':>16} | {'mean R^2':>9}")
for N_ref in CONTROL_NREFS:
    cu2, lab2 = ctrl_cu[N_ref]; dd, lw, r2m = ctrl_acc[N_ref]
    drift = max(np.linalg.norm(cu2[lab2 == li].mean(0) - cuf[labf == li].mean(0)) for li in range(len(names)))
    print(f"{N_ref:>7} | {dd:>9.3f} {lw:>11.3f} | {drift:>16.3f} | {r2m:>9.2f}")

# ================================================================ CSV + plot
csv = os.path.join(TAB, "burgers_nonlinear_results.csv")
with open(csv, "w") as f:
    f.write("scheme,c2,c3,c4,c3pos_frac,ansatz_r2\n")
    for li, name in enumerate(names):
        sel = lab == li; m = cu[sel].mean(0)
        f.write(f"{name},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{(c_clean[sel,1]>0).mean():.3f},{r2[sel].mean():.3f}\n")
print(f"\nsignatures -> {csv}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
col = {"upwind":"C0","lax_friedrichs":"C1","lax_wendroff":"C3","beam_warming":"C2"}
for li, name in enumerate(names):
    sel = lab == li
    ax[0].scatter(c_clean[sel,0], c_clean[sel,1], s=10, alpha=0.5, color=col[name], label=name)
ax[0].axhline(0, color="grey", lw=.8); ax[0].axvline(0, color="grey", lw=.8)
ax[0].set_xlabel("recovered c2 (diffusion)"); ax[0].set_ylabel("recovered c3 (dispersion)")
ax[0].set_title("Burgers: recovered coefficient signatures\n(c3 SIGN splits LW/BW)"); ax[0].legend(fontsize=8)
for N_ref in CONTROL_NREFS:
    cuc, labc = ctrl_cu[N_ref]
    ax[1].scatter([N_ref]*2, [cuc[labc == LW,1].mean(), cuc[labc == BW,1].mean()],
                  color=["C3","C2"], s=40)
ax[1].axhline(0, color="grey", lw=.8); ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("reference resolution N_ref"); ax[1].set_ylabel("mean unit-c3 (LW red, BW green)")
ax[1].set_title("Reference-refinement control:\nLW/BW c3-sign vs N_ref (must stabilize)")
plt.tight_layout(); plot = os.path.join(FIG, "burgers_nonlinear_result.png"); plt.savefig(plot, dpi=130)
print(f"plot       -> {plot}")

# ================================================================ decision
print("\n" + "="*70 + "\nDECISION  (Burgers checkpoint)\n" + "="*70)
F0 = features_all(add_field_noise(Uc, 0.01, 7), R, WF)
dd_wk = cv_acc(F0["weak-form"], lab, grp, "diff_disp")
lw_wk = cv_acc(F0["weak-form"], lab, grp, "lw_bw")
lw_sf = cv_acc(F0["strong-form"], lab, grp, "lw_bw")
r2_mean = r2.mean()
# reference stability: drift between the two finest references
cuf, labf = ctrl_cu[CONTROL_NREFS[-1]]; cup, labp = ctrl_cu[CONTROL_NREFS[-2]]
ref_drift = max(np.linalg.norm(cup[labp == li].mean(0) - cuf[labf == li].mean(0)) for li in range(len(names)))
print(f"ansatz fit R^2 (Burgers) = {r2_mean:.2f}   (linear ~0.99; a drop quantifies nonlinear misfit)")
print(f"diffusive-vs-dispersive, weak-form, field 1% = {dd_wk:.3f}")
print(f"same-order LW/BW, field 1%: weak={lw_wk:.3f}  strong={lw_sf:.3f}  (c3-sign - expected to degrade first)")
print(f"reference stability: c-drift between N_ref={CONTROL_NREFS[-2]} and {CONTROL_NREFS[-1]} = {ref_drift:.3f}")
stable = ref_drift < 0.1
if dd_wk >= 0.90 and max(lw_wk, lw_sf) >= 0.75 and stable:
    print("\n[GO-NONLINEAR]  diffusive-vs-dispersive AND the c3-sign taxonomy survive nonlinearity,")
    print("   and coefficients are reference-stable -> serious project. Next: irregular mesh (the weak-")
    print("   form differentiator vs SITE), then KdV, then near-shock stress.")
elif dd_wk >= 0.90 and stable:
    print(f"\n[PARTIAL]  diffusive-vs-dispersive survives ({dd_wk:.3f}) and is reference-stable, but the")
    print(f"   c3-sign taxonomy degraded (best {max(lw_wk,lw_sf):.3f}) - exactly the banked caveat. Scope")
    print("   fine same-order taxonomy to the clean/linear case; keep diff-vs-disp as the nonlinear claim.")
elif not stable:
    print(f"\n[REFERENCE-LIMITED]  coefficients still move with N_ref (drift {ref_drift:.3f}) - partly")
    print("   measuring the reference. Refine further / use the spectral reference before trusting numbers.")
else:
    print(f"\n[NO]  even diffusive-vs-dispersive weakened under nonlinearity ({dd_wk:.3f}). The constant-c")
    print("   ansatz does not reach Burgers as-is. Rethink the feature before any nonlinear claim.")
