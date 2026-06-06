"""
Multi-resolution audit: break the grid-resolution confound by convergence order.
=================================================================================
The single-snapshot audit cannot separate a scheme change from a grid change at
unknown resolution (the structural grid confound). This experiment tests whether
observing the SAME solver at several grid resolutions, and using the convergence
ORDER p (slope of log relative-residual vs log N), removes the confound. Order is a
property of the scheme (1st vs 2nd order), not the grid, so it is grid-invariant by
construction.

Substrate: linear advection u_t + a u_x = 0, analytic exact, four schemes
{upwind, lax_friedrichs (1st order), lax_wendroff, beam_warming (2nd order)}.

Pre-registered decision (degraded = 1% field noise):
  GO    : order-feature diffusive-vs-dispersive AUROC >= 0.85, grid control NC2 <= 0.65,
          margin (scheme - NC2) >= 0.20, AND the single-snapshot baseline shows the
          confound (NC2 high) so the improvement is genuine.
  KILL  : multi-resolution NC2 still fires (> 0.65) -> the confound is irreducible even
          with multi-resolution access (a stronger measured limit); or order is too
          noise-fragile (attribution < 0.85).
  MIXED : otherwise.
Assumes multi-resolution query access; order estimation is noise-fragile (tested at 0 and 1%).
Pure numpy + sklearn, numpy-2-safe; bug-fixed Fourier resample throughout.
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

L, a, NU, T = 1.0, 1.0, 0.6, 0.30
N_BASE, N_IC, COMMON = 192, 60, 32
R1 = (48, 64, 80, 96)          # resolution set A
R2 = (56, 72, 88, 104)         # resolution set B (disjoint grids, same scheme -> NC2 must stay at chance)

def upwind(u, nu):         return u - nu*(u - np.roll(u,1))
def lax_friedrichs(u, nu): return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))
def lax_wendroff(u, nu):   return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))
def beam_warming(u, nu):   return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))
SCHEMES = {"upwind":upwind, "lax_friedrichs":lax_friedrichs, "lax_wendroff":lax_wendroff, "beam_warming":beam_warming}
names = list(SCHEMES); DIFFUSIVE = {"upwind", "lax_friedrichs"}; UP = "upwind"

def exact(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
    return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*a*t), n=N)
def random_ic(N, rng, n_modes=4):                    # smooth (modes 1-4) -> exactly resolved at every pool grid
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes): u += rng.normal()*np.sin(2*np.pi*rng.integers(1, 5)*x/L + rng.uniform(0, 2*np.pi))
    return u/(np.std(u) + 1e-9)
def antialias(u, M):                                 # proper Fourier resample (bug-fixed)
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M) * (M/N)
def run(scheme, N, u0_N):
    dx = L/N; dt = NU*dx/a; ns = int(round(T/dt))
    u = u0_N.copy()
    for _ in range(ns): u = SCHEMES[scheme](u, NU)
    return u, exact(u0_N, ns*dt, N)
def coeffs(U, R):
    h = L/U.shape[1]
    A = np.stack([(np.roll(U,-1,1)-2*U+np.roll(U,1,1))/h**2,
                  (np.roll(U,-2,1)-2*np.roll(U,-1,1)+2*np.roll(U,1,1)-np.roll(U,2,1))/(2*h**3),
                  (np.roll(U,-2,1)-4*np.roll(U,-1,1)+6*U-4*np.roll(U,1,1)+np.roll(U,2,1))/h**4], 2)
    AtA = np.einsum('mni,mnk->mik', A, A) + 1e-9*np.eye(3)
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', A, R)[..., None])[..., 0]
def direction(C): return np.nan_to_num(C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))

def observe(scheme, N, u0_base, noise, gn):
    u0_N = antialias(u0_base, N)
    un, ex = run(scheme, N, u0_N)
    if noise > 0:
        nz = noise*np.sqrt(np.mean(ex**2))*gn.standard_normal(N); un = un + nz
    r = un - ex
    mag = np.sqrt(np.mean(r**2)) / (np.sqrt(np.mean(ex**2)) + 1e-12)          # native-N residual magnitude (for the order fit)
    d = direction(coeffs(antialias(un, COMMON)[None], antialias(r, COMMON)[None]))[0]
    return mag, d

def order_p(scheme, Nset, bases, noise, seed):                               # multi-resolution feature: convergence order
    gn = np.random.default_rng(seed); P = []
    for u0 in bases:
        mags = [observe(scheme, N, u0, noise, gn)[0] for N in Nset]
        P.append(-np.polyfit(np.log(Nset), np.log(np.array(mags) + 1e-12), 1)[0])
    return np.array(P)[:, None]
def snapshot_dir(scheme, N, bases, noise, seed):                            # single-snapshot baseline: direction at one grid
    gn = np.random.default_rng(seed)
    return np.array([observe(scheme, N, u0, noise, gn)[1] for u0 in bases])

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def auroc(Fa, Fb, ga, gb):
    X = np.vstack([Fa, Fb]); y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]; g = np.r_[ga, gb]
    return cross_val_score(CLF(), X, y, groups=g, cv=GroupKFold(5), scoring="roc_auc").mean()

# ================================================================ RUN
print(f"multi-resolution audit | linear advection, 4 schemes, {N_IC} ICs, resolution sets {R1} vs {R2}\n")
rng = np.random.default_rng(0); bases = [random_ic(N_BASE, rng) for _ in range(N_IC)]; ic = np.arange(N_IC); h = N_IC//2

NLOW, NMID, NHIGH = 48, 64, 96    # single-snapshot grids; NC2 uses the strong R5-scale change 64 vs 96
results = {}
for noise in (0.0, 0.01):
    yd = {sc: (0 if sc in DIFFUSIVE else 1) for sc in names}
    # order feature (multi-resolution): p over R1 for every scheme; upwind p over R2 for NC2
    pR1 = {sc: order_p(sc, R1, bases, noise, 1) for sc in names}
    pR2_up = order_p(UP, R2, bases, noise, 2)
    # single-snapshot baseline: direction at NMID for every scheme; upwind at NMID and NHIGH for NC2
    dMID = {sc: snapshot_dir(sc, NMID, bases, noise, 3) for sc in names}
    dHIGH_up = snapshot_dir(UP, NHIGH, bases, noise, 4)
    out = {}
    for tag, perfeat, nc2a, nc2b in [
        ("order", pR1, pR1[UP], pR2_up),                 # NC2: upwind {R1} vs upwind {R2}
        ("single", dMID, dMID[UP], dHIGH_up)]:           # NC2: upwind@64 vs upwind@96 (R5-scale)
        F = np.vstack([perfeat[sc] for sc in names]); y = np.concatenate([[yd[sc]]*N_IC for sc in names])
        g = np.concatenate([ic]*len(names))
        out[(tag, "scheme")] = cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5), scoring="roc_auc").mean()
        out[(tag, "nc2")] = auroc(nc2a, nc2b, ic, ic)
        out[(tag, "nc1")] = auroc(perfeat[UP][:h], perfeat[UP][h:], ic[:h], ic[h:])
    results[noise] = out
    if noise == 0.0:
        print("self-check, mean convergence order p per scheme (native-N magnitude; expect ~1 diffusive, ~2 dispersive):")
        for sc in names: print(f"   {sc:16s} p = {pR1[sc].mean():.2f}")
        print()

# ================================================================ table + decision
print(f"{'noise':>6} {'feature':22s} {'diff-vs-disp↑':>13s} {'NC2(grid)↓':>11s} {'NC1':>7s} {'margin':>8s}")
LBL = {"order":"order p (multi-res)", "single":"direction (single-snap)"}
for noise in (0.0, 0.01):
    for tag in ("order", "single"):
        o = results[noise]; s, n2, n1 = o[(tag,"scheme")], o[(tag,"nc2")], o[(tag,"nc1")]
        print(f"{noise*100:>5.0f}% {LBL[tag]:22s} {s:>13.3f} {n2:>11.3f} {n1:>7.3f} {s-n2:>+8.3f}")

with open(os.path.join(TAB, "multiresolution_results.csv"), "w") as f:
    f.write("noise,feature,diff_vs_disp,nc2_grid,nc1,margin\n")
    for noise in (0.0, 0.01):
        for tag in ("order", "single"):
            o = results[noise]; f.write(f"{noise},{tag},{o[(tag,'scheme')]:.4f},{o[(tag,'nc2')]:.4f},"
                                        f"{o[(tag,'nc1')]:.4f},{o[(tag,'scheme')]-o[(tag,'nc2')]:.4f}\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = np.arange(2); w = 0.35; plt.figure(figsize=(7, 5))
o = results[0.01]
plt.bar(xs-w/2, [o[(t,"scheme")] for t in ("order","single")], w, label="diff-vs-disp scheme↑", color="C0")
plt.bar(xs+w/2, [o[(t,"nc2")] for t in ("order","single")], w, label="grid control NC2↓ (want low)", color="C3")
plt.axhline(0.85, color="C0", ls="--", alpha=.5); plt.axhline(0.65, color="C3", ls="--", alpha=.5); plt.axhline(0.5, color="grey", ls=":")
plt.xticks(xs, ["order p\n(multi-resolution)", "direction\n(single-snapshot)"]); plt.ylim(0.3, 1.03)
plt.ylabel("AUROC (1% noise)"); plt.title("Multi-resolution audit: does convergence order break the grid confound?")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "multiresolution_result.png"), dpi=130)

print("\n" + "="*72 + "\nPRE-REGISTERED DECISION (degraded = 1% noise, order feature vs baseline)\n" + "="*72)
o = results[0.01]; s, n2 = o[("order","scheme")], o[("order","nc2")]; base_n2 = o[("single","nc2")]
print(f"order feature: diff-vs-disp={s:.3f}  NC2(grid)={n2:.3f}  margin={s-n2:+.3f}  | single-snapshot NC2={base_n2:.3f}")
if s >= 0.85 and n2 <= 0.65 and (s-n2) >= 0.20 and base_n2 > n2 + 0.10:
    print("\n[GO]  convergence order attributes diffusive-vs-dispersive while the grid control falls to chance,")
    print(f"  and the single-snapshot baseline still shows the confound (NC2 {base_n2:.2f} -> {n2:.2f}).")
    print("  Multi-resolution observation BREAKS the grid confound for the robust diffusive-vs-dispersive claim:")
    print("  the audit no longer requires known resolution for that distinction. (Same-order discrimination still")
    print("  needs the grid-dependent direction and is unchanged.)")
elif n2 > 0.65:
    print(f"\n[KILL]  multi-resolution NC2 still fires ({n2:.3f} > 0.65): the grid confound is irreducible even with")
    print("  multi-resolution access. Report as a stronger measured limit. Audit stays controlled-resolution-only.")
elif s < 0.85:
    print(f"\n[KILL]  order is too noise-fragile (diff-vs-disp {s:.3f} < 0.85 at 1% noise): the convergence-order")
    print("  feature does not survive noise. Audit stays controlled-resolution-only.")
else:
    print(f"\n[MIXED]  partial: order reduces the grid confound (NC2 {base_n2:.2f} -> {n2:.2f}) but does not clear the")
    print("  bar. Report as attenuation, not removal.")
print(f"\nartifacts -> {os.path.join(TAB, 'multiresolution_results.csv')}")
