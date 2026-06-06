"""
solver-forensics :: GRID-INVARIANT ATTRIBUTION GATE
========================================================================
NCS-upgrade experiment (NOT a rescue). R5 showed the open-solver audit detects a
numerical-configuration change but cannot separate a SCHEME change from a GRID
change in a fully blind setting (NC2 fired as hard as A-vs-B). This gate tests
whether grid-aware / nondimensionalized features can REDUCE that confound while
PRESERVING scheme-change detection.

R5 is the locked baseline; it stays unchanged unless a blind-compatible feature set
genuinely separates scheme from grid. KILL or MIXED -> R5's measured limit stands.

Same py-pde advection-diffusion setup as the audit demo. Contrasts (AUROC,
GroupKFold-by-IC, 1% field noise):
  A vs B   : central (A) vs upwind (B)            -- real SCHEME change, must stay HIGH
  NC2      : central@64 (A) vs central@96 (A')    -- pure GRID change, want it to DROP
  NC1      : central, IC-set1 vs IC-set2          -- IC/noise only, must stay at chance

FOUR feature sets:
  raw       : R5 feature - unit-direction of [c2,c3,c4] + c3/c2          (baseline)
  dxnorm    : c_p / dx^(p-1) (solver dx) then unit-direction + ratio     (needs known dx)
  common    : coarsen ALL observations to a COMMON grid, recover c       (blind-compatible)
  combined  : common-resolution recovery + dx-normalization              (needs known dx)

PRE-REGISTERED DECISION (fixed before running):
  GO    : some set has scheme-AUROC >= 0.85 AND grid-AUROC <= 0.65 AND
          margin(scheme-grid) >= 0.20 AND NC1 <= 0.60.
          -> if the winner is `common` (no solver-dx needed) = genuine blind upgrade;
             if only `dxnorm`/`combined` win = controlled-resolution tightening only.
  KILL  : best margin across all sets < 0.10  -> R5 limit confirmed, unchanged.
  MIXED : margin >= 0.10 but no set clears the GO bar -> partial, report as limit.

Pure py-pde + numpy + sklearn, CPU. numpy-2-safe.
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from pde import CartesianGrid, ScalarField, PDEBase
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

L, A_SPEED, D, T = 1.0, 1.0, 0.01, 0.30
N_A, N_GRID2, N_COMMON, N_BASE = 64, 96, 32, 192
N_IC, DT, SIGMA = 60, 2e-4, 0.01
FSETS = ["raw", "dxnorm", "common", "combined"]

class AdvDiff(PDEBase):
    def __init__(self, scheme): super().__init__(); self.scheme = scheme
    def evolution_rate(self, state, t=0):
        u = state.data; dx = state.grid.discretization[0]
        ux = (np.roll(u,-1)-np.roll(u,1))/(2*dx) if self.scheme == "central" else (u-np.roll(u,1))/dx
        return ScalarField(state.grid, -A_SPEED*ux + D*(np.roll(u,-1)-2*u+np.roll(u,1))/dx**2)

def exact(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
    return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*A_SPEED*t - D*k*k*t), n=N)
def ic_base(rng):
    x = np.linspace(0, L, N_BASE, endpoint=False); u = np.zeros(N_BASE)
    for _ in range(4): u += rng.normal()*np.sin(2*np.pi*rng.integers(1,5)*x + rng.uniform(0,2*np.pi))
    return 1.0 + 0.4*u/(np.max(np.abs(u)) + 1e-9)
def run(scheme, N, u0_N):
    g = CartesianGrid([[0, L]], N, periodic=True)
    return AdvDiff(scheme).solve(ScalarField(g, u0_N), t_range=T, dt=DT, solver="explicit",
                                 backend="numpy", tracker=None).data
def antialias(u, N_obs):                                       # proper Fourier resample to exactly N_obs (handles non-integer N/N_obs)
    N = len(u)
    if N == N_obs: return u
    F = np.fft.rfft(u)[:N_obs//2 + 1]
    return np.fft.irfft(F, n=N_obs) * (N_obs / N)
def coeffs(U, R):                                              # numpy-2-safe stacked solve
    h = L/U.shape[1]
    Axx   = (np.roll(U,-1,1) - 2*U + np.roll(U,1,1))/h**2
    Axxx  = (np.roll(U,-2,1) - 2*np.roll(U,-1,1) + 2*np.roll(U,1,1) - np.roll(U,2,1))/(2*h**3)
    Axxxx = (np.roll(U,-2,1) - 4*np.roll(U,-1,1) + 6*U - 4*np.roll(U,1,1) + np.roll(U,2,1))/h**4
    Am = np.stack([Axx, Axxx, Axxxx], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(3)
    Atb = np.einsum('mni,mn->mi', Am, R)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]
def feats(C):
    return np.nan_to_num(np.hstack([C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12),
                                    np.clip(np.nan_to_num(C[:,1]/C[:,0])[:,None], -10, 10)]))

def recover(fields, u0s, N, fset, seed):
    """Coefficient array (one per IC) for a feature set, at 1% field noise."""
    # FAIR grid comparison: raw/dxnorm observe at a COMMON 64-grid (not native N) so a pure
    # grid change is not an observation-resolution (h) mismatch. common/combined coarsen to 32.
    dx = L/N; nobs = N_COMMON if fset in ("common", "combined") else N_A
    gn = np.random.default_rng(seed); C = []
    for uf, u0 in zip(fields, u0s):
        ex = exact(u0, T, N)
        un = uf + SIGMA*np.sqrt(np.mean(uf**2))*gn.standard_normal(N)
        c = coeffs(antialias(un, nobs)[None], antialias(un - ex, nobs)[None])[0]
        if fset in ("dxnorm", "combined"):
            c = c / np.array([dx, dx**2, dx**3])               # nondimensionalize by solver dx powers
        C.append(c)
    return np.array(C)

def auroc(Ca, Cb, ica, icb, shuffle=False, seed=0):
    X = np.vstack([feats(Ca), feats(Cb)]); y = np.r_[np.zeros(len(Ca)), np.ones(len(Cb))]
    g = np.r_[ica, icb]
    if shuffle: y = np.random.default_rng(seed).permutation(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, groups=g, cv=GroupKFold(5), scoring="roc_auc").mean()

def stability_deg(C):                                          # median angle of per-IC unit-c from the mean dir
    u = C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12); m = u.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(np.median(np.degrees(np.arccos(np.clip(np.abs(u @ m), 0, 1)))))

# ================================================================ RUN
print(f"grid-invariant gate: py-pde adv-diff, A=central@{N_A} B=upwind@{N_A} A'=central@{N_GRID2}, "
      f"common-obs={N_COMMON}, {N_IC} ICs, field noise {SIGMA:.0%}\n")
rng = np.random.default_rng(0)
u0_base = [ic_base(rng) for _ in range(N_IC)]
u0_64 = [u[::N_BASE//N_A] for u in u0_base]
u0_96 = [u[::N_BASE//N_GRID2] for u in u0_base]
ic = np.arange(N_IC); half = N_IC//2
print("running py-pde solvers ...")
F_c64 = [run("central", N_A, u) for u in u0_64]
F_u64 = [run("upwind",  N_A, u) for u in u0_64]
F_c96 = [run("central", N_GRID2, u) for u in u0_96]
print("  done.\n")

floor = auroc(recover(F_c64, u0_64, N_A, "raw", 1), recover(F_u64, u0_64, N_A, "raw", 2),
              ic, ic, shuffle=True, seed=7)
res = {}
for fs in FSETS:
    A  = recover(F_c64, u0_64, N_A, fs, 1)
    B  = recover(F_u64, u0_64, N_A, fs, 2)
    Ap = recover(F_c96, u0_96, N_GRID2, fs, 3)
    scheme = auroc(A, B, ic, ic)
    grid   = auroc(A, Ap, ic, ic)
    nc1    = auroc(A[:half], A[half:], ic[:half], ic[half:])
    res[fs] = dict(scheme=scheme, grid=grid, nc1=nc1, margin=scheme - grid, stab=stability_deg(A))
    print(f"  evaluated: {fs}")

# ================================================================ table + csv + plot
print(f"\npermutation floor (shuffled labels) AUROC = {floor:.3f}  (chance ~0.5)\n")
print(f"{'feature set':<11} {'scheme↑':>9} {'grid↓':>8} {'margin↑':>9} {'NC1':>7} {'A-stab°':>9}  verdict")
def vtag(d):
    if d["scheme"] >= 0.85 and d["grid"] <= 0.65 and d["margin"] >= 0.20 and d["nc1"] <= 0.60: return "GO-candidate"
    if d["margin"] >= 0.10: return "partial"
    return "confound stands"
for fs in FSETS:
    d = res[fs]
    print(f"{fs:<11} {d['scheme']:>9.3f} {d['grid']:>8.3f} {d['margin']:>+9.3f} {d['nc1']:>7.3f} {d['stab']:>9.1f}  {vtag(d)}")

with open(os.path.join(TAB, "grid_invariance_results.csv"), "w") as f:
    f.write("feature_set,scheme_auroc,grid_auroc,margin,nc1_auroc,A_stability_deg\n")
    for fs in FSETS:
        d = res[fs]; f.write(f"{fs},{d['scheme']:.4f},{d['grid']:.4f},{d['margin']:.4f},{d['nc1']:.4f},{d['stab']:.2f}\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = np.arange(len(FSETS)); w = 0.38; plt.figure(figsize=(9, 5))
plt.bar(xs - w/2, [res[f]["scheme"] for f in FSETS], w, label="scheme change (A vs B) ↑", color="C0")
plt.bar(xs + w/2, [res[f]["grid"] for f in FSETS], w, label="grid change (NC2) ↓ want low", color="C3")
plt.axhline(0.85, color="C0", ls="--", alpha=.5); plt.axhline(0.65, color="C3", ls="--", alpha=.5)
plt.axhline(floor, color="grey", ls=":", label=f"permutation floor {floor:.2f}")
plt.xticks(xs, FSETS); plt.ylim(0.4, 1.03); plt.ylabel("AUROC (GroupKFold-by-IC)")
plt.title("Grid-invariant gate: can a feature keep scheme HIGH while pushing grid LOW?"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "grid_invariance_result.png"), dpi=130)

# ================================================================ pre-registered decision
print("\n" + "="*74 + "\nPRE-REGISTERED DECISION\n" + "="*74)
gocands = [fs for fs in FSETS if vtag(res[fs]) == "GO-candidate"]
best = max(FSETS, key=lambda f: res[f]["margin"]); best_margin = res[best]["margin"]
print(f"GO-candidate sets: {gocands or 'none'} | best margin = {best_margin:+.3f} ({best})")
if gocands:
    blind = "common" in gocands
    print(f"\n[GO - confound REDUCED]  feature set(s) {gocands} keep scheme detection high while pushing")
    print(f"  grid detection toward the floor.")
    if blind:
        print("  The winner includes `common` (needs NO knowledge of solver dx) -> GENUINE BLIND-SETTING")
        print("  upgrade: scheme-vs-grid separable from observation alone. This DOES change the auditability")
        print("  story; propose updating R5 (with a fresh confirmatory run before editing the locked draft).")
    else:
        print("  Winner is dx-normalized/combined only (REQUIRES known solver dx) -> tightens the CONTROLLED")
        print("  audit, does NOT extend to the blind setting. R5's blind-setting limit stands; note the")
        print("  controlled-resolution improvement as a refinement, not a removal of the confound.")
elif best_margin < 0.10:
    print(f"\n[KILL - confound STANDS]  no feature set separates grid from scheme (best margin {best_margin:+.3f}")
    print("  < 0.10). The scheme-vs-grid confound is not reducible by these features. R5 UNCHANGED;")
    print("  report as a confirmed measured limit (the confound is intrinsic to single-snapshot features).")
else:
    print(f"\n[MIXED - partial]  best set `{best}` reduces grid detection (margin {best_margin:+.3f}) but does")
    print(f"  not clear the GO bar (scheme>=0.85, grid<=0.65). Report as an auditability-limit improvement,")
    print("  not a solved confound. R5 UNCHANGED; the confound is attenuated, not removed.")
