"""
solver-forensics :: LIBRARY-SOLVER AUDIT  (py-pde, DOCUMENTED ground truth)
============================================================================
ITEM-1 deliverable. The toy proof-of-concept (open_solver_audit.py) changed a
HAND-WRITTEN advection STENCIL. This experiment instead audits a REAL change in
the internals of a PUBLISHED library (py-pde 0.56.1) that we did NOT write: the
documented TIME INTEGRATOR.

SAME PDE, SAME RHS object, SAME grid / dt / ICs across A and B. The ONLY thing
that differs is which py-pde solver class advances time:
  A = EulerSolver        -> explicit (forward) Euler, FIRST-ORDER in time
  B = RungeKuttaSolver   -> explicit Runge-Kutta, documented "order 5(4)"
Both are shipped, documented py-pde classes (pde.solvers); their order is stated
in py-pde's own docstrings (verified: RungeKuttaSolver.__doc__ = "Explicit
Runge-Kutta PDE solver of order 5(4)"; EulerSolver = "Explicit Euler solver").

PDE: advection-diffusion  u_t = -a u_x + D u_xx  on a periodic line.
Truth is ANALYTIC (linear: u_k(t) = u_k(0) e^{-i a k t - D k^2 t}), so the
residual r = u_solver - u_ref is genuinely solver-vs-truth, not contaminated by
a numerical reference. (A numerical fine-solve reference cross-check is included
for the methodology, gate #5.)

DOCUMENTED EXPECTATION (modified-equation / numerical-analysis textbook):
  Forward Euler in time has a leading O(dt) time-truncation error. Pushed
  through the PDE (u_tt = a^2 u_xx - 2 a D u_xxx + D^2 u_xxxx), its dominant
  added term for advection-dominated transport is  -(a^2 dt / 2) u_xx  -- i.e.
  forward Euler injects an ANTI-DIFFUSIVE 2nd-derivative (u_xx) time error.
  The order-5(4) Runge-Kutta solver has negligible time error, so ITS residual
  is essentially the spatial central-difference error, which is DISPERSIVE and
  loads on u_xxx. PREDICTION: the recovered coefficient DIRECTION for A loads on
  u_xx (the diffusive axis) far more than B does; the Euler-minus-RK pure-time
  residual has a NEGATIVE u_xx coefficient. Gate #4 verifies the recovered
  signature agrees with this documented expectation, sign-consistently across ICs.

PRE-REGISTERED (fixed before the final run; decision read at 1% field noise):
  Decision metric = GroupKFold-by-IC classification accuracy on the coefficient-
  DIRECTION features, with a label-PERMUTATION floor on every reported number.
  GO requires ALL:
   1. A-vs-B  >= 0.85
   2. NC1 (A vs A, ICs+noise only)        <= 0.60   (control: must sit ~chance)
   3a. NC2 (A vs A', GRID change only)    <= 0.70
   3b. (A-vs-B) - NC2                      >= 0.15   (scheme beats the confound)
   4. signature agrees with documented nature: Euler dir has MORE |u_xx| weight
      than RK (sign-consistent in >= 80% of ICs); Euler-minus-RK u_xx coeff < 0.
   5. reference-convergence: A-vs-B with a numerical fine-solve reference and with
      the analytic reference agree to within 0.05.
   6. robustness: 1-3 hold under coarsened (N=32) observation at 1% noise.
   7. not single-IC artefact: 5-fold MIN A-vs-B >= 0.75.
   8. permutation floor for A-vs-B <= 0.60 (label-shuffled accuracy is ~chance).
  KILL/DOWNGRADE if any: A-vs-B < 0.75 | NC1 > 0.65 | NC2 > 0.75 |
   (A-vs-B)-NC2 < 0.10 | sign-consistency < 0.70 | fold-min < 0.65 |
   perm-floor > 0.65 | signal gone under coarsening.
  OVERCLAIM GUARD: report "flagged the documented time-integrator change
   (first-order Euler vs high-order Runge-Kutta) and the recovered signature
   matches Euler's documented anti-diffusive leading time error", NOT "identified
   the exact integrator implementation".

Pure py-pde + numpy + scipy + sklearn, CPU. numpy-2 safe.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pde import CartesianGrid, ScalarField, PDEBase
from pde.solvers import EulerSolver, RungeKuttaSolver, Controller
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures")
TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ---- problem / solver settings (advection-dominated => time error is structured & detectable) ----
L, A_SPEED, D, T = 1.0, 1.0, 0.01, 0.30
N_A, N_GRID2 = 64, 96       # baseline solver grid; NC2 alternate grid (the resolution confound)
N_BASE = 192               # IC base resolution (divisible by 64 and 96)
DT = 4e-3                  # within CFL (a*dt/dx=0.26) and diffusion stability for both grids
N_IC = 60
SIGMA = 0.01              # field-relative observation noise (decision is read at this level)


# ============================================================== py-pde model + solvers
class AdvDiff(PDEBase):
    """u_t = -a u_x + D u_xx, central-in-space RHS. py-pde advances time."""

    def evolution_rate(self, state, t=0):
        u = state.data
        dx = state.grid.discretization[0]
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        uxx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        return ScalarField(state.grid, -A_SPEED * ux + D * uxx)


SOLVERS = {"euler": EulerSolver, "rk": RungeKuttaSolver}


def run(solver_key, N, u0_N, dt=DT):
    """Solve with the chosen documented py-pde TIME INTEGRATOR; everything else fixed."""
    g = CartesianGrid([[0, L]], N, periodic=True)
    solver = SOLVERS[solver_key](AdvDiff(), backend="numpy")
    return Controller(solver, t_range=T, tracker=None).run(ScalarField(g, u0_N).copy(), dt=dt).data


def exact(u0, t, N):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    return np.fft.irfft(np.fft.rfft(u0) * np.exp(-1j * k * A_SPEED * t - D * k * k * t), n=N)


def ic_base(rng):
    x = np.linspace(0, L, N_BASE, endpoint=False)
    u = np.zeros(N_BASE)
    for _ in range(4):
        u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 6) * x + rng.uniform(0, 2 * np.pi))
    return 1.0 + 0.4 * u / (np.max(np.abs(u)) + 1e-9)


def antialias(u, N_obs):
    """Proper Fourier resample to exactly N_obs (handles non-integer N/N_obs)."""
    N = len(u)
    if N == N_obs:
        return u
    F = np.fft.rfft(u)[: N_obs // 2 + 1]
    return np.fft.irfft(F, n=N_obs) * (N_obs / N)


# ============================================================== strong-form signature
def coeffs(U, R):
    """Least-squares coefficient vector c in r ~ c0 u_xx + c1 u_xxx + c2 u_xxxx (numpy-2 safe)."""
    h = L / U.shape[1]
    Axx = (np.roll(U, -1, 1) - 2 * U + np.roll(U, 1, 1)) / h ** 2
    Axxx = (np.roll(U, -2, 1) - 2 * np.roll(U, -1, 1) + 2 * np.roll(U, 1, 1) - np.roll(U, 2, 1)) / (2 * h ** 3)
    Axxxx = (np.roll(U, -2, 1) - 4 * np.roll(U, -1, 1) + 6 * U - 4 * np.roll(U, 1, 1) + np.roll(U, 2, 1)) / h ** 4
    Am = np.stack([Axx, Axxx, Axxxx], 2)
    AtA = np.einsum("mni,mnk->mik", Am, Am) + 1e-12 * np.eye(3)
    Atb = np.einsum("mni,mn->mi", Am, R)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]


def feats(C):
    """Unit-normalized DIRECTION + clipped c1/c0 ratio (magnitude-invariant signature)."""
    unit = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(np.nan_to_num(C[:, 1] / C[:, 0]), -10, 10)
    return np.nan_to_num(np.hstack([unit, ratio[:, None]]))


def recover(fields, u0s, N, sigma, n_obs, seed):
    """fields -> per-IC coefficient vector, from the OBSERVED (noisy, possibly coarsened) field."""
    gn = np.random.default_rng(seed)
    C = []
    for uf, u0 in zip(fields, u0s):
        ex = exact(u0, T, N)
        un = uf + sigma * np.sqrt(np.mean(uf ** 2)) * gn.standard_normal(N) if sigma > 0 else uf
        u_obs = antialias(un, n_obs)
        r_obs = antialias(un - ex, n_obs)
        C.append(coeffs(u_obs[None], r_obs[None])[0])
    return np.array(C)


def detect(Ca, Cb, ica, icb, n_perm=200, seed=0):
    """GroupKFold-by-IC accuracy + label-permutation floor. Returns (mean, min, perm_mean)."""
    X = np.vstack([feats(Ca), feats(Cb)])
    y = np.r_[np.zeros(len(Ca)), np.ones(len(Cb))]
    g = np.r_[ica, icb]  # same IC -> same fold (no IC leakage)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = GroupKFold(5)
    s = cross_val_score(clf, X, y, groups=g, cv=cv)
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perm.append(cross_val_score(clf, X, yp, groups=g, cv=cv).mean())
    return s.mean(), s.min(), float(np.mean(perm))


# ============================================================== RUN
if __name__ == "__main__":
    print("library-solver audit: py-pde advection-diffusion (analytic truth)")
    print(f"  A = EulerSolver (1st-order in time)  vs  B = RungeKuttaSolver (order 5(4))")
    print(f"  documented py-pde classes; same RHS / grid N={N_A} / dt={DT} / ICs; only the integrator differs")
    print(f"  {N_IC} paired ICs, field noise {SIGMA:.0%}\n")

    rng = np.random.default_rng(0)
    u0_base = [ic_base(rng) for _ in range(N_IC)]
    u0_64 = [u[:: N_BASE // N_A] for u in u0_base]
    u0_96 = [u[:: N_BASE // N_GRID2] for u in u0_base]
    ic = np.arange(N_IC)

    print("running py-pde solvers (euler@64, rk@64, euler@96) ...")
    F_euler64 = [run("euler", N_A, u) for u in u0_64]
    F_rk64 = [run("rk", N_A, u) for u in u0_64]
    F_euler96 = [run("euler", N_GRID2, u) for u in u0_96]
    print("  done.")

    # stability sanity (solvers must be stable for the residual to be trustworthy)
    max_e = max(np.max(np.abs(f)) for f in F_euler64)
    max_r = max(np.max(np.abs(f)) for f in F_rk64)
    print(f"  stability: max|euler|={max_e:.3f}  max|rk|={max_r:.3f}  (bounded => stable)\n")

    def cond(noise, n_obs):
        return dict(
            A=recover(F_euler64, u0_64, N_A, noise, n_obs, 1),
            B=recover(F_rk64, u0_64, N_A, noise, n_obs, 2),
            Ap=recover(F_euler96, u0_96, N_GRID2, noise, n_obs, 3),
        )

    SETTINGS = [("clean", 0.0, N_A), ("1% noise", SIGMA, N_A), ("1%+coarsened(32)", SIGMA, 32)]
    rows = {}
    for name, noise, nobs in SETTINGS:
        c = cond(noise, nobs)
        half = N_IC // 2
        ab, ab_min, ab_perm = detect(c["A"], c["B"], ic, ic)
        nc1, _, nc1_perm = detect(c["A"][:half], c["A"][half:], ic[:half], ic[half:])
        nc2, _, _ = detect(c["A"], c["Ap"], ic, ic)
        rows[name] = dict(ab=ab, ab_min=ab_min, ab_perm=ab_perm, nc1=nc1, nc1_perm=nc1_perm, nc2=nc2, c=c)
        print(f"  evaluated: {name}")

    # ---- interpretability (#4): documented prediction = Euler loads MORE on u_xx than RK ----
    cc = rows["clean"]["c"]
    dirA = cc["A"] / (np.linalg.norm(cc["A"], axis=1, keepdims=True) + 1e-12)
    dirB = cc["B"] / (np.linalg.norm(cc["B"], axis=1, keepdims=True) + 1e-12)
    # per-IC: does Euler have more |u_xx| direction weight than RK?  (documented: yes)
    sign_consistent = float(np.mean(np.abs(dirA[:, 0]) > np.abs(dirB[:, 0])))
    mc_A = dirA.mean(0)
    mc_B = dirB.mean(0)
    # pure time error: Euler-minus-RK should have NEGATIVE u_xx coeff (anti-diffusive forward Euler)
    h = L / N_A
    timeerr_c = []
    for ue, ur in zip(F_euler64, F_rk64):
        timeerr_c.append(coeffs(ue[None], (ue - ur)[None])[0])
    timeerr_c = np.array(timeerr_c)
    timeerr_uxx_neg = float(np.mean(timeerr_c[:, 0] < 0))
    # median (not mean) -- a rare near-singular AtA on the noiseless Euler-RK residual can blow one
    # IC up; the robust summary is the median coefficient and the per-IC sign fraction above.
    mean_timeerr = np.median(timeerr_c, axis=0)

    # ---- magnitude-aware confound diagnostic (the substantive NC2 finding) ----
    # Binary GroupKFold accuracy answers "is there ANY consistent shift?" -- it saturates at 1.0 for an
    # arbitrarily small but systematic offset. The physically meaningful question is HOW LARGE the
    # signature rotation is. Measure the angle between mean unit-signatures: the documented integrator
    # change vs the grid-change confound, against the within-class per-IC spread.
    dirAp = cc["Ap"] / (np.linalg.norm(cc["Ap"], axis=1, keepdims=True) + 1e-12)

    def _ang(a, b):
        return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))

    def _meanunit(D):
        m = D.mean(0)
        return m / (np.linalg.norm(m) + 1e-12)

    muA, muB, muAp = _meanunit(dirA), _meanunit(dirB), _meanunit(dirAp)
    ang_integrator = _ang(muA, muB)        # Euler vs RK signature rotation (the signal)
    ang_grid = _ang(muA, muAp)             # Euler@64 vs Euler@96 (the confound)
    within_spread = float(np.std([_ang(d, muA) for d in dirA]))  # per-IC scatter of the Euler signature
    ang_ratio = ang_integrator / (ang_grid + 1e-9)

    # ---- reference-convergence (#5): CONVERGED numerical fine-solve reference vs analytic, on a subset ----
    # A genuine reference: high-order RK on a fine grid with a small CFL-safe dt, so both space and time
    # errors are negligible (it should agree with the analytic exact). Verify the convergence, then audit
    # using this numerical reference and check the A-vs-B accuracy matches the analytic-reference result.
    sub = slice(0, 30)
    Nref = 384
    # fine-grid stability: explicit RK needs dt < dx^2/(2D) = (1/384)^2/(2*0.01) = 3.4e-4 (diffusion)
    # and a*dt/dx < ~1 (advection). pick dt = 1.5e-4: well inside both -> converged in space & time.
    DT_REF = 1.5e-4
    u0_ref = [antialias(u, Nref) for u in u0_base[sub]]  # Fourier-upsample IC to the fine grid
    ref_fine = [run("rk", Nref, u, dt=DT_REF) for u in u0_ref]  # converged numerical reference
    # confirm the numerical reference really is converged (matches the analytic exact)
    ref_vs_analytic = np.mean(
        [
            np.sqrt(np.mean((rf - exact(u0, T, Nref)) ** 2)) / np.sqrt(np.mean(exact(u0, T, Nref) ** 2))
            for rf, u0 in zip(ref_fine, u0_ref)
        ]
    )
    print(f"  numerical reference convergence: rel-rms(fine-RK - analytic) = {ref_vs_analytic:.2e}")

    def recover_numref(fields, refs, seed):
        gn = np.random.default_rng(seed)
        C = []
        for uf, rf in zip(fields, refs):
            ref_on = antialias(rf, N_A)
            un = uf + SIGMA * np.sqrt(np.mean(uf ** 2)) * gn.standard_normal(N_A)
            C.append(coeffs(un[None], (un - ref_on)[None])[0])
        return np.array(C)

    A_num = recover_numref(F_euler64[sub], ref_fine, 1)
    B_num = recover_numref(F_rk64[sub], ref_fine, 2)
    ab_numref, _, _ = detect(A_num, B_num, ic[sub], ic[sub], n_perm=50)
    ab_analytic_sub, _, _ = detect(
        rows["1% noise"]["c"]["A"][sub], rows["1% noise"]["c"]["B"][sub], ic[sub], ic[sub], n_perm=50
    )
    ref_gap = abs(ab_numref - ab_analytic_sub)

    # ============================================================== report
    print("\n=== DETECTION ACCURACY (GroupKFold-by-IC, coefficient-direction features) ===")
    print(f"{'pair':<32}" + "".join(f"{s[0]:>20}" for s in SETTINGS))
    labels = {
        "ab": "A vs B  (INTEGRATOR change)",
        "nc1": "NC1  (A vs A, ICs+noise)",
        "nc2": "NC2  (A vs A', GRID change)",
    }
    for key in ("ab", "nc1", "nc2"):
        print(f"{labels[key]:<32}" + "".join(f"{rows[s[0]][key]:>20.3f}" for s in SETTINGS))
    print(f"{'A-vs-B permutation floor':<32}" + "".join(f"{rows[s[0]]['ab_perm']:>20.3f}" for s in SETTINGS))

    print(
        f"\ninterpretable shift (documented): Euler dir=[{mc_A[0]:+.2f},{mc_A[1]:+.2f},{mc_A[2]:+.2f}]  "
        f"RK dir=[{mc_B[0]:+.2f},{mc_B[1]:+.2f},{mc_B[2]:+.2f}]"
    )
    print(
        f"  -> Euler loads MORE on |u_xx| than RK in {sign_consistent*100:.0f}% of ICs "
        f"(documented forward-Euler anti-diffusive time error)"
    )
    print(
        f"  -> Euler-minus-RK pure-time residual median c=[{mean_timeerr[0]:+.2e},{mean_timeerr[1]:+.2e},"
        f"{mean_timeerr[2]:+.2e}]  u_xx<0 in {timeerr_uxx_neg*100:.0f}% of ICs (predicted negative)"
    )
    print(
        f"reference-convergence: A-vs-B numerical-ref={ab_numref:.3f} vs analytic-ref={ab_analytic_sub:.3f}  "
        f"(gap {ref_gap:.3f})"
    )
    print(
        f"\nMAGNITUDE-AWARE confound diagnostic (mean unit-signature angles):"
        f"\n  integrator change (Euler vs RK)   = {ang_integrator:5.1f} deg   <- the documented signal"
        f"\n  grid change       (Euler 64 vs 96)= {ang_grid:5.2f} deg   <- the NC2 confound"
        f"\n  within-IC spread of Euler sig.    = {within_spread:5.1f} deg"
        f"\n  => integrator effect is {ang_ratio:.0f}x the grid-change effect in signature angle,"
        f"\n     but binary GroupKFold accuracy saturates (1.0) on the grid offset too because it is"
        f"\n     tiny yet perfectly SYSTEMATIC; accuracy measures separability, not magnitude."
    )

    # ---- CSV ----
    csv = os.path.join(TAB, "library_solver_audit_results.csv")
    with open(csv, "w") as f:
        f.write("setting,A_vs_B,A_vs_B_foldmin,A_vs_B_perm,NC1_ICs,NC1_perm,NC2_grid\n")
        for s in SETTINGS:
            r = rows[s[0]]
            f.write(
                f"{s[0]},{r['ab']:.4f},{r['ab_min']:.4f},{r['ab_perm']:.4f},"
                f"{r['nc1']:.4f},{r['nc1_perm']:.4f},{r['nc2']:.4f}\n"
            )
    csv2 = os.path.join(TAB, "library_solver_audit_signature_angles.csv")
    with open(csv2, "w") as f:
        f.write("quantity,degrees\n")
        f.write(f"integrator_euler_vs_rk,{ang_integrator:.3f}\n")
        f.write(f"grid_euler64_vs_euler96,{ang_grid:.3f}\n")
        f.write(f"within_ic_spread_euler,{within_spread:.3f}\n")
        f.write(f"ratio_integrator_over_grid,{ang_ratio:.3f}\n")
    print(f"\nmetrics -> {csv}")
    print(f"angles  -> {csv2}")

    # ---- figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # panel A: detection bars across settings
    xs = np.arange(len(SETTINGS))
    w = 0.26
    for i, (key, lab) in enumerate(labels.items()):
        ax[0].bar(xs + (i - 1) * w, [rows[s[0]][key] for s in SETTINGS], w, label=lab)
    ax[0].axhline(0.85, color="green", ls="--", alpha=0.6, label="GO bar (A-vs-B)")
    ax[0].axhline(0.5, color="grey", ls=":", alpha=0.7, label="chance")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([s[0] for s in SETTINGS], fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("GroupKFold-by-IC accuracy")
    ax[0].set_title("Integrator change detected, IC change not (NC1);\ngrid change (NC2) also separable -- the measured confound")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].text(-0.12, 1.04, "A", transform=ax[0].transAxes, fontsize=15, fontweight="bold")

    # panel B: recovered signature directions (Euler vs RK) -- the documented mechanism
    comps = ["$u_{xx}$", "$u_{xxx}$", "$u_{xxxx}$"]
    xb = np.arange(3)
    ax[1].bar(xb - 0.18, mc_A, 0.36, label="A = Euler (1st-order)", color=sns.color_palette("muted")[0])
    ax[1].bar(xb + 0.18, mc_B, 0.36, label="B = Runge-Kutta 5(4)", color=sns.color_palette("muted")[1])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels(comps, fontsize=11)
    ax[1].set_ylabel("mean unit-signature component")
    ax[1].set_title("Signature matches Euler's documented\nanti-diffusive ($u_{xx}$) time error")
    ax[1].legend(fontsize=9)
    ax[1].text(-0.12, 1.04, "B", transform=ax[1].transAxes, fontsize=15, fontweight="bold")

    fig.tight_layout()
    figpath = os.path.join(FIG, "library_solver_audit_result.png")
    fig.savefig(figpath, dpi=130)
    print(f"figure  -> {figpath}")

    # ============================================================== pre-registered decision
    print("\n" + "=" * 74 + "\nPRE-REGISTERED DECISION  (read at 1% field noise)\n" + "=" * 74)
    ab = rows["1% noise"]["ab"]
    nc1 = rows["1% noise"]["nc1"]
    nc2 = rows["1% noise"]["nc2"]
    abc = rows["1%+coarsened(32)"]["ab"]
    nc1c = rows["1%+coarsened(32)"]["nc1"]
    nc2c = rows["1%+coarsened(32)"]["nc2"]
    foldmin = rows["1% noise"]["ab_min"]
    abperm = rows["1% noise"]["ab_perm"]
    checks = {
        "1. A-vs-B >= 0.85":                         ab >= 0.85,
        "2. NC1 <= 0.60":                            nc1 <= 0.60,
        "3a. NC2 <= 0.70":                           nc2 <= 0.70,
        "3b. (A-vs-B) - NC2 >= 0.15":                (ab - nc2) >= 0.15,
        "4a. signature: Euler>|u_xx| 80% ICs":       sign_consistent >= 0.80,
        "4b. Euler-RK u_xx<0 (anti-diffusive)":      timeerr_uxx_neg >= 0.80,
        "5. ref-convergence gap <= 0.05":            ref_gap <= 0.05,
        "6a. robust: coarsened A-vs-B >= 0.85":      abc >= 0.85,
        "6b. robust: coarsened NC1<=.60,NC2<=.70":   (nc1c <= 0.60 and nc2c <= 0.70),
        "7. fold-min A-vs-B >= 0.75":                foldmin >= 0.75,
        "8. perm floor A-vs-B <= 0.60":              abperm <= 0.60,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print(
        f"\n  numbers: A-vs-B(1%)={ab:.3f}  NC1={nc1:.3f}  NC2={nc2:.3f}  margin={ab-nc2:+.3f}  "
        f"coarsened A-vs-B={abc:.3f}  fold-min={foldmin:.3f}  perm-floor={abperm:.3f}"
    )
    print(
        f"           sign-consistency={sign_consistent*100:.0f}%  Euler-RK u_xx<0={timeerr_uxx_neg*100:.0f}%  "
        f"ref-gap={ref_gap:.3f}"
    )
    if all(checks.values()):
        print("\n[LIBRARY AUDIT GO]  On a published library (py-pde) we did NOT write, the modified-")
        print("  equation signature flags the DOCUMENTED time-integrator change (first-order Euler vs")
        print("  order-5(4) Runge-Kutta) from a single output field, the recovered signature MATCHES")
        print("  py-pde's documented expectation (forward Euler's anti-diffusive u_xx time error), and")
        print("  it does NOT fire on IC variation (NC1) OR a grid change (NC2). CLAIM: 'flagged the")
        print("  documented integrator change; signature consistent with first-order Euler time error' -")
        print("  NOT 'identified the exact integrator implementation'.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n[LIBRARY AUDIT DOWNGRADE]  failed gates: {failed}")
        print("  Report the failing gate(s) as a measured limit; do not claim the full audit capability.")
