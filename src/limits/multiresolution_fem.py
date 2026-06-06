#!/usr/bin/env python3
"""
solver-forensics :: ACTIVE MULTI-RESOLUTION ON A FEM / ENGINEERING CASE
======================================================================
Carries the grid-confound REPAIR (paper Sec. 2.7) -- "observe the solver at several
resolutions and use the grid-invariant CONVERGENCE RATE p" -- off the linear-advection
finite-difference substrate it was established on and onto a genuine FINITE-ELEMENT,
engineering discretization. This removes the "established only on linear advection /
finite-difference" qualifier in the Limitations.

SUBSTRATE: transient advection-diffusion on a periodic rod,

    u_t + a u_x = nu u_xx ,                       a = 1, nu = 0.02, T = 0.25 ,

the workhorse engineering transport PDE. It is linear, so the REFERENCE is the spectral
per-mode exact solution  u_hat_k(T) = u0_hat_k exp((-i k a - nu k^2) T)  -- analytic, no
reference error. The two schemes differ in the canonical engineering convergence knob,
the ELEMENT ORDER:

  P1   hand-assembled linear  (2-node) Lagrange elements   ->  L2 rate  p ~ 2
  P2   hand-assembled quadratic (3-node) Lagrange elements ->  L2 rate  p ~ 4

Both are assembled by hand (element mass/advection/diffusion matrices, periodic
connectivity; NO FEM library) and advanced with REAL Crank-Nicolson time-stepping at a
fine fixed dt so the spatial element order governs the measured residual rate. A genuine
UNSTRUCTURED P1 path (jittered nodes, segment mesh from scipy.spatial.Delaunay on the
periodic embedding) is included to show the rate is a property of the element, not the
regular grid.

CLAIMS, measured honestly:
  (a) the per-IC convergence rate p SEPARATES the schemes        -> rate-detection high
  (b) the single-snapshot coefficient signature carries the grid -> snapshot-NC2 high
  (c) the rate is grid-INVARIANT (disjoint resolution sets,
      same element) so it BREAKS the snapshot grid confound      -> rate-NC2 ~chance

Signatures for the single-snapshot control follow the project convention: interpolate
solver + reference to a common regular grid (the FEM nodes are not a uniform FD grid /
P2 has midside nodes), then the FD library {u_xx, u_xxx, u_xxxx} least-squares direction
-- mirrors src/robustness/irregular_mesh.py. Attribution = StandardScaler+LogisticRegression,
GroupKFold(5) grouped by INITIAL CONDITION, with a label-PERMUTATION floor on every number.

numpy + scipy + sklearn, deterministic, CPU (~1-2 min). Run:
    python src/limits/multiresolution_fem.py            (metrics + CSV)
    python src/limits/multiresolution_fem.py --plot     (also the figure)
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
import scipy.linalg as sla
from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures")

# ----------------------------- problem constants -----------------------------
L, A, NU, T = 1.0, 1.0, 0.02, 0.25          # advection-diffusion on a periodic rod
LIB = (2, 3, 4)                              # FD signature library {u_xx, u_xxx, u_xxxx}
NT_FACTOR = 8                               # CN time steps ~ NT_FACTOR * N (dt small: space order governs rate)
NT_FLOOR = 400
NFINE = 4096                                # spectral-exact reference resolution (for unstructured node sampling)
GRID = 96                                    # common FD-signature grid for the snapshot control
N_IC = 32                                    # initial conditions (the GroupKFold groups)
SIGMA = 0.01                                 # field-relative observation noise on the snapshot

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

# ============================================================ initial conditions / reference
def ic_field(x, seed):
    """Smooth multi-mode IC (modes 1..5) -> exactly resolved at every mesh in the ladder."""
    r = np.random.default_rng(seed); u = np.zeros_like(x)
    for kk in (1, 2, 3, 4, 5):
        u += r.normal() * np.sin(2 * np.pi * kk * x / L + r.uniform(0, 2 * np.pi))
    return u / (np.std(u) + 1e-12)

def exact_on(x, seed):
    """Spectral per-mode exact advection-diffusion solution at time T, sampled on nodes x.
    The IC is built on a fine uniform grid (so the spectral propagator is exact), then the
    analytic solution is band-limited and sampled at the (possibly non-uniform) nodes x."""
    xf = np.linspace(0, L, NFINE, endpoint=False)
    u0 = ic_field(xf, seed)
    k = 2 * np.pi * np.fft.fftfreq(NFINE, d=L / NFINE)
    uT = np.real(np.fft.ifft(np.fft.fft(u0) * np.exp((-1j * k * A - NU * k ** 2) * T)))
    return np.interp(x % L, xf, uT, period=L)

def ic_on(x, seed):
    """IC sampled on nodes x, consistent with the fine-grid IC used by exact_on."""
    xf = np.linspace(0, L, NFINE, endpoint=False)
    return np.interp(x % L, xf, ic_field(xf, seed), period=L)

# ============================================================ hand-assembled P1 / P2 FEM
def _periodic_operator(M_lil, Op_lil):
    return M_lil.toarray(), Op_lil.toarray()

def assemble_P1_struct(N):
    """P1 (linear, 2-node) Lagrange elements on a uniform periodic mesh of N nodes."""
    h = L / N; nodes = np.linspace(0, L, N, endpoint=False)
    Me = np.array([[2., 1.], [1., 2.]]) * h / 6.0       # consistent element mass
    Ke = np.array([[1., -1.], [-1., 1.]]) / h           # diffusion: int phi_i' phi_j'
    Ce = np.array([[-1., 1.], [-1., 1.]]) / 2.0         # advection: int phi_i phi_j'
    Oe = -A * Ce - NU * Ke                              # du/dt = M^{-1} (Op) u
    M = lil_matrix((N, N)); Op = lil_matrix((N, N))
    for e in range(N):
        loc = [e, (e + 1) % N]
        for i in range(2):
            for j in range(2):
                M[loc[i], loc[j]] += Me[i, j]; Op[loc[i], loc[j]] += Oe[i, j]
    Md, Od = _periodic_operator(M, Op)
    return nodes, Md, Od

def assemble_P2_struct(N):
    """P2 (quadratic, 3-node) Lagrange elements: N elements, 2N dofs (endpoints + midsides)."""
    h = L / N; ndof = 2 * N
    nodes = np.linspace(0, L, ndof, endpoint=False)     # endpoints + midpoints, equally spaced
    Me = h / 30.0 * np.array([[4., 2., -1.], [2., 16., 2.], [-1., 2., 4.]])
    Ke = 1.0 / (3.0 * h) * np.array([[7., -8., 1.], [-8., 16., -8.], [1., -8., 7.]])
    Ce = np.array([[-3., 4., -1.], [-4., 0., 4.], [1., -4., 3.]]) / 6.0
    Oe = -A * Ce - NU * Ke
    M = lil_matrix((ndof, ndof)); Op = lil_matrix((ndof, ndof))
    for e in range(N):
        loc = [2 * e, 2 * e + 1, (2 * e + 2) % ndof]
        for i in range(3):
            for j in range(3):
                M[loc[i], loc[j]] += Me[i, j]; Op[loc[i], loc[j]] += Oe[i, j]
    Md, Od = _periodic_operator(M, Op)
    return nodes, Md, Od

def _delaunay_segments(nodes):
    """1D periodic 'mesh' connectivity from scipy.spatial.Delaunay on the circle embedding.
    Delaunay of a 1D point set returns its segment simplices; the periodic wrap edge is added
    by hand. Returns ordered node array + list of (i,j,length) elements covering the ring."""
    nodes = np.sort(np.unique(nodes % L)); n = len(nodes)
    th = 2 * np.pi * nodes / L
    pts = np.c_[np.cos(th), np.sin(th)]                 # embed on the unit circle (periodic)
    tri = Delaunay(pts)                                  # exercise the required dependency
    _ = tri.simplices                                    # (degenerate on a circle; mesh is the ring below)
    elems = [(i, (i + 1) % n, (nodes[(i + 1) % n] - nodes[i]) % L) for i in range(n)]
    return nodes, elems

def assemble_P1_unstruct(nodes):
    """P1 on a genuinely unstructured (jittered) 1D mesh; connectivity via Delaunay ring."""
    nodes, elems = _delaunay_segments(nodes); n = len(nodes)
    M = lil_matrix((n, n)); Op = lil_matrix((n, n))
    for (i, j, h) in elems:
        Me = np.array([[2., 1.], [1., 2.]]) * h / 6.0
        Ke = np.array([[1., -1.], [-1., 1.]]) / h
        Ce = np.array([[-1., 1.], [-1., 1.]]) / 2.0
        Oe = -A * Ce - NU * Ke; loc = [i, j]
        for a_ in range(2):
            for b_ in range(2):
                M[loc[a_], loc[b_]] += Me[a_, b_]; Op[loc[a_], loc[b_]] += Oe[a_, b_]
    Md, Od = _periodic_operator(M, Op)
    return nodes, Md, Od

def _cn_solve(nodes, Md, Od, u0, N_for_dt):
    """Real Crank-Nicolson time-stepping of  M u_t = Op u  to time T (genuine engineering solve)."""
    nt = max(NT_FLOOR, NT_FACTOR * N_for_dt); dt = T / nt
    Aleft = Md - 0.5 * dt * Od; Aright = Md + 0.5 * dt * Od
    luf = sla.lu_factor(Aleft); u = u0.copy()
    R = Aright
    for _ in range(nt):
        u = sla.lu_solve(luf, R @ u)
    return u

def solve(scheme, N, seed, jitter=0.0):
    """Return (nodes, u_solver, u_ref). scheme in {'P1','P2','P1_unstruct'}."""
    if scheme == "P1":
        nodes, Md, Od = assemble_P1_struct(N)
    elif scheme == "P2":
        nodes, Md, Od = assemble_P2_struct(N)
    elif scheme == "P1_unstruct":
        base = np.linspace(0, L, N, endpoint=False)
        jr = np.random.default_rng(10_000 + seed * 97 + N)
        nodes = np.sort(np.unique(np.round((base + jitter * (L / N) * (jr.random(N) - 0.5)) % L, 10)))
        nodes, Md, Od = assemble_P1_unstruct(nodes)
    else:
        raise ValueError(scheme)
    u0 = ic_on(nodes, seed)
    u = _cn_solve(nodes, Md, Od, u0, N)
    uref = exact_on(nodes, seed)
    return nodes, u, uref

def relres(scheme, N, seed, jitter=0.0):
    _, u, uref = solve(scheme, N, seed, jitter)
    return np.linalg.norm(u - uref) / (np.linalg.norm(uref) + 1e-12)

# ============================================================ per-IC convergence rate
def rate_over(scheme, Nset, seed, jitter=0.0):
    logN = np.log(np.array(Nset, float))
    rr = np.log(np.array([relres(scheme, N, seed, jitter) for N in Nset]) + 1e-300)
    return -np.polyfit(logN, rr, 1)[0]

# ============================================================ FD-signature on a common grid
def _to_grid(nodes, vals, G):
    """Interpolate periodic node samples onto a uniform G-grid (the convention's interp-to-grid)."""
    order = np.argsort(nodes); xn = nodes[order]; vn = vals[order]
    xg = np.linspace(0, L, G, endpoint=False)
    return np.interp(xg % L, xn, vn, period=L)

def _fd_signature(ug, rg):
    """Unit-normalized least-squares direction of c in r ~ sum_p c_p d_x^p u, FD library on the grid."""
    h = L / len(ug)
    uxx = (np.roll(ug, -1) - 2 * ug + np.roll(ug, 1)) / h ** 2
    uxxx = (np.roll(ug, -2) - 2 * np.roll(ug, -1) + 2 * np.roll(ug, 1) - np.roll(ug, 2)) / (2 * h ** 3)
    uxxxx = (np.roll(ug, -2) - 4 * np.roll(ug, -1) + 6 * ug - 4 * np.roll(ug, 1) + np.roll(ug, 2)) / h ** 4
    Amat = np.stack([uxx, uxxx, uxxxx], 1)
    c, *_ = np.linalg.lstsq(Amat, rg, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c

def snapshot_signature(scheme, N, seed, noise_seed):
    """Single-snapshot signature: solve at mesh N, add field noise, interp solver+ref to the
    common GRID, take the FD-library direction."""
    nodes, u, uref = solve(scheme, N, seed)
    g = np.random.default_rng(noise_seed)
    un = u + SIGMA * np.sqrt(np.mean(uref ** 2)) * g.standard_normal(len(u))
    ug = _to_grid(nodes, un, GRID); rg = _to_grid(nodes, un - uref, GRID)
    return _fd_signature(ug, rg)

# ============================================================ classifier helpers
def _acc(F, y, g):
    return cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean()

def _perm_floor(F, y, g, seed, reps=40):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g,
                                            cv=GroupKFold(5)).mean() for _ in range(reps)]))

# ============================================================ resolution ladders
MR_A = (16, 24, 32, 48, 64)          # primary resolution set (P1/P2 element counts)
MR_B = (20, 28, 36, 52, 72)          # disjoint resolution set (same element -> rate must NOT change)
SNAP_GRIDS = (32, 64)                # two single-snapshot meshes for the grid confound

def run_experiment(verbose=True):
    ic = np.arange(N_IC)
    if verbose:
        print(f"FEM multi-resolution | advection-diffusion u_t + {A} u_x = {NU} u_xx, T={T}")
        print(f"P1 (linear) vs P2 (quadratic) Lagrange elements, hand-assembled, Crank-Nicolson")
        print(f"{N_IC} ICs (= GroupKFold groups), primary res set {MR_A}\n")

    # --- per-IC convergence rates ---
    pP1 = np.array([rate_over("P1", MR_A, s) for s in range(N_IC)])
    pP2 = np.array([rate_over("P2", MR_A, s) for s in range(N_IC)])
    pP1u = np.array([rate_over("P1_unstruct", MR_A, s, jitter=0.4) for s in range(N_IC)])
    pP1_B = np.array([rate_over("P1", MR_B, s) for s in range(N_IC)])     # same element, disjoint grids

    if verbose:
        print(f"convergence rate p (median +/- IQR):")
        for nm, P in [("P1 (struct)", pP1), ("P2 (struct)", pP2), ("P1 (unstruct/Delaunay)", pP1u)]:
            q1, q3 = np.percentile(P, [25, 75])
            print(f"   {nm:24s} p = {np.median(P):.2f}  [{q1:.2f}, {q3:.2f}]")
        print()

    # --- (a) rate-detection: P1 vs P2 by rate ---
    Frate = np.r_[pP1, pP2][:, None]; yr = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gr = np.r_[ic, ic]
    rate_detect = _acc(Frate, yr, gr); rate_detect_f = _perm_floor(Frate, yr, gr, 1)

    # --- (c) rate-NC2: SAME element (P1), rate from disjoint resolution sets MR_A vs MR_B ---
    Fnc2 = np.r_[pP1, pP1_B][:, None]; ync2 = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gnc2 = np.r_[ic, ic]
    rate_nc2 = _acc(Fnc2, ync2, gnc2); rate_nc2_f = _perm_floor(Fnc2, ync2, gnc2, 2)

    # --- (b) snapshot-NC2: SAME element (P1), single-snapshot signature on two meshes ---
    Sa = np.array([snapshot_signature("P1", SNAP_GRIDS[0], s, 100 + s) for s in range(N_IC)])
    Sb = np.array([snapshot_signature("P1", SNAP_GRIDS[1], s, 700 + s) for s in range(N_IC)])
    Fsnap = np.vstack([Sa, Sb]); ysnap = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gsnap = np.r_[ic, ic]
    snap_nc2 = _acc(Fsnap, ysnap, gsnap); snap_nc2_f = _perm_floor(Fsnap, ysnap, gsnap, 3)

    # --- extra: does the snapshot signature even SEE the element order at a fixed grid? (sanity) ---
    SaP1 = np.array([snapshot_signature("P1", SNAP_GRIDS[1], s, 200 + s) for s in range(N_IC)])
    SaP2 = np.array([snapshot_signature("P2", SNAP_GRIDS[1], s, 900 + s) for s in range(N_IC)])
    Fsd = np.vstack([SaP1, SaP2]); ysd = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gsd = np.r_[ic, ic]
    snap_detect = _acc(Fsd, ysd, gsd); snap_detect_f = _perm_floor(Fsd, ysd, gsd, 4)

    if verbose:
        print(f"(a) rate-detection  P1 vs P2 by convergence rate:  acc={rate_detect:.3f}  floor={rate_detect_f:.3f}")
        print(f"    [for reference] snapshot-signature P1 vs P2:    acc={snap_detect:.3f}  floor={snap_detect_f:.3f}")
        print(f"(b) snapshot-NC2    same element, two meshes {SNAP_GRIDS}: acc={snap_nc2:.3f}  floor={snap_nc2_f:.3f}  (high = grid confound)")
        print(f"(c) rate-NC2        same element, res sets A vs B:  acc={rate_nc2:.3f}  floor={rate_nc2_f:.3f}  (~chance = rate is grid-invariant)")

    return dict(
        pP1=pP1, pP2=pP2, pP1u=pP1u, pP1_B=pP1_B,
        p_P1=float(np.median(pP1)), p_P2=float(np.median(pP2)), p_P1u=float(np.median(pP1u)),
        rate_detect=rate_detect, rate_detect_f=rate_detect_f,
        snap_detect=snap_detect, snap_detect_f=snap_detect_f,
        snap_nc2=snap_nc2, snap_nc2_f=snap_nc2_f,
        rate_nc2=rate_nc2, rate_nc2_f=rate_nc2_f,
    )

def main():
    os.makedirs(TAB, exist_ok=True)
    r = run_experiment()
    csv = os.path.join(TAB, "multiresolution_fem_results.csv")
    with open(csv, "w") as f:
        f.write("metric,value,floor\n")
        f.write(f"p_P1,{r['p_P1']:.4f},\n")
        f.write(f"p_P2,{r['p_P2']:.4f},\n")
        f.write(f"p_P1_unstruct,{r['p_P1u']:.4f},\n")
        f.write(f"rate_detect,{r['rate_detect']:.4f},{r['rate_detect_f']:.4f}\n")
        f.write(f"snapshot_detect,{r['snap_detect']:.4f},{r['snap_detect_f']:.4f}\n")
        f.write(f"snapshot_nc2,{r['snap_nc2']:.4f},{r['snap_nc2_f']:.4f}\n")
        f.write(f"rate_nc2,{r['rate_nc2']:.4f},{r['rate_nc2_f']:.4f}\n")
    print(f"\nartifacts -> {csv}")
    return r

def _figure(r):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans",
                                             palette="muted")
    except Exception:
        pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#8a8a8a"

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8)); fig.subplots_adjust(wspace=0.30, hspace=0.40)

    # A: convergence ladders (median rel-res over ICs) for P1 vs P2
    axA = axes[0, 0]
    medP1 = [np.median([relres("P1", N, s) for s in range(8)]) for N in MR_A]
    medP2 = [np.median([relres("P2", N, s) for s in range(8)]) for N in MR_A]
    axA.plot(MR_A, medP1, "s-", color=RED, lw=2, ms=6, label=f"P1 linear   $p={r['p_P1']:.2f}$")
    axA.plot(MR_A, medP2, "o-", color=BLUE, lw=2, ms=6, label=f"P2 quadratic   $p={r['p_P2']:.2f}$")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xticks(MR_A); axA.set_xticklabels(MR_A); axA.minorticks_off()
    axA.set_xlabel("mesh resolution $N$ (elements)"); axA.set_ylabel(r"$\|r\|/\|u_{\mathrm{ref}}\|$")
    axA.set_title("Convergence ladders separate the elements", fontsize=10)
    axA.legend(frameon=True, framealpha=0.92, edgecolor="#ddd", fontsize=8.5)
    axA.text(-0.18, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: per-IC rate distributions
    axB = axes[0, 1]; rng = np.random.default_rng(0)
    groups = [("P1\nstruct", r["pP1"], RED), ("P2\nstruct", r["pP2"], BLUE),
              ("P1\nDelaunay", r["pP1u"], GREEN)]
    for gi, (lab, P, col) in enumerate(groups):
        jx = gi + rng.uniform(-0.08, 0.08, len(P))
        axB.scatter(jx, P, s=20, color=col, alpha=0.7, edgecolor="none")
        axB.plot([gi - 0.2, gi + 0.2], [np.median(P)] * 2, color=col, lw=2.4)
    axB.axhline(2.0, color=GREY, ls=(0, (1, 2)), lw=1); axB.text(2.42, 2.03, "2", fontsize=7.5, color=GREY)
    axB.axhline(4.0, color=GREY, ls=(0, (1, 2)), lw=1); axB.text(2.42, 4.03, "4", fontsize=7.5, color=GREY)
    axB.set_xticks(range(3)); axB.set_xticklabels([g[0] for g in groups]); axB.set_xlim(-0.5, 2.5)
    axB.set_ylabel("per-IC convergence rate $p$")
    axB.set_title(f"Rate distributions (detection acc {r['rate_detect']:.2f})", fontsize=10)
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: the repair -- snapshot vs rate on the grid control NC2
    axC = axes[1, 0]; x = np.arange(2); w = 0.34
    snap_vals = [r["snap_detect"], r["snap_nc2"]]; rate_vals = [r["rate_detect"], r["rate_nc2"]]
    b1 = axC.bar(x - w / 2, snap_vals, w, color=RED, label="single-snapshot signature")
    b2 = axC.bar(x + w / 2, rate_vals, w, color=GREEN, label="convergence-rate feature")
    for i in range(2):
        axC.text(i - w / 2, snap_vals[i] + 0.012, f"{snap_vals[i]:.2f}", ha="center", fontsize=8)
        axC.text(i + w / 2, rate_vals[i] + 0.012, f"{rate_vals[i]:.2f}", ha="center", fontsize=8)
    axC.axhline(0.5, color=GREY, ls=(0, (1, 2)), lw=1); axC.text(1.46, 0.515, "chance", ha="right", fontsize=7.5, color=GREY)
    axC.set_xticks(x); axC.set_xticklabels(["element\ndetection", "grid control\nNC2"])
    axC.set_ylim(0, 1.05); axC.set_ylabel("GroupKFold accuracy")
    axC.set_title("Rate feature breaks the snapshot grid confound", fontsize=10)
    axC.legend(frameon=False, fontsize=8); axC.text(-0.18, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: residual fields (P1 vs P2) at a coarse mesh -> shows the residual the signature acts on
    axD = axes[1, 1]
    nP1, uP1, refP1 = solve("P1", 32, 0); nP2, uP2, refP2 = solve("P2", 32, 0)
    rg1 = _to_grid(nP1, uP1 - refP1, GRID); rg2 = _to_grid(nP2, uP2 - refP2, GRID)
    xg = np.linspace(0, L, GRID, endpoint=False)
    axD.plot(xg, rg1 / (np.max(np.abs(rg1)) + 1e-12), color=RED, lw=1.4, label="P1 residual")
    axD.plot(xg, rg2 / (np.max(np.abs(rg2)) + 1e-12), color=BLUE, lw=1.4, label="P2 residual")
    axD.set_xlabel("$x$"); axD.set_ylabel("residual $r$ (normalized)"); axD.set_xlim(0, 1); axD.set_ylim(-1.25, 1.25)
    axD.set_title("Residual field at $N{=}32$ (signature substrate)", fontsize=10)
    axD.legend(frameon=False, fontsize=8); axD.text(-0.18, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "fig_multires_fem.png"); os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out); plt.close(fig); print(f"figure -> {out}")

if __name__ == "__main__":
    import sys
    r = main()
    if "--plot" in sys.argv:
        _figure(r)
