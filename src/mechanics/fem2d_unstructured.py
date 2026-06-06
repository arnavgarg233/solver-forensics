#!/usr/bin/env python3
"""
solver-forensics :: 2D UNSTRUCTURED-MESH FEM MECHANICS
================================================================================
Forensic attribution of a discretization choice in a 2D scalar membrane /
elastic-wave solver built ENTIRELY by hand (NO FEM library):

    u_tt = c^2 (u_xx + u_yy)        unit square, homogeneous Dirichlet u=0 on d
                                     the boundary, u_t(.,0) = 0.

The reference is the ANALYTIC modal solution of the Dirichlet membrane,
    u(x,y,t) = sum_{m,n} a_{mn} sin(m pi x) sin(n pi y) cos(c pi sqrt(m^2+n^2) t),
i.e. each initial sine mode oscillates with frequency omega_{mn} = c pi sqrt(m^2+n^2).
Because we seed the IC as a finite sine-mode superposition, the analytic solution is
EXACT (no reference-discretization error to control for).

The spatial discretization is hand-assembled P1 (linear-triangle) FEM on an
UNSTRUCTURED triangular mesh (scipy.spatial.Delaunay on randomized interior points
plus a fixed boundary frame). We assemble the global stiffness K and the consistent
mass M element-by-element from the closed-form P1 element matrices. The time scheme
solves the semidiscrete system  M u_tt + K u = 0.

The REAL discretization choices we attribute (each a genuine modelling decision a
practitioner makes when hand-coding an FEM dynamics solver):

  consistent_CD   consistent mass M + explicit central-difference (leapfrog)
  lumped_CD       row-sum LUMPED mass + central-difference            (the fine, same-order knob)
  newmark_damped  consistent mass + Newmark-beta with numerical damping (gamma=0.7) -> dissipative
  rayleigh_damped consistent mass + central-diff + Rayleigh stiffness damping        -> dissipative

Forensic SIGNATURE (mirrors src/robustness/irregular_mesh.py for irregular meshes):
  INTERPOLATE the solver field AND the analytic reference from the scattered mesh
  nodes to a regular grid with scipy.interpolate.griddata(cubic), then recover the
  unit-normalized least-squares coefficient direction of
      r = u_solver - u_ref  ~  sum_p c_p d_x^p u
  on the 2D derivative library {u_xx, u_yy, u_xxx, u_yyy} (finite differences on the
  regular grid). Direction features are magnitude-invariant.

Attribution = StandardScaler + LogisticRegression, GroupKFold(5) grouped by INITIAL
CONDITION, with a label-PERMUTATION floor on every reported number.

Controls:
  NC1  same scheme, IC + observation noise only           -> must sit ~chance
  NC2  same scheme, MESH-RESOLUTION change (the confound)  -> high => geometry confound

VALIDATION: before trusting any residual we verify the hand-assembled FEM is
convergent -- spatial L2 error vs the analytic eigenmode should fall ~ h^2 (P1).

HONEST: the 1D analogue (src/mechanics/wave_attribution.py) found lumped-vs-consistent
mass is a WEAK, noise-mediated distinction. We report whatever genuinely separates and
state the lumped-vs-consistent result as a measured limit if it is at/near chance.

Self-contained: numpy + scipy + scikit-learn. CPU, a couple of minutes.
Run:  python src/mechanics/fem2d_unstructured.py [--plot]
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import factorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures")

# -------------------- problem constants --------------------
C = 1.0                       # wave speed
T = 0.25                      # final time (sub-period so dispersion/damping has visibly accumulated)
GRID_OBS = 48                 # regular grid the signature is recovered on (after interpolation)
LIB = ("u_xx", "u_yy", "u_xxx", "u_yyy")
SCHEMES = ("consistent_CD", "lumped_CD", "newmark_damped", "rayleigh_damped")
DISSIPATIVE = {"newmark_damped", "rayleigh_damped"}
SIGMA = 0.01                  # field-relative observation noise
N_IC = 120                    # ICs (group unit); large enough that GroupKFold estimates are stable

# ====================================================================
#  UNSTRUCTURED MESH  (Delaunay on randomized interior points + boundary frame)
# ====================================================================
def make_mesh(n_interior, seed):
    """Unstructured triangular mesh of the unit square.
    Boundary nodes are placed on a regular frame (so Dirichlet edges are exact and
    griddata interpolation is well-posed up to the border); interior nodes are
    randomized -> genuinely unstructured Delaunay triangulation."""
    rng = np.random.default_rng(seed)
    nb = max(8, int(round(np.sqrt(n_interior))) + 4)        # boundary points per side
    s = np.linspace(0, 1, nb)
    bottom = np.column_stack([s, np.zeros(nb)])
    top    = np.column_stack([s, np.ones(nb)])
    left   = np.column_stack([np.zeros(nb - 2), s[1:-1]])
    right  = np.column_stack([np.ones(nb - 2),  s[1:-1]])
    bnodes = np.vstack([bottom, top, left, right])
    # interior: jittered grid -> well-spread but unstructured
    m = int(np.ceil(np.sqrt(n_interior)))
    gx = (np.arange(m) + 0.5) / m
    GX, GY = np.meshgrid(gx, gx)
    inodes = np.column_stack([GX.ravel(), GY.ravel()])
    inodes = inodes[:n_interior]
    jit = 0.35 / m
    inodes = inodes + jit * (rng.random(inodes.shape) - 0.5) * 2
    inodes = np.clip(inodes, 1.5 / m, 1 - 1.5 / m)
    pts = np.vstack([bnodes, inodes])
    pts = np.unique(np.round(pts, 9), axis=0)
    tri = Delaunay(pts)
    on_bnd = (np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) |
              np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1))
    return pts, tri.simplices, on_bnd

# ====================================================================
#  HAND-ASSEMBLED P1 FEM  (linear triangles): stiffness K, consistent mass M
# ====================================================================
def assemble(pts, elems):
    """Global P1 stiffness K and consistent mass M (sparse), assembled
    element-by-element from the closed-form linear-triangle matrices."""
    npt = len(pts)
    rows, cols, kdat, mdat = [], [], [], []
    for tr in elems:
        p = pts[tr]                                  # (3,2)
        x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(detJ)
        if area < 1e-14:
            continue
        # gradients of the P1 basis functions (constant per element)
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ
        cc = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ
        Ke = area * (np.outer(b, b) + np.outer(cc, cc))
        Me = (area / 12.0) * np.array([[2., 1., 1.],
                                       [1., 2., 1.],
                                       [1., 1., 2.]])
        for a in range(3):
            for bb in range(3):
                rows.append(tr[a]); cols.append(tr[bb])
                kdat.append(Ke[a, bb]); mdat.append(Me[a, bb])
    K = csr_matrix((kdat, (rows, cols)), shape=(npt, npt))
    M = csr_matrix((mdat, (rows, cols)), shape=(npt, npt))
    return K, M

def lumped_mass(M):
    """Row-sum (HRZ-style for P1 it equals the row sum) lumped diagonal mass."""
    return diags(np.asarray(M.sum(axis=1)).ravel()).tocsr()

# ====================================================================
#  ANALYTIC MODAL REFERENCE (Dirichlet membrane)
# ====================================================================
def modal_ic(seed, n_modes=4, mmax=4):
    """Random superposition of Dirichlet sine eigenmodes -> analytic solution is EXACT."""
    rng = np.random.default_rng(seed)
    modes = []
    seen = set()
    while len(modes) < n_modes:
        m = int(rng.integers(1, mmax + 1)); n = int(rng.integers(1, mmax + 1))
        if (m, n) in seen:
            continue
        seen.add((m, n))
        modes.append((m, n, rng.uniform(-1, 1)))
    return modes

def eval_ic(modes, XY):
    x, y = XY[:, 0], XY[:, 1]
    u = np.zeros(len(XY))
    for m, n, a in modes:
        u += a * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    s = np.std(u)
    return u / s if s > 0 else u, s

def eval_ref(modes, XY, t, norm_s):
    """Exact membrane solution at time t (uses the SAME normalization as the IC)."""
    x, y = XY[:, 0], XY[:, 1]
    u = np.zeros(len(XY))
    for m, n, a in modes:
        om = C * np.pi * np.sqrt(m * m + n * n)
        u += a * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) * np.cos(om * t)
    return u / norm_s if norm_s > 0 else u

# ====================================================================
#  TIME INTEGRATION of the semidiscrete system M u_tt + K u = 0
# ====================================================================
def _dt_for(K, M, free, safety=0.5):
    """Stable explicit dt from the max generalized eigenvalue estimate (power iter)."""
    Kf = K[free][:, free]; Mf = M[free][:, free]
    Minv = diags(1.0 / np.asarray(Mf.sum(axis=1)).ravel())   # lumped approx for the estimate
    A = Minv @ Kf
    v = np.random.default_rng(0).standard_normal(A.shape[0])
    for _ in range(60):
        v = A @ v; nv = np.linalg.norm(v)
        if nv == 0:
            break
        v = v / nv
    lam = float(v @ (A @ v))
    om_max = np.sqrt(max(lam, 1e-12))
    return safety * 2.0 / om_max

def solve_fem(scheme, pts, elems, on_bnd, u0, T=T):
    K, Mc = assemble(pts, elems)
    free = ~on_bnd
    idx = np.where(free)[0]
    Kf = K[idx][:, idx].tocsc()
    Mc_f = Mc[idx][:, idx].tocsc()
    Ml_f = lumped_mass(Mc)[idx][:, idx].tocsc()
    u_free = u0[idx].copy()

    dt = _dt_for(K, Mc, idx)
    ns = max(2, int(np.ceil(T / dt))); dt = T / ns

    if scheme in ("consistent_CD", "lumped_CD", "rayleigh_damped"):
        Mf = Mc_f if scheme != "lumped_CD" else Ml_f
        # damping C = beta_K * K (stiffness-proportional Rayleigh) for the damped variant
        beta_K = 0.02 if scheme == "rayleigh_damped" else 0.0
        Cf = beta_K * Kf
        # central difference: (M/dt^2 + C/2dt) u^{n+1} = (2M/dt^2 - K) u^n - (M/dt^2 - C/2dt) u^{n-1}
        A = (Mf / dt**2 + Cf / (2 * dt)).tocsc()
        solve = factorized(A)
        B = (2.0 / dt**2) * Mf - Kf
        Cm = (Mf / dt**2 - Cf / (2 * dt))
        # first step from Taylor (v0 = 0): u^1 = u^0 + 0.5 dt^2 a0,  M a0 = -K u0
        a0 = factorized(Mf)(-Kf @ u_free)
        u_prev = u_free
        u_cur = u_free + 0.5 * dt**2 * a0
        for _ in range(ns - 1):
            rhs = B @ u_cur - Cm @ u_prev
            u_new = solve(rhs)
            u_prev, u_cur = u_cur, u_new
        u_final = u_cur
    else:  # newmark_damped : Newmark-beta with numerical damping (gamma>0.5)
        Mf = Mc_f
        gamma, beta = 0.7, 0.25 * (0.7 + 0.5)**2
        a0 = factorized(Mf)(-Kf @ u_free)
        u = u_free.copy(); v = np.zeros_like(u); a = a0
        Aeff = (Mf + beta * dt**2 * Kf).tocsc()
        solve = factorized(Aeff)
        for _ in range(ns):
            u_pred = u + dt * v + dt**2 * (0.5 - beta) * a
            a_new = solve(-(Kf @ u_pred))
            v = v + dt * ((1 - gamma) * a + gamma * a_new)
            u = u_pred + beta * dt**2 * a_new
            a = a_new
        u_final = u

    full = np.zeros(len(pts)); full[idx] = u_final
    return full

# ====================================================================
#  SIGNATURE: interpolate scattered fields -> regular grid -> FD coefficient direction
# ====================================================================
_gx = np.linspace(0, 1, GRID_OBS)
_GX, _GY = np.meshgrid(_gx, _gx, indexing="ij")
_GPTS = np.column_stack([_GX.ravel(), _GY.ravel()])
_H = _gx[1] - _gx[0]

def _to_grid(pts, vals):
    g = griddata(pts, vals, _GPTS, method="cubic")
    # cubic can NaN at the convex-hull border on irregular meshes -> backfill nearest
    nan = np.isnan(g)
    if nan.any():
        g[nan] = griddata(pts, vals, _GPTS[nan], method="nearest")
    return g.reshape(GRID_OBS, GRID_OBS)

def _fd_library(U):
    h = _H
    uxx  = (np.roll(U, -1, 0) - 2 * U + np.roll(U, 1, 0)) / h**2
    uyy  = (np.roll(U, -1, 1) - 2 * U + np.roll(U, 1, 1)) / h**2
    uxxx = (np.roll(U, -2, 0) - 2 * np.roll(U, -1, 0) + 2 * np.roll(U, 1, 0) - np.roll(U, 2, 0)) / (2 * h**3)
    uyyy = (np.roll(U, -2, 1) - 2 * np.roll(U, -1, 1) + 2 * np.roll(U, 1, 1) - np.roll(U, 2, 1)) / (2 * h**3)
    # drop a 2-cell border (periodic roll is invalid for a Dirichlet box)
    sl = slice(2, GRID_OBS - 2)
    return {"u_xx": uxx[sl, sl], "u_yy": uyy[sl, sl],
            "u_xxx": uxxx[sl, sl], "u_yyy": uyyy[sl, sl]}, sl

def signature(pts, u_solver, u_ref):
    Us = _to_grid(pts, u_solver)
    Ur = _to_grid(pts, u_ref)
    R = (Us - Ur)
    D, sl = _fd_library(Us)
    A = np.column_stack([D[name].ravel() for name in LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c

def feats(C):
    C = np.asarray(C)
    unit = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(np.nan_to_num(C[:, 2] / C[:, 0]), -10, 10)   # c_xxx / c_xx
    return np.nan_to_num(np.hstack([unit, ratio[:, None]]))

# ====================================================================
#  classification helpers
# ====================================================================
def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

def cv_acc(X, y, g):
    return cross_val_score(_clf(), X, y, groups=g, cv=GroupKFold(5)).mean()

def perm_floor(X, y, g, seed, reps=30):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(_clf(), X, r.permutation(y), groups=g,
                                            cv=GroupKFold(5)).mean() for _ in range(reps)]))

# ====================================================================
#  VALIDATION: FEM spatial convergence on a single eigenmode
# ====================================================================
def validate_convergence():
    """L2 error of the FEM dynamics vs the analytic mode (1,1) at T; expect ~ h^2 (P1)."""
    modes = [(1, 1, 1.0)]
    ns = [40, 80, 160, 320]   # interior node counts -> decreasing h
    errs, hs = [], []
    for ni in ns:
        pts, elems, on_bnd = make_mesh(ni, seed=12345)
        u0, s = eval_ic(modes, pts)
        uf = solve_fem("consistent_CD", pts, elems, on_bnd, u0)
        uref = eval_ref(modes, pts, T, s)
        # mesh-area-weighted L2 (use nodal error, normalize by ref norm)
        err = np.linalg.norm(uf - uref) / (np.linalg.norm(uref) + 1e-12)
        # representative mesh size h ~ 1/sqrt(#interior)
        h = 1.0 / np.sqrt(ni)
        errs.append(err); hs.append(h)
    hs = np.array(hs); errs = np.array(errs)
    rate = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    return ns, hs, errs, rate

# ====================================================================
#  build signatures for one scheme over all ICs on a fixed mesh
# ====================================================================
def build_sigs(scheme, pts, elems, on_bnd, ic_modes, sigma, seed):
    C = []
    rng = np.random.default_rng(seed)
    for modes in ic_modes:
        u0, s = eval_ic(modes, pts)
        uf = solve_fem(scheme, pts, elems, on_bnd, u0)
        uref = eval_ref(modes, pts, T, s)
        if sigma > 0:
            uf = uf + sigma * np.sqrt(np.mean(uf**2)) * rng.standard_normal(uf.shape)
        C.append(signature(pts, uf, uref))
    return np.array(C)

# ====================================================================
#  MAIN
# ====================================================================
def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    print("=" * 78)
    print("2D UNSTRUCTURED-MESH FEM MECHANICS  (hand-assembled P1, analytic modal reference)")
    print("=" * 78)

    # ---- (0) convergence validation ----
    ns, hs, errs, rate = validate_convergence()
    print("\n[VALIDATION] hand-assembled P1 FEM spatial convergence (mode (1,1), T=%.2f):" % T)
    for ni, h, e in zip(ns, hs, errs):
        print(f"   interior={ni:>4d}  h~{h:.4f}  rel-L2-err={e:.3e}")
    print(f"   fitted rate  err ~ h^{rate:.2f}  (P1 expects ~2; >~1.3 => convergent & trustworthy)")
    converged = rate > 1.3

    # ---- meshes ----
    N_INT = 200                          # primary mesh interior-node budget
    pts, elems, on_bnd = make_mesh(N_INT, seed=2026)
    print(f"\nprimary mesh: {len(pts)} nodes ({on_bnd.sum()} boundary), {len(elems)} triangles")

    # ---- ICs (shared across schemes; group = IC index) ----
    ic_modes = [modal_ic(1000 + i) for i in range(N_IC)]
    ic = np.arange(N_IC)

    print("\n[solving FEM for each scheme over %d ICs] ..." % N_IC)
    F = {}
    for j, sc in enumerate(SCHEMES):
        F[sc] = build_sigs(sc, pts, elems, on_bnd, ic_modes, SIGMA, seed=100 + 1000 * j)
        print(f"   {sc:<16} signatures done")

    results = {}

    # ---- ID: 4-way scheme identification ----
    X = np.vstack([F[s] for s in SCHEMES]); X = feats(X)
    y = np.concatenate([np.full(N_IC, i) for i in range(len(SCHEMES))])
    g = np.concatenate([ic] * len(SCHEMES))
    id_acc = cv_acc(X, y, g); id_fl = perm_floor(X, y, g, 7)
    results["id_4way"] = (id_acc, id_fl)
    print(f"\nID  4-way scheme identification:        acc={id_acc:.3f}  perm-floor={id_fl:.3f}  (chance {1/len(SCHEMES):.2f})")

    # ---- dissipation: damped (newmark+rayleigh) vs non-dissipative (the two CD) ----
    Xd = feats(np.vstack([F["newmark_damped"], F["rayleigh_damped"], F["consistent_CD"], F["lumped_CD"]]))
    yd = np.r_[np.ones(2 * N_IC), np.zeros(2 * N_IC)]
    gd = np.r_[ic, ic, ic, ic]
    diss_acc = cv_acc(Xd, yd, gd); diss_fl = perm_floor(Xd, yd, gd, 11)
    results["dissipation"] = (diss_acc, diss_fl)
    print(f"diss dissipative vs non-dissipative:    acc={diss_acc:.3f}  perm-floor={diss_fl:.3f}  (balanced 2:2)")

    # ---- integrator: Newmark vs central-difference (both consistent mass) ----
    Xi = feats(np.vstack([F["newmark_damped"], F["consistent_CD"]]))
    yi = np.r_[np.ones(N_IC), np.zeros(N_IC)]; gi = np.r_[ic, ic]
    intg_acc = cv_acc(Xi, yi, gi); intg_fl = perm_floor(Xi, yi, gi, 13)
    results["integrator"] = (intg_acc, intg_fl)
    print(f"intg Newmark vs central-diff:           acc={intg_acc:.3f}  perm-floor={intg_fl:.3f}")

    # ---- mass: lumped vs consistent (both central-diff) -- the FINE same-order knob ----
    Xm = feats(np.vstack([F["lumped_CD"], F["consistent_CD"]]))
    ym = np.r_[np.ones(N_IC), np.zeros(N_IC)]; gm = np.r_[ic, ic]
    mass_acc = cv_acc(Xm, ym, gm); mass_fl = perm_floor(Xm, ym, gm, 17)
    results["mass"] = (mass_acc, mass_fl)
    # mass at sigma=0 (clean) to see if the distinction is deterministic or noise-mediated
    F0l = build_sigs("lumped_CD", pts, elems, on_bnd, ic_modes, 0.0, seed=900)
    F0c = build_sigs("consistent_CD", pts, elems, on_bnd, ic_modes, 0.0, seed=901)
    Xm0 = feats(np.vstack([F0l, F0c]))
    mass0_acc = cv_acc(Xm0, ym, gm)
    results["mass_sigma0"] = (mass0_acc, None)
    print(f"mass lumped vs consistent (sigma={SIGMA}):  acc={mass_acc:.3f}  perm-floor={mass_fl:.3f}")
    print(f"     lumped vs consistent (sigma=0):    acc={mass0_acc:.3f}  (clean; reveals if deterministic)")

    # ---- NC1: same scheme, IC + noise only.  Split ICs into two halves labelled 0/1.
    # With 25 vs 25 ICs and 5 grouped folds a SINGLE split is high-variance, so we
    # report the MEDIAN over 20 random IC-halvings (the honest estimate of the control).
    Fnc1 = build_sigs("consistent_CD", pts, elems, on_bnd, ic_modes, SIGMA, seed=5000)
    half = N_IC // 2
    rng_nc1 = np.random.default_rng(19)
    nc1_runs = []
    for _ in range(20):
        perm = rng_nc1.permutation(N_IC); A, B = perm[:half], perm[half:]
        Xn1 = feats(np.vstack([Fnc1[A], Fnc1[B]]))
        yn1 = np.r_[np.zeros(half), np.ones(N_IC - half)]
        gn1 = np.r_[ic[A], ic[B]]
        nc1_runs.append(cv_acc(Xn1, yn1, gn1))
    nc1_acc = float(np.median(nc1_runs)); nc1_fl = 0.5
    results["nc1"] = (nc1_acc, nc1_fl)
    print(f"\nNC1 IC+noise (same scheme):             acc={nc1_acc:.3f} "
          f"(median over 20 IC-halvings; range {min(nc1_runs):.2f}-{max(nc1_runs):.2f})  (chance ~0.50)")

    # ---- NC2: same scheme, MESH-RESOLUTION change (the confound) ----
    ptsA, elemsA, bndA = make_mesh(120, seed=3001)   # coarser unstructured mesh
    ptsB, elemsB, bndB = make_mesh(320, seed=3002)   # finer  unstructured mesh
    FncA = build_sigs("consistent_CD", ptsA, elemsA, bndA, ic_modes, SIGMA, seed=7000)
    FncB = build_sigs("consistent_CD", ptsB, elemsB, bndB, ic_modes, SIGMA, seed=8000)
    Xn2 = feats(np.vstack([FncA, FncB]))
    yn2 = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gn2 = np.r_[ic, ic]
    nc2_acc = cv_acc(Xn2, yn2, gn2); nc2_fl = perm_floor(Xn2, yn2, gn2, 23)
    results["nc2"] = (nc2_acc, nc2_fl)
    print(f"NC2 mesh-resolution change (120 vs 320): acc={nc2_acc:.3f}  perm-floor={nc2_fl:.3f}  (high=geometry confound)")

    # ---- robustness: does dissipation channel hold across the NC2 mesh change? ----
    # (re-recover on coarse mesh ptsA to confirm the dissipation signature is not mesh-specific)
    Fa_n = build_sigs("newmark_damped", ptsA, elemsA, bndA, ic_modes, SIGMA, seed=4100)
    Fa_c = build_sigs("consistent_CD", ptsA, elemsA, bndA, ic_modes, SIGMA, seed=4101)
    Xtr = feats(np.vstack([F["newmark_damped"], F["consistent_CD"]]))   # train on primary mesh
    ytr = np.r_[np.ones(N_IC), np.zeros(N_IC)]
    Xte = feats(np.vstack([Fa_n, Fa_c])); yte = np.r_[np.ones(N_IC), np.zeros(N_IC)]
    clf = _clf().fit(Xtr, ytr)
    transfer_acc = clf.score(Xte, yte)
    results["diss_transfer"] = (transfer_acc, None)
    print(f"\ntransfer: dissipation classifier trained on primary mesh, tested on coarse mesh: acc={transfer_acc:.3f}")

    # ---- CSV ----
    csv = os.path.join(TAB, "fem2d_unstructured_results.csv")
    with open(csv, "w") as f:
        f.write("task,accuracy,perm_floor\n")
        for key in ("id_4way", "dissipation", "integrator", "mass", "mass_sigma0",
                    "nc1", "nc2", "diss_transfer"):
            a, fl = results[key]
            f.write(f"{key},{a:.4f},{'' if fl is None else f'{fl:.4f}'}\n")
        f.write(f"convergence_rate,{rate:.4f},\n")
    print(f"\nresults -> {csv}")

    summary = dict(id_acc=id_acc, id_fl=id_fl, diss=diss_acc, diss_fl=diss_fl,
                   intg=intg_acc, intg_fl=intg_fl, mass=mass_acc, mass_fl=mass_fl,
                   mass0=mass0_acc, nc1=nc1_acc, nc1_fl=nc1_fl, nc2=nc2_acc, nc2_fl=nc2_fl,
                   transfer=transfer_acc, rate=rate, converged=converged,
                   ns=ns, hs=hs, errs=errs, pts=pts, elems=elems,
                   F=F, ic_modes=ic_modes, on_bnd=on_bnd)
    return summary

def _figure(r):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    except Exception:
        pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0", "#dd8452"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8)); fig.subplots_adjust(wspace=0.28, hspace=0.34)

    # A: the unstructured mesh
    axA = axes[0, 0]
    pts, elems = r["pts"], r["elems"]
    axA.triplot(pts[:, 0], pts[:, 1], elems, color="#7aa6c2", lw=0.5)
    bnd = r["on_bnd"]
    axA.plot(pts[bnd, 0], pts[bnd, 1], "o", color=RED, ms=2.5, label="Dirichlet boundary")
    axA.plot(pts[~bnd, 0], pts[~bnd, 1], "o", color=BLUE, ms=1.8)
    axA.set_aspect("equal"); axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.02, 1.02)
    axA.set_title(f"Hand-assembled P1 mesh ({len(pts)} nodes, {len(elems)} tris)", fontsize=9.5)
    axA.set_xlabel("x"); axA.set_ylabel("y"); axA.legend(frameon=False, fontsize=7.5, loc="upper right")
    axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: convergence
    axB = axes[0, 1]
    hs, errs = r["hs"], r["errs"]
    axB.loglog(hs, errs, "o-", color=BLUE, lw=2, ms=6, label=fr"FEM   $p={r['rate']:.2f}$")
    ref = errs[0] * (hs / hs[0]) ** 2
    axB.loglog(hs, ref, "--", color=GREY, lw=1.4, label=r"$h^2$ reference")
    axB.set_xlabel("mesh size $h\\sim 1/\\sqrt{N_{int}}$"); axB.set_ylabel(r"relative $L_2$ error vs analytic")
    axB.set_title("Spatial convergence (validation)", fontsize=9.5)
    axB.legend(frameon=False, fontsize=8); axB.grid(True, which="both", color="#e6e6e6", lw=0.7)
    axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: residual field for each scheme (one IC, interpolated to grid)
    axC = axes[1, 0]
    modes = r["ic_modes"][0]
    u0, s = eval_ic(modes, pts); uref = eval_ref(modes, pts, T, s)
    rep = {}
    for sc in SCHEMES:
        uf = solve_fem(sc, pts, elems, r["on_bnd"], u0)
        rep[sc] = np.linalg.norm(_to_grid(pts, uf - uref))
    cols = {"consistent_CD": BLUE, "lumped_CD": GREEN, "newmark_damped": RED, "rayleigh_damped": ORANGE}
    names = list(SCHEMES)
    axC.bar(range(len(names)), [rep[s] for s in names], color=[cols[s] for s in names], width=0.62)
    axC.set_xticks(range(len(names)))
    axC.set_xticklabels(["consist.\nCD", "lumped\nCD", "Newmark\ndamped", "Rayleigh\ndamped"], fontsize=8)
    axC.set_ylabel(r"$\|r\|_2$ on grid"); axC.set_title("Residual magnitude by scheme (one IC)", fontsize=9.5)
    axC.grid(axis="y", color="#e6e6e6", lw=0.7); axC.set_axisbelow(True)
    axC.text(-0.13, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: attribution accuracies with per-task permutation floors
    axD = axes[1, 1]
    keys = [("id_4way", "4-way\nID"), ("dissipation", "dissip."), ("integrator", "integr."),
            ("mass", "mass"), ("nc1", "NC1"), ("nc2", "NC2\n(mesh)")]
    vals = [r["id_acc"], r["diss"], r["intg"], r["mass"], r["nc1"], r["nc2"]]
    fls = [r["id_fl"], r["diss_fl"], r["intg_fl"], r["mass_fl"], r["nc1_fl"], r["nc2_fl"]]
    cols2 = [BLUE, GREEN, PURP, ORANGE, GREY, RED]
    axD.bar(range(len(keys)), vals, color=cols2, width=0.66)
    for i, fl in enumerate(fls):
        axD.plot([i - 0.34, i + 0.34], [fl, fl], color="#333", ls=(0, (2, 1.5)), lw=1.4, zorder=6)
    for i, v in enumerate(vals):
        axD.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    axD.text(3, r["mass"] - 0.05, f"σ=0:\n{r['mass0']:.2f}", ha="center", va="top", fontsize=6.6, color="#555")
    axD.set_xticks(range(len(keys))); axD.set_xticklabels([k[1] for k in keys], fontsize=8)
    axD.set_ylim(0, 1.05); axD.set_ylabel("GroupKFold accuracy")
    axD.set_title("Attribution (dashed = permutation floor)", fontsize=9.5)
    axD.grid(axis="y", color="#e6e6e6", lw=0.7); axD.set_axisbelow(True)
    axD.text(-0.13, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "fig_fem2d_unstructured.png")
    fig.savefig(out); plt.close(fig)
    print(f"figure -> {out}")
    return out

if __name__ == "__main__":
    import sys
    np.seterr(all="ignore")
    r = main()
    if "--plot" in sys.argv:
        _figure(r)
    print("\ndone.")
