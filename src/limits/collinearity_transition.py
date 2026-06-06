#!/usr/bin/env python3
"""
solver-forensics :: COLLINEARITY TRANSITION CURVE  (a CONFESSED limit, CHARACTERIZED)
=====================================================================================
The stabilization audit (src/audit/stabilization_audit.py) reports one HONEST at-chance
contrast: SUPG vs. matched artificial-viscosity. When the artificial-viscosity coefficient
is tuned to MATCH the SUPG added streamline diffusion (nu_AV = tau|a|^2) and the flow is
axis-aligned, the residual signatures are near-collinear and attribution sits on the
permutation floor. The audit states this as a flat "at-chance" footnote.

This script does the on-brand thing for the 'measured-envelope-as-deliverable' philosophy:
instead of reporting "at chance", it MAPS WHY the contrast dies and EXACTLY WHERE it crosses
from COLLINEAR to SEPARABLE, by sweeping the geometry knob (the flow-skew angle theta) and
the magnitude knob (the mismatch ratio rho).

------------------------------------------------------------------------------------- WHY
The two added stabilization operators differ in TWO physical ways:

  (1) ANISOTROPY (carried by the flow-skew angle theta) -- THIS is what separates them.
      Streamline upwind diffusion adds a streamline-aligned RANK-ONE tensor that ROTATES
      WITH THE FLOW,
          D_SD   = tau * a a^T            (tau|a|^2 along the flow, ZERO cross-stream),
      whereas artificial viscosity adds a FIXED tensor that does NOT track the flow. The
      sharpest, mechanism-isolating contrast is against a practitioner artificial viscosity
      that was tuned/pinned to the MESH x-axis (a rank-one nu_AV * e_x e_x^T -- e.g. an
      'x-direction upwind' added on a grid laid out along x). At theta=0 the streamline
      tensor and the x-pinned tensor are the IDENTICAL operator (verified bit-for-bit), so
      the two solver fields, residuals and signatures coincide -> COLLINEAR, attribution at
      chance. As the flow SKEWS off the mesh x-axis, the streamline tensor rotates away from
      the pinned tensor and the modified-equation footprints separate -> SEPARABLE.

  (2) MAGNITUDE MISMATCH (the ratio rho = nu_AV/(tau|a|^2)).
      Swept axis-aligned. The unit-normalized coefficient DIRECTION is far less sensitive to
      a scalar magnitude mismatch than to the anisotropy: rho moves the signature only along
      the SAME (axis-pinned) direction, so the cosine stays ~1 over a wide band. This is
      itself a finding -- the often-quoted "|a|h/2 vs tau|a|^2" *magnitude* argument is NOT
      what makes the schemes attributable from a direction signature; GEOMETRY (anisotropy) is.

A SEPARATE, honestly-reported finding (overlay, not the headline): the FULL CONSISTENT SUPG
(Petrov-Galerkin: test space w + tau a.grad w, which also REWEIGHTS THE LOAD by tau a.grad w
times the residual) carries an EXTRA, consistency-specific footprint that is attributable
even when the diffusion tensors coincide (theta=0). So consistent-SUPG vs the matched
artificial viscosity is SEPARABLE across the whole angle sweep -- it never reaches chance in
this manufactured-solution setting. We therefore DECOMPOSE the SUPG-vs-AV question into its
two independent channels: a CONSISTENCY channel (attributable at any angle) and an ANISOTROPY
channel (the transition curve below). The at-chance regime exists only when BOTH are removed,
i.e. inconsistent streamline diffusion vs the matched/pinned artificial viscosity at theta=0.

------------------------------------------------------------------------------------ HOW
Substrate: 2D steady skew advection-diffusion (the CMAME-native stabilization testbed),
    a . grad u  -  D_phys lap u  =  f ,    on (0,1)^2 ,  u = 0 on the boundary (Dirichlet),
with a = |a| (cos theta, sin theta), high cell-Peclet (advection-dominated) so stabilization
matters. REFERENCE is a MANUFACTURED exact solution u_star (smooth sin x sin y product modes
that vanish on the boundary): the forcing is f = a.grad u_star - D_phys lap u_star evaluated
analytically, so the BVP's exact solution IS u_star (NO reference-discretization error -- a
GENUINE reference). Solver = hand-assembled P1 FEM on a structured-triangulated mesh
(scipy.spatial.Delaunay on a regular node set; NO FEM library). Each scheme adds its element
diffusion tensor to the Galerkin bilinear form; consistent-SUPG additionally reweights the load.

  schemes:
    SD          inconsistent streamline diffusion: D = tau a a^T, load NOT reweighted
    AV          artificial viscosity PINNED to the mesh x-axis: D = rho*tau|a|^2 e_x e_x^T
                (rank-one; == SD exactly at theta=0, rho=1 -> the collinear endpoint)
    SUPG        full CONSISTENT SUPG: D = tau a a^T AND load reweighted by tau a.grad w * f
                (the consistency overlay; separable even at theta=0)

SIGNATURE: r = u_solver - u_ref interpolated to a regular grid (griddata), then the
unit-normalized least-squares coefficient DIRECTION of r ~ sum c_p L_p[u] on the 2D
high-derivative library {u_xx, u_yy, u_xy, u_xxxx, u_yyyy} (FD on the grid). Direction is
magnitude-invariant -- exactly the project signature.

ATTRIBUTION: StandardScaler+LogisticRegression, GroupKFold(5) grouped by INITIAL CONDITION
(here the manufactured-solution realization), label-PERMUTATION floor on EVERY point.

CONTROLS (reported at the collinear endpoint):
  NC1  same scheme (SD), IC + observation noise only             -> must sit ~chance
  NC2  same scheme (SD), mesh-resolution change (the confound)    -> reported as diagnostic

VALIDATION (printed, must pass before residuals are trusted):
  * MMS convergence: SD rel-L2 error vs manufactured u_star falls under mesh refinement.
  * advection-dominated regime: cell Peclet Pe_h >> 1, added stab diffusion >> physical.
  * collinear endpoint is REAL: at theta=0,rho=1 the SD and AV solver fields are identical
    bit-for-bit (max|u_SD - u_AV| ~ 0), so the at-chance start is a true degeneracy.

DECISION RULE: report the swept-parameter value where attribution CROSSES from COLLINEAR
(~floor) to SEPARABLE (> floor + 0.2). A transition curve that shows WHERE and WHY
separability is born converts the confessed at-chance footnote into a CHARACTERIZED boundary.

Self-contained: numpy + scipy + sklearn, CPU (~3-5 min). Guarded by __main__.
  metrics -> results/tables/collinearity_transition.csv
  figure  -> figures/fig_collinearity_transition.png
Run:  python src/limits/collinearity_transition.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

# ----------------------------- problem constants -----------------------------
AMAG    = 1.0          # advection speed magnitude |a|
D_PHYS  = 2.0e-3       # physical diffusivity (small -> advection-dominated)
N_SIDE  = 24           # structured node grid is N_SIDE x N_SIDE -> h = 1/(N_SIDE-1)
N_OBS   = 40           # regular observation grid the residual signature is read on
N_IC    = 40           # manufactured-solution realizations (the GroupKFold group key)
SIGMA   = 0.01         # field-relative observation noise
SEP_THR = 0.20         # "separable" := accuracy gap over the permutation floor exceeds this
# 2D high-derivative library: anisotropy lives in {u_xx, u_yy, u_xy}; high orders {u_xxxx,u_yyyy}
LIB     = ("u_xx", "u_yy", "u_xy", "u_xxxx", "u_yyyy")

def mesh_h(n_side):
    return 1.0 / (n_side - 1)

def supg_tau(h):
    """Classic optimal SUPG tau for the streamwise cell-Peclet (doubly-asymptotic coth form):
    tau = h/(2|a|) (coth(Pe) - 1/Pe), Pe = |a| h /(2 D_phys)."""
    pe = AMAG * h / (2.0 * D_PHYS)
    return (h / (2.0 * AMAG)) * (1.0 / np.tanh(pe) - 1.0 / pe), pe

# ============================================================ structured-triangulated mesh
def make_mesh(n_side):
    """Regular n_side x n_side node lattice on (0,1)^2, triangulated by Delaunay (NO FEM lib).
    Boundary nodes flagged for strong homogeneous Dirichlet BCs (u_star vanishes there)."""
    s = np.linspace(0.0, 1.0, n_side)
    GX, GY = np.meshgrid(s, s)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    tri = Delaunay(pts)
    on_bnd = (np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) |
              np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1))
    return pts, tri.simplices, on_bnd

# ============================================================ manufactured solution (MMS reference)
def mms_modes(rng, n_modes=3, kmax=3):
    """A manufactured-solution realization: u_star = sum a_mn sin(m pi x) sin(n pi y) (vanishes on
    the boundary -> homogeneous Dirichlet, clean BCs). The GROUP key (the 'initial condition')."""
    modes = []; seen = set()
    while len(modes) < n_modes:
        m = int(rng.integers(1, kmax + 1)); n = int(rng.integers(1, kmax + 1))
        if (m, n) in seen: continue
        seen.add((m, n)); modes.append((m, n, float(rng.uniform(-1, 1))))
    return modes

def u_star(modes, XY):
    x, y = XY[:, 0], XY[:, 1]; u = np.zeros(len(XY))
    for m, n, a in modes:
        u += a * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return u

def f_source(modes, XY, ax, ay):
    """Forcing that makes u_star the EXACT BVP solution:  f = a.grad u_star - D_phys lap u_star.
    grad and laplacian of each sin-sin mode are analytic."""
    x, y = XY[:, 0], XY[:, 1]; f = np.zeros(len(XY))
    for m, n, a in modes:
        km, kn = m * np.pi, n * np.pi
        sx, cx = np.sin(km * x), np.cos(km * x)
        sy, cy = np.sin(kn * y), np.cos(kn * y)
        ux = a * km * cx * sy
        uy = a * kn * sx * cy
        lap = -a * (km * km + kn * kn) * sx * sy
        f += ax * ux + ay * uy - D_PHYS * lap
    return f

# ============================================================ hand-assembled P1 FEM (skew adv-diff + stabilization)
def assemble(modes, pts, elems, on_bnd, scheme, theta, rho, h):
    """P1 FEM for  a.grad u - D_phys lap u = f  with a = |a|(cos theta, sin theta).
    Stabilization (advection-dominated) added as an element diffusion TENSOR:
        scheme='SD'   : D_stab = tau a a^T (streamline, rotates with flow); load NOT reweighted (inconsistent)
        scheme='AV'   : D_stab = rho*tau|a|^2 e_x e_x^T (rank-1, PINNED to mesh x-axis); load NOT reweighted
        scheme='SUPG' : D_stab = tau a a^T  AND  load reweighted by tau (a.grad phi_i) f (consistent Petrov-Galerkin)
    rho = nu_AV / (tau|a|^2) is the magnitude-mismatch ratio (rho=1 == matched magnitude).
    At theta=0, rho=1 the SD and AV diffusion tensors are the IDENTICAL operator tau|a|^2 e_x e_x^T
    -> the two solver fields coincide bit-for-bit (the collinear endpoint).
    Linear-triangle element matrices (constant gradients per element):
        advection  C_e[i,j] = (area/3) (a.grad phi_j)
        diffusion  S_e[i,j] = area * grad phi_i^T D grad phi_j
        load       F_e[i]   = (area/3) f(centroid)
    SUPG load:  + tau (a.grad phi_i) f(centroid) area  (residual-weighted source)."""
    ax, ay = AMAG * np.cos(theta), AMAG * np.sin(theta)
    a_vec = np.array([ax, ay])
    tau, _ = supg_tau(h)
    if scheme in ("SD", "SUPG"):
        D_stab = tau * np.outer(a_vec, a_vec)              # streamline tensor, rotates with the flow
    elif scheme == "AV":
        ex = np.array([1.0, 0.0])                          # PINNED to mesh x-axis
        D_stab = (rho * tau * AMAG * AMAG) * np.outer(ex, ex)
    else:
        D_stab = np.zeros((2, 2))                          # Galerkin (unstabilized; validation only)
    D_tot = D_PHYS * np.eye(2) + D_stab

    npt = len(pts)
    K = np.zeros((npt, npt)); F = np.zeros(npt)
    for tr in elems:
        p = pts[tr]
        x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(detJ)
        if area < 1e-14: continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ   # d phi / dx
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ   # d phi / dy
        grad = np.column_stack([b, c])                      # (3,2): rows = grad phi_i
        adv_j = grad @ a_vec                                # a.grad phi_j  (constant per elem)
        Ce = (area / 3.0) * np.tile(adv_j, (3, 1))          # C_e[i,j] = area/3 * adv_j
        Se = area * (grad @ D_tot @ grad.T)
        Ke = Ce + Se
        centroid = p.mean(0)[None, :]
        fc = f_source(modes, centroid, ax, ay)[0]
        Fe = (area / 3.0) * fc * np.ones(3)
        if scheme == "SUPG":                                # consistent residual load
            Fe = Fe + tau * adv_j * fc * area
        K[np.ix_(tr, tr)] += Ke
        F[tr] += Fe
    for nd in np.where(on_bnd)[0]:                          # strong homogeneous Dirichlet
        K[nd, :] = 0.0; K[nd, nd] = 1.0; F[nd] = 0.0
    return np.linalg.solve(K, F)

# ============================================================ signature (interp -> grid -> FD direction)
_og = np.linspace(0.0, 1.0, N_OBS)
_OGX, _OGY = np.meshgrid(_og, _og, indexing="ij")
_OGPTS = np.column_stack([_OGX.ravel(), _OGY.ravel()])
_OH = _og[1] - _og[0]

def _to_grid(pts, vals):
    g = griddata(pts, vals, _OGPTS, method="cubic")
    nan = np.isnan(g)
    if nan.any():
        g[nan] = griddata(pts, vals, _OGPTS[nan], method="nearest")
    return g.reshape(N_OBS, N_OBS)

def _fd_library(U):
    h = _OH
    uxx   = (np.roll(U, -1, 0) - 2 * U + np.roll(U, 1, 0)) / h**2
    uyy   = (np.roll(U, -1, 1) - 2 * U + np.roll(U, 1, 1)) / h**2
    uxy   = (np.roll(np.roll(U, -1, 0), -1, 1) - np.roll(np.roll(U, -1, 0), 1, 1)
             - np.roll(np.roll(U, 1, 0), -1, 1) + np.roll(np.roll(U, 1, 0), 1, 1)) / (4 * h**2)
    uxxxx = (np.roll(U, -2, 0) - 4 * np.roll(U, -1, 0) + 6 * U - 4 * np.roll(U, 1, 0) + np.roll(U, 2, 0)) / h**4
    uyyyy = (np.roll(U, -2, 1) - 4 * np.roll(U, -1, 1) + 6 * U - 4 * np.roll(U, 1, 1) + np.roll(U, 2, 1)) / h**4
    sl = slice(2, N_OBS - 2)
    return {"u_xx": uxx[sl, sl], "u_yy": uyy[sl, sl], "u_xy": uxy[sl, sl],
            "u_xxxx": uxxxx[sl, sl], "u_yyyy": uyyyy[sl, sl]}, sl

def signature(pts, u_solver, u_ref):
    Us = _to_grid(pts, u_solver); Ur = _to_grid(pts, u_ref)
    R = Us - Ur
    D, sl = _fd_library(Us)
    Amat = np.column_stack([D[name].ravel() for name in LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c

# ============================================================ build signature clouds for one (scheme,theta,rho)
def build_sigs(scheme, modes_list, pts, elems, on_bnd, theta, rho, h, sigma, seed):
    out = []
    rng = np.random.default_rng(seed)
    for modes in modes_list:
        u_h = assemble(modes, pts, elems, on_bnd, scheme, theta, rho, h)
        u_ref = u_star(modes, pts)
        if sigma > 0:
            u_h = u_h + sigma * np.sqrt(np.mean(u_h**2)) * rng.standard_normal(u_h.shape)
        out.append(signature(pts, u_h, u_ref))
    return np.array(out)

# ============================================================ attribution machinery
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def cv_acc(F, y, g):
    return float(cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean())
def perm_floor(F, y, g, seed, reps=30):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean()
                            for _ in range(reps)]))

def pair_acc_floor(Fa, Fb, ic, seed):
    X = np.vstack([Fa, Fb]); y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]; g = np.r_[ic, ic]
    return cv_acc(X, y, g), perm_floor(X, y, g, seed)

def mean_dir(F):
    m = F.mean(0); n = np.linalg.norm(m)
    return m / n if n > 0 else m

# ============================================================ validation
def validate():
    """(1) MMS convergence of SD vs the manufactured u_star under mesh refinement.
       (2) advection-dominated regime: cell Peclet and added-stab/physical-diffusion ratio.
       (3) collinear endpoint is REAL: at theta=0,rho=1 the SD and AV solver fields coincide."""
    rng = np.random.default_rng(123)
    modes = mms_modes(rng, n_modes=3, kmax=3)
    rep = {}
    errs = {}
    for n_side in (14, 20, 28, 40, 56):
        h = mesh_h(n_side)
        ptsc, elemsc, bndc = make_mesh(n_side)
        u_h = assemble(modes, ptsc, elemsc, bndc, "SD", theta=0.0, rho=1.0, h=h)
        u_ex = u_star(modes, ptsc)
        errs[n_side] = float(np.linalg.norm(u_h - u_ex) / (np.linalg.norm(u_ex) + 1e-12))
    rep["conv"] = errs
    rep["conv_ok"] = errs[56] < errs[14] and errs[56] < 0.05
    h0 = mesh_h(N_SIDE)
    tau0, pe0 = supg_tau(h0)
    rep["pe_h"] = pe0
    rep["stab_over_phys"] = (tau0 * AMAG * AMAG) / D_PHYS
    rep["adv_dominated"] = pe0 > 1.0 and rep["stab_over_phys"] > 1.0
    # collinear endpoint: SD and AV identical at theta=0, rho=1
    ptsp, elemsp, bndp = make_mesh(N_SIDE)
    uS = assemble(modes, ptsp, elemsp, bndp, "SD", 0.0, 1.0, h0)
    uA = assemble(modes, ptsp, elemsp, bndp, "AV", 0.0, 1.0, h0)
    rep["endpoint_identical"] = float(np.max(np.abs(uS - uA)))
    rep["endpoint_ok"] = rep["endpoint_identical"] < 1e-10
    return rep

# ============================================================ MAIN
def main():
    print("=" * 92)
    print("COLLINEARITY TRANSITION CURVE  (streamline diffusion vs artificial viscosity: WHERE it dies)")
    print("=" * 92)
    h = mesh_h(N_SIDE)
    tau0, pe0 = supg_tau(h)
    print(f"2D skew advection-diffusion  a.grad u - D_phys lap u = f  on (0,1)^2 (MMS reference)")
    print(f"|a|={AMAG}, D_phys={D_PHYS}, mesh {N_SIDE}x{N_SIDE} (h={h:.4f}), obs grid {N_OBS}x{N_OBS}, "
          f"{N_IC} MMS realizations, sigma={SIGMA}")
    print(f"cell Peclet Pe_h = |a|h/(2 D_phys) = {pe0:.2f}   SUPG tau = {tau0:.5f}   "
          f"added stab diffusion tau|a|^2 = {tau0*AMAG*AMAG:.5f} ({tau0*AMAG*AMAG/D_PHYS:.1f}x physical)\n")

    pts, elems, on_bnd = make_mesh(N_SIDE)
    modes_list = [mms_modes(np.random.default_rng(2000 + i), n_modes=3, kmax=3) for i in range(N_IC)]
    ic = np.arange(N_IC)
    print(f"primary mesh: {len(pts)} nodes ({on_bnd.sum()} boundary), {len(elems)} triangles\n")

    # ---- validation ----
    vr = validate()
    print("[VALIDATION] MMS convergence of streamline-diffusion solver vs manufactured u_star (rel-L2):")
    prev = None; prev_n = None
    for n_side in sorted(vr["conv"]):
        e = vr["conv"][n_side]
        rate = "" if prev is None else f"rate~{np.log(prev/e)/np.log(mesh_h(prev_n)/mesh_h(n_side)):.2f}"
        print(f"          {n_side:3d}x{n_side:<3d}  err={e:.3e}  {rate}")
        prev, prev_n = e, n_side
    print(f"          convergent: {vr['conv_ok']}")
    print(f"[VALIDATION] advection-dominated: Pe_h={vr['pe_h']:.2f}>1 and stab/phys-diff="
          f"{vr['stab_over_phys']:.1f}>1  ->  {vr['adv_dominated']}")
    print(f"[VALIDATION] collinear endpoint is real: max|u_SD - u_AV| at theta=0,rho=1 = "
          f"{vr['endpoint_identical']:.2e}  (identical operator: {vr['endpoint_ok']})\n")
    if not (vr["conv_ok"] and vr["adv_dominated"] and vr["endpoint_ok"]):
        print("  !! validation FAILED -- residuals not trustworthy; aborting.")
        return dict(status="blocked", reason="validation failed", vr=vr)

    # =========================================================== SWEEP 1: FLOW-SKEW ANGLE theta (the transition)
    thetas = np.concatenate([np.linspace(0.0, np.radians(20.0), 9),
                             np.radians(np.array([25.0, 30.0, 37.5, 45.0]))])  # dense near onset
    print("=" * 92)
    print("SWEEP 1: flow-skew angle theta  (streamline-diffusion SD vs x-pinned artificial viscosity AV; rho=1)")
    print("         theta=0 == identical operator (collinear); theta grows == streamline tensor rotates off-axis")
    print("=" * 92)
    print(f"{'theta(deg)':>10s} {'acc':>7s} {'floor':>7s} {'gap':>7s} {'|cos|':>7s}   regime")
    sweep1 = []
    for th in thetas:
        Fs = build_sigs("SD", modes_list, pts, elems, on_bnd, th, 1.0, h, SIGMA, seed=1000)
        Fa = build_sigs("AV", modes_list, pts, elems, on_bnd, th, 1.0, h, SIGMA, seed=2000)
        acc, fl = pair_acc_floor(Fs, Fa, ic, seed=300)
        cosv = float(abs(mean_dir(Fs) @ mean_dir(Fa)))
        gap = acc - fl
        regime = "SEPARABLE" if gap > SEP_THR else ("collinear~chance" if gap < 0.10 else "transition")
        sweep1.append(dict(theta=float(th), theta_deg=float(np.degrees(th)), acc=acc, floor=fl,
                           gap=gap, cos=cosv, regime=regime))
        print(f"{np.degrees(th):10.1f} {acc:7.3f} {fl:7.3f} {gap:+7.3f} {cosv:7.3f}   {regime}")

    # =========================================================== SWEEP 1b: CONSISTENCY OVERLAY (consistent SUPG vs AV)
    # the SEPARATE finding: full consistent SUPG carries a consistency footprint attributable at ANY angle.
    print("\n" + "-" * 92)
    print("OVERLAY: full CONSISTENT SUPG (load reweighted) vs the same x-pinned AV -- the CONSISTENCY channel")
    print("-" * 92)
    print(f"{'theta(deg)':>10s} {'acc':>7s} {'floor':>7s} {'gap':>7s} {'|cos|':>7s}   regime")
    sweep1c = []
    for th in thetas:
        Fs = build_sigs("SUPG", modes_list, pts, elems, on_bnd, th, 1.0, h, SIGMA, seed=1500)
        Fa = build_sigs("AV",   modes_list, pts, elems, on_bnd, th, 1.0, h, SIGMA, seed=2500)
        acc, fl = pair_acc_floor(Fs, Fa, ic, seed=305)
        cosv = float(abs(mean_dir(Fs) @ mean_dir(Fa)))
        gap = acc - fl
        regime = "SEPARABLE" if gap > SEP_THR else ("collinear~chance" if gap < 0.10 else "transition")
        sweep1c.append(dict(theta=float(th), theta_deg=float(np.degrees(th)), acc=acc, floor=fl,
                            gap=gap, cos=cosv, regime=regime))
        print(f"{np.degrees(th):10.1f} {acc:7.3f} {fl:7.3f} {gap:+7.3f} {cosv:7.3f}   {regime}")

    # =========================================================== SWEEP 2: MAGNITUDE MISMATCH rho (axis-aligned)
    rhos = np.linspace(0.5, 2.0, 9)
    print("\n" + "=" * 92)
    print("SWEEP 2: magnitude-mismatch ratio rho = nu_AV/(tau|a|^2)  (theta=0, AXIS-ALIGNED; SD vs AV)")
    print("=" * 92)
    print(f"{'rho':>7s} {'acc':>7s} {'floor':>7s} {'gap':>7s} {'|cos|':>7s}   regime")
    sweep2 = []
    for rh in rhos:
        Fs = build_sigs("SD", modes_list, pts, elems, on_bnd, 0.0, 1.0, h, SIGMA, seed=1100)
        Fa = build_sigs("AV", modes_list, pts, elems, on_bnd, 0.0, rh,  h, SIGMA, seed=2100)
        acc, fl = pair_acc_floor(Fs, Fa, ic, seed=310)
        cosv = float(abs(mean_dir(Fs) @ mean_dir(Fa)))
        gap = acc - fl
        regime = "SEPARABLE" if gap > SEP_THR else ("collinear~chance" if gap < 0.10 else "transition")
        sweep2.append(dict(rho=float(rh), acc=acc, floor=fl, gap=gap, cos=cosv, regime=regime))
        print(f"{rh:7.2f} {acc:7.3f} {fl:7.3f} {gap:+7.3f} {cosv:7.3f}   {regime}")

    # =========================================================== CONTROLS at the COLLINEAR endpoint (theta=0, rho=1)
    print("\n" + "=" * 92)
    print("CONTROLS at the COLLINEAR endpoint (theta=0, rho=1; SD scheme)")
    print("=" * 92)
    Fnc = build_sigs("SD", modes_list, pts, elems, on_bnd, 0.0, 1.0, h, SIGMA, seed=9000)
    half = N_IC // 2
    nc1_draws = []
    for s in range(8):
        perm = np.random.default_rng(1000 + s).permutation(N_IC)
        gA, gB = perm[:half], perm[half:]
        Xn = np.vstack([Fnc[gA], Fnc[gB]]); yn = np.r_[np.zeros(half), np.ones(N_IC - half)]
        gn = np.r_[ic[gA], ic[gB]]
        nc1_draws.append(cv_acc(Xn, yn, gn))
    nc1 = float(np.mean(nc1_draws)); nc1_sd = float(np.std(nc1_draws))
    nc1_fl = perm_floor(np.vstack([Fnc[:half], Fnc[half:]]),
                        np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[:half], ic[half:]], 31)
    print(f"NC1 IC+noise (same scheme SD): acc={nc1:.3f} +/- {nc1_sd:.3f}  floor={nc1_fl:.3f}  "
          f"(mean over 8 arbitrary-label splits; chance~0.50)")
    h_b = mesh_h(34); pts_b, elems_b, bnd_b = make_mesh(34)
    Fg_a = build_sigs("SD", modes_list, pts,   elems,   on_bnd, 0.0, 1.0, h,   SIGMA, seed=7000)
    Fg_b = build_sigs("SD", modes_list, pts_b, elems_b, bnd_b,  0.0, 1.0, h_b, SIGMA, seed=7700)
    nc2, nc2_fl = pair_acc_floor(Fg_a, Fg_b, ic, seed=41)
    print(f"NC2 mesh-resolution change (24 vs 34, same SD): acc={nc2:.3f}  floor={nc2_fl:.3f}  "
          f"(h-dependent tau confound; diagnostic)")

    # =========================================================== CROSSING POINT (the headline)
    def crossing(sweep, key, thr=SEP_THR):
        """First parameter value where the accuracy gap CROSSES UP through `thr` (collinear -> separable).
        Linear-interpolate between the adjacent sweep points that bracket the crossing."""
        xs = np.array([s[key] for s in sweep]); gaps = np.array([s["gap"] for s in sweep])
        order = np.argsort(xs); xs = xs[order]; gaps = gaps[order]
        for i in range(1, len(xs)):
            if gaps[i - 1] < thr <= gaps[i]:
                g0, g1, x0, x1 = gaps[i - 1], gaps[i], xs[i - 1], xs[i]
                return float(x0 + (thr - g0) * (x1 - x0) / (g1 - g0))
        return None

    th_cross = crossing(sweep1, "theta_deg")
    rho_cross = crossing(sweep2, "rho")
    s1_lo, s1_hi = sweep1[0], sweep1[-1]
    s1c_lo = sweep1c[0]
    s2_min = min(sweep2, key=lambda s: s["gap"]); s2_max = max(sweep2, key=lambda s: s["gap"])

    print("\n" + "=" * 92)
    print("TRANSITION (the characterized boundary)")
    print("=" * 92)
    print(f"ANISOTROPY channel (SD vs AV): theta=0deg gap={s1_lo['gap']:+.3f} (|cos|={s1_lo['cos']:.3f}, "
          f"COLLINEAR/at-chance)  ->  theta=45deg gap={s1_hi['gap']:+.3f} (|cos|={s1_hi['cos']:.3f}, SEPARABLE)")
    if th_cross is not None:
        print(f"  *** CROSSES to SEPARABLE (gap>{SEP_THR}) at theta ~ {th_cross:.1f} deg ***")
        print(f"      below ~{th_cross:.0f}deg: collinear/at-chance ; above: attributable. THIS is the boundary.")
    else:
        print(f"  no clean crossing of gap={SEP_THR} on theta in [0,45deg] (max gap {s1_hi['gap']:+.3f})")
    print(f"CONSISTENCY channel (consistent SUPG vs AV): theta=0deg gap={s1c_lo['gap']:+.3f} "
          f"-> {'SEPARABLE even axis-aligned (consistency footprint attributable at any angle)' if s1c_lo['gap']>SEP_THR else 'at-chance'}")
    print(f"MAGNITUDE channel (rho, axis-aligned): gap in [{s2_min['gap']:+.3f} (rho={s2_min['rho']:.2f}), "
          f"{s2_max['gap']:+.3f} (rho={s2_max['rho']:.2f})]")
    if rho_cross is not None:
        print(f"  crosses gap>{SEP_THR} at rho ~ {rho_cross:.2f}")
    else:
        print(f"  unit-direction signature is largely BLIND to pure magnitude mismatch (no gap>{SEP_THR} "
              f"crossing on rho in [0.5,2]): geometry, not magnitude, carries separability")

    # =========================================================== CSV
    csv = os.path.join(TAB, "collinearity_transition.csv")
    with open(csv, "w") as fcsv:
        fcsv.write("sweep,channel,param,value,accuracy,perm_floor,gap,abs_cos,regime\n")
        for s in sweep1:
            fcsv.write(f"theta,anisotropy_SDvsAV,theta_deg,{s['theta_deg']:.4f},{s['acc']:.4f},{s['floor']:.4f},"
                       f"{s['gap']:.4f},{s['cos']:.4f},{s['regime']}\n")
        for s in sweep1c:
            fcsv.write(f"theta,consistency_SUPGvsAV,theta_deg,{s['theta_deg']:.4f},{s['acc']:.4f},{s['floor']:.4f},"
                       f"{s['gap']:.4f},{s['cos']:.4f},{s['regime']}\n")
        for s in sweep2:
            fcsv.write(f"rho,magnitude_SDvsAV,nu_AV_over_tau_a2,{s['rho']:.4f},{s['acc']:.4f},{s['floor']:.4f},"
                       f"{s['gap']:.4f},{s['cos']:.4f},{s['regime']}\n")
        fcsv.write(f"control,nc1,NC1_ic_noise,,{nc1:.4f},{nc1_fl:.4f},{nc1-nc1_fl:.4f},,same_scheme_sd_{nc1_sd:.3f}\n")
        fcsv.write(f"control,nc2,NC2_mesh_change,,{nc2:.4f},{nc2_fl:.4f},{nc2-nc2_fl:.4f},,h_dependent_tau_confound\n")
        fcsv.write(f"crossing,anisotropy,theta_deg_at_gap{SEP_THR},{'' if th_cross is None else f'{th_cross:.4f}'},,,,,separability_onset\n")
        fcsv.write(f"crossing,magnitude,rho_at_gap{SEP_THR},{'' if rho_cross is None else f'{rho_cross:.4f}'},,,,,separability_onset\n")
        fcsv.write(f"regime,,Pe_h,{pe0:.4f},,,,,cell_Peclet\n")
        fcsv.write(f"regime,,tau,{tau0:.6f},,,,,SUPG_tau\n")
        fcsv.write(f"regime,,stab_over_phys,{tau0*AMAG*AMAG/D_PHYS:.4f},,,,,added_stab_over_physical_diff\n")
        fcsv.write(f"validation,,conv_rel_L2_finest,{vr['conv'][max(vr['conv'])]:.6f},,,,,MMS_convergence\n")
        fcsv.write(f"validation,,endpoint_identical_maxabs,{vr['endpoint_identical']:.3e},,,,,collinear_endpoint_real\n")
    print(f"\nmetrics -> {csv}")

    res = dict(sweep1=sweep1, sweep1c=sweep1c, sweep2=sweep2, thetas=thetas, rhos=rhos,
               th_cross=th_cross, rho_cross=rho_cross, nc1=nc1, nc1_sd=nc1_sd, nc1_fl=nc1_fl,
               nc2=nc2, nc2_fl=nc2_fl, pe0=pe0, tau0=tau0, vr=vr,
               pts=pts, elems=elems, on_bnd=on_bnd, modes_list=modes_list, h=h, ic=ic,
               s1c_lo=s1c_lo, s1_lo=s1_lo, s1_hi=s1_hi)
    _figure(res)
    return res

# ============================================================ figure
def _figure(r):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", palette="muted", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP, ORNG = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0", "#dd8452"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    fig.subplots_adjust(wspace=0.27, hspace=0.37)

    s1, s1c, s2 = r["sweep1"], r["sweep1c"], r["sweep2"]
    th = [s["theta_deg"] for s in s1]; acc1 = [s["acc"] for s in s1]; fl1 = [s["floor"] for s in s1]
    cos1 = [s["cos"] for s in s1]
    accc = [s["acc"] for s in s1c]
    rho = [s["rho"] for s in s2]; acc2 = [s["acc"] for s in s2]; fl2 = [s["floor"] for s in s2]; cos2 = [s["cos"] for s in s2]

    # ---- A: the transition curve vs theta (headline) ----
    axA = axes[0, 0]
    sep = np.mean(fl1) + 0.20
    axA.axhspan(sep, 1.02, color=GREEN, alpha=0.07)
    axA.axhspan(0.40, np.mean(fl1) + 0.10, color=RED, alpha=0.05)
    axA.plot(th, acc1, "o-", color=BLUE, lw=2.2, ms=5.5, label="anisotropy channel (SD vs AV)", zorder=4)
    axA.plot(th, accc, "^--", color=PURP, lw=1.6, ms=4.5, label="consistency channel (consistent SUPG vs AV)", zorder=3)
    axA.plot(th, fl1, "s:", color=GREY, lw=1.3, ms=3.5, label="permutation floor")
    axA.axhline(sep, color=GREEN, ls=(0, (4, 3)), lw=1.1)
    axA.text(46, sep + 0.005, "separable\n(floor+0.2)", color=GREEN, fontsize=6.8, va="bottom", ha="right")
    if r["th_cross"] is not None:
        axA.axvline(r["th_cross"], color=RED, ls="-.", lw=1.7, zorder=5)
        axA.annotate(f"crossing\n$\\theta\\approx{r['th_cross']:.0f}\\degree$", (r["th_cross"], 0.62),
                     textcoords="offset points", xytext=(7, 0), color=RED, fontsize=8.5, fontweight="bold")
    axA.set_xlabel("flow-skew angle $\\theta$ (deg)"); axA.set_ylabel("GroupKFold accuracy")
    axA.set_ylim(0.40, 1.02)
    axA.set_title("Transition: anisotropy is BORN as flow skews off the mesh axis\n(consistency is attributable at any angle)", fontsize=9.4)
    axA.legend(frameon=False, fontsize=6.9, loc="center right")
    axA.text(-0.14, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # ---- B: WHY -- collinearity cos vs theta ----
    axB = axes[0, 1]
    axB.plot(th, cos1, "o-", color=PURP, lw=2.2, ms=5.5)
    axB.axhline(1.0, color=GREY, ls=":", lw=1.0)
    axB.set_xlabel("flow-skew angle $\\theta$ (deg)")
    axB.set_ylabel(r"$|\cos\angle(\hat c_{\rm SD},\,\hat c_{\rm AV})|$")
    axB.set_ylim(min(0.0, min(cos1) - 0.05), 1.03)
    axB.text(1.0, 0.97, "$|\\cos|{=}1$ at $\\theta{=}0$:\ncollinear $\\Rightarrow$ at-chance", fontsize=8.0, color=PURP, va="top")
    axB.set_title("WHY: signature collinearity falls as the streamline\ntensor rotates away from the pinned AV tensor", fontsize=9.4)
    axB.text(-0.14, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # ---- C: magnitude sweep (rho, axis-aligned) ----
    axC = axes[1, 0]
    axC.plot(rho, acc2, "o-", color=ORNG, lw=2.0, ms=5, label="attribution acc (SD vs AV)")
    axC.plot(rho, fl2, "s:", color=GREY, lw=1.3, ms=3.5, label="permutation floor")
    axC.axhline(np.mean(fl2) + 0.20, color=GREEN, ls=(0, (4, 3)), lw=1.1, label="separable threshold")
    axC.axvline(1.0, color=GREY, ls="-.", lw=1.0)
    axC.text(1.03, 0.93, "matched\n$\\rho{=}1$", fontsize=7.3, color="#555", va="top")
    axC.set_xlabel(r"magnitude mismatch $\rho=\nu_{\rm AV}/(\tau|a|^2)$")
    axC.set_ylabel("GroupKFold accuracy"); axC.set_ylim(0.40, 1.02)
    axC.set_title("Axis-aligned ($\\theta{=}0$): the unit-direction signature is\nlargely BLIND to pure magnitude mismatch", fontsize=9.4)
    axC.legend(frameon=False, fontsize=7.0, loc="center right")
    axC.text(-0.14, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # ---- D: mean signature directions at the two theta endpoints ----
    axD = axes[1, 1]
    h = r["h"]; pts, elems, on_bnd = r["pts"], r["elems"], r["on_bnd"]; modes_list = r["modes_list"]
    Fs0 = build_sigs("SD", modes_list, pts, elems, on_bnd, 0.0, 1.0, h, SIGMA, 1000)
    Fa0 = build_sigs("AV", modes_list, pts, elems, on_bnd, 0.0, 1.0, h, SIGMA, 2000)
    th_hi = r["thetas"][-1]
    Fs1 = build_sigs("SD", modes_list, pts, elems, on_bnd, th_hi, 1.0, h, SIGMA, 1000)
    Fa1 = build_sigs("AV", modes_list, pts, elems, on_bnd, th_hi, 1.0, h, SIGMA, 2000)
    labs = [r"$c_{xx}$", r"$c_{yy}$", r"$c_{xy}$", r"$c_{xxxx}$", r"$c_{yyyy}$"]
    xb = np.arange(len(labs)); w = 0.20
    axD.bar(xb - 1.5 * w, mean_dir(Fs0), w, color=BLUE, label=r"SD $\theta{=}0$")
    axD.bar(xb - 0.5 * w, mean_dir(Fa0), w, color="#a8c0e0", label=r"AV $\theta{=}0$")
    axD.bar(xb + 0.5 * w, mean_dir(Fs1), w, color=RED, label=r"SD $\theta{=}45\degree$")
    axD.bar(xb + 1.5 * w, mean_dir(Fa1), w, color="#e8a8a8", label=r"AV $\theta{=}45\degree$")
    axD.axhline(0, color=GREY, lw=0.8); axD.set_xticks(xb); axD.set_xticklabels(labs, fontsize=8.5)
    axD.set_ylabel("unit coeff direction")
    cos0 = float(abs(mean_dir(Fs0) @ mean_dir(Fa0))); cosh = float(abs(mean_dir(Fs1) @ mean_dir(Fa1)))
    axD.set_title(f"Mean signatures: $|\\cos|$ {cos0:.2f} ($\\theta{{=}}0$, collinear) $\\to$ {cosh:.2f} ($\\theta{{=}}45\\degree$)", fontsize=9.0)
    axD.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper right")
    axD.text(-0.14, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "fig_collinearity_transition.png")
    fig.savefig(out); plt.close(fig)
    print(f"figure  -> {out}")
    return out

if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
    print("\ndone.")
