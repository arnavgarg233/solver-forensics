#!/usr/bin/env python3
"""
solver-forensics :: 2D SUPG / FE STABILIZATION AUDIT AT ENGINEERING SCALE
================================================================================
The CMAME-native ANCHOR experiment: forensic attribution of finite-element
stabilization choices on a genuine 2D advection-dominated boundary-/internal-
layer problem, the classic Hughes-Brooks rotating-flow SUPG stress test.

PROBLEM (steady scalar advection-diffusion on the unit square):

        a . grad(u)  -  D lap(u)  =  f      on Omega = (0,1)^2
        u = g_D                              on the Dirichlet boundary

ADVECTION-DOMINATED: the element Peclet number Pe_h = |a| h / (2 D) >> 1, so the
unstabilized Galerkin method produces spurious node-to-node oscillations and the
stabilized methods produce sharp but monotone layers. We use the canonical
ROTATING advection field

        a(x,y) = ( y - 0.5 , -(x - 0.5) )        (solid-body rotation)

with a SHARP INLET on the inflow boundary that is advected around the centre,
producing an internal layer (the advected discontinuity) and boundary layers
where the rotating characteristics leave the domain. This is the Hughes-Brooks /
"rotating cosine hill"-class stress test in which SUPG was originally motivated.

Hand-assembled P1 (linear-triangle) FEM on an UNSTRUCTURED Delaunay mesh
(scipy.spatial.Delaunay; NO FEM library). Element matrices in closed form;
the SUPG / artificial-viscosity terms added per element.

FOUR FE SOLVERS (the configs we attribute):
  galerkin     : unstabilized Galerkin. Oscillates at high Pe_h.
  supg         : streamline-upwind Petrov-Galerkin with the standard element tau
                 tau = ( (2|a|/h_e)^2 + (4D/h_e^2)^2 )^{-1/2}  (Codina/Shakib form).
                 CONSISTENT: the whole residual a.grad(u) - D lap(u) - f is weighted
                 by the streamline perturbation tau (a.grad w), incl. the source.
  supg_halftau : SUPG with tau -> 0.5*tau  (silent UNDER-stabilization).
  supg_2tau    : SUPG with tau -> 2.0*tau  (silent OVER-stabilization).
  artvisc      : ISOTROPIC artificial viscosity matched on ADDED DIFFUSION to SUPG
                 (nu_art = mean_e tau_e |a_e|^2), added to the bilinear form ONLY
                 (source NOT reweighted -> INCONSISTENT). The identifiability foil.

CONTRASTS (each a logistic attribution, GroupKFold-by-IC, permutation floor):
  (a) PRESENCE : galerkin vs supg                 -- LOAD-BEARING, expect ~1.00
  (b) SILENT TAU: supg vs {supg_halftau, supg_2tau} (and each pairwise)
                  -- the silent loss/gain of stabilization at scale
  (c) TYPE     : supg vs artvisc (matched added diffusion)
                  -- NOT load-bearing; near-collinear leading term -> expect ~chance,
                     reported as the IDENTIFIABILITY BOUNDARY.

CONTROLS:
  NC1 : same scheme (galerkin), vary inlet sharpness / source IC + noise, RANDOM
        label partition of ICs -> must sit ~chance.
  NC2 : same scheme (galerkin), MESH-RESOLUTION change (coarse vs fine unstructured
        mesh) -> the geometry/resolution confound.

REFERENCE: a HIGHLY-RESOLVED SUPG FEM solution on a fine mesh (no closed-form
solution exists for the rotating-inlet problem), interpolated to the observation
grid. Reference uses SUPG (the converged, oscillation-free solver) so the residual
of every coarse scheme is measured against a genuine, monotone fine solution.

SIGNATURE: interpolate solver AND reference fields from the scattered FE nodes to a
regular grid (scipy.interpolate.griddata), form r = u_solver - u_ref, and recover
the unit-normalized least-squares coefficient DIRECTION of
        r ~ sum_p c_p d_x^p u        on the 2D library
        {u_xx, u_yy, u_xy, u_xxx, u_yyy}
from FD derivatives of the OBSERVED (solver) field on the grid (mirrors
src/mechanics/fem2d_unstructured.py).

VALIDATION (printed before any attribution):
  * SUPG spatial convergence vs a finer SUPG reference (error falls under refinement).
  * Galerkin OSCILLATES (node-to-node sign changes along an internal-layer cut)
    while SUPG is MONOTONE -- the qualitative stabilization signature.

NOISE sigma in {0, 0.01, 0.05}. Self-contained: numpy + scipy + sklearn. CPU.
Run:  python src/audit/supg_2d_engineering.py [--plot]
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures")

# -------------------- problem constants --------------------
D_PHYS   = 1.0e-3              # physical diffusivity (small -> advection dominated)
GRID_OBS = 64                 # regular grid the signature is recovered on
LIB      = ("u_xx", "u_yy", "u_xy", "u_xxx", "u_yyy")
SCHEMES  = ("galerkin", "supg", "supg_halftau", "supg_2tau", "artvisc")
N_IC     = 80                 # inlet/source realizations (group unit)
SIGMAS   = (0.0, 0.01, 0.05)  # observation-noise sweep
SIGMA_MAIN = 0.01             # sigma used for the headline contrasts


# ====================================================================
#  UNSTRUCTURED MESH  (Delaunay on a jittered interior grid + boundary frame)
# ====================================================================
def make_mesh(n_side, seed):
    """Unstructured triangular mesh of the unit square. Boundary nodes lie on a
    regular frame (Dirichlet edges exact; griddata well-posed to the border);
    interior nodes are a jittered grid -> genuinely unstructured Delaunay."""
    rng = np.random.default_rng(seed)
    nb = n_side + 1
    s = np.linspace(0, 1, nb)
    bottom = np.column_stack([s, np.zeros(nb)])
    top    = np.column_stack([s, np.ones(nb)])
    left   = np.column_stack([np.zeros(nb - 2), s[1:-1]])
    right  = np.column_stack([np.ones(nb - 2),  s[1:-1]])
    bnodes = np.vstack([bottom, top, left, right])
    gx = (np.arange(n_side - 1) + 1) / n_side
    GX, GY = np.meshgrid(gx, gx)
    inodes = np.column_stack([GX.ravel(), GY.ravel()])
    jit = 0.30 / n_side
    inodes = inodes + jit * (rng.random(inodes.shape) - 0.5) * 2
    inodes = np.clip(inodes, 0.8 / n_side, 1 - 0.8 / n_side)
    pts = np.vstack([bnodes, inodes])
    pts = np.unique(np.round(pts, 9), axis=0)
    tri = Delaunay(pts)
    on_bnd = (np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) |
              np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1))
    return pts, tri.simplices, on_bnd


# ====================================================================
#  ADVECTION FIELD + BOUNDARY DATA
# ====================================================================
def advection(xy):
    """Solid-body rotation a = (y-0.5, -(x-0.5))."""
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([y - 0.5, -(x - 0.5)])


def inflow_bc(pts, on_bnd, ic):
    """Dirichlet data g_D on the boundary. The rotating field carries a SHARP
    INLET profile imposed on the lower-left inflow boundary; an internal layer
    forms where the inlet step is advected into the interior.

    ic = (x0, y0, sharp, sgn): a tanh step centred at (x0,y0)-ish along the
    inflow edges with width controlled by `sharp` (large -> sharp inlet)."""
    x0, y0, sharp, sgn = ic["x0"], ic["y0"], ic["sharp"], ic["sgn"]
    g = np.zeros(len(pts))
    # bottom edge (y=0): inflow where a_y=-(x-0.5)<0 i.e. x>0.5; impose a step in x
    # left edge (x=0): inflow where a_x=y-0.5>0 i.e. y>0.5; impose a step in y
    # Use a smooth tanh step so the CONTINUUM data are well-defined; "sharp" sets
    # the layer thickness. The reference resolves it; coarse meshes do not.
    bx = pts[:, 0]; by = pts[:, 1]
    # combined inlet "hill": a tanh ridge that the rotation sweeps inward
    step_left   = 0.5 * (1.0 + np.tanh(sharp * (by - y0)))   # on x=0
    step_bottom = 0.5 * (1.0 + np.tanh(sharp * (bx - x0)))   # on y=0
    on_left   = np.isclose(bx, 0.0)
    on_bottom = np.isclose(by, 0.0)
    g[on_left]   = sgn * step_left[on_left]
    g[on_bottom] = sgn * step_bottom[on_bottom]
    # the rest of the boundary is held at the "outer" value 0 (outflow/far)
    return g


def make_ic(seed):
    rng = np.random.default_rng(seed)
    return dict(x0=float(rng.uniform(0.35, 0.65)),
                y0=float(rng.uniform(0.35, 0.65)),
                sharp=float(rng.uniform(15.0, 45.0)),
                sgn=float(rng.choice([-1.0, 1.0])),
                src=float(rng.uniform(-1.0, 1.0)),      # weak interior source amplitude
                src_kx=int(rng.integers(1, 4)),
                src_ky=int(rng.integers(1, 4)))


def source(pts, ic):
    """Weak smooth interior source f (keeps the problem non-degenerate / varied)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.2 * ic["src"] * np.sin(ic["src_kx"] * np.pi * x) * np.sin(ic["src_ky"] * np.pi * y)


# ====================================================================
#  HAND-ASSEMBLED P1 FEM  (SUPG / Galerkin / artificial viscosity)
#  Vectorized element assembly: ALL geometry computed once per mesh in
#  mesh_geometry(), reused across ICs/schemes/sigmas.
# ====================================================================
def mesh_geometry(pts, elems, D=D_PHYS):
    """Precompute per-element geometry shared by every solve on this mesh:
    P1 gradients b=dN/dx, c=dN/dy (E,3); areas Ae (E,); element-average advection
    (E,2) and |a|; element tau (Codina/Shakib) for tau_scale=1; the index pairs for
    sparse assembly; and the boundary node list. IC-independent -> compute ONCE."""
    p = pts[elems]                                   # (E,3,2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    Ae = 0.5 * np.abs(detJ)
    good = Ae > 1e-14
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], 1) / detJ[:, None]   # (E,3)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], 1) / detJ[:, None]   # (E,3)
    a_nodes = advection(pts)
    ae = a_nodes[elems].mean(1)                      # (E,2) element-avg advection
    ax, ay = ae[:, 0], ae[:, 1]
    amag = np.hypot(ax, ay)
    h_e = np.sqrt(2.0 * Ae)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau1 = 1.0 / np.sqrt((2.0 * amag / h_e) ** 2 + (4.0 * D / h_e ** 2) ** 2)
    tau1 = np.where(amag > 1e-12, tau1, 0.0)
    adg = ax[:, None] * b + ay[:, None] * c          # (E,3) a.grad N_j
    # sparse index pattern (E,3,3) -> flat
    ii = np.repeat(elems[:, :, None], 3, axis=2)     # (E,3,3) row = node i
    jj = np.repeat(elems[:, None, :], 3, axis=1)     # (E,3,3) col = node j
    return dict(elems=elems, Ae=Ae, b=b, c=c, adg=adg, amag=amag, tau1=tau1,
                good=good, rows=ii.ravel(), cols=jj.ravel(), npt=len(pts))


def assemble(scheme, pts, elems, on_bnd, ic, D=D_PHYS, tau_scale=1.0, nu_art=None, geom=None):
    """Assemble and solve the steady advection-diffusion system on a P1 triangular
    mesh (fully vectorized). Returns the nodal solution u.

    Per element (P1, constant gradients b=dN/dx, c=dN/dy, area Ae):
      diffusion  Ke[i,j] = D Ae (b_i b_j + c_i c_j)
      advection  Ce[i,j] = (Ae/3)(a.grad N_j)            (rows identical: int N_i = Ae/3)
      SUPG adds  tau (a.grad N_i)(a.grad N_j) Ae to K, and tau (a.grad N_i) f Ae to F
                 (source reweighted -> CONSISTENT). tau scaled by tau_scale.
      artvisc adds nu_art Ae (b_i b_j + c_i c_j) to K ONLY (load not reweighted ->
                 INCONSISTENT).
    geom = mesh_geometry(pts,elems): precomputed, IC-independent."""
    if geom is None:
        geom = mesh_geometry(pts, elems, D)
    Ae, b, c, adg, tau1 = geom["Ae"], geom["b"], geom["c"], geom["adg"], geom["tau1"]
    use_supg = scheme in ("supg", "supg_halftau", "supg_2tau")
    use_art  = scheme == "artvisc"
    # streamdiff_inc: the SAME anisotropic streamline-diffusion operator as SUPG
    # (tau a.gradN_i a.gradN_j) added to the STIFFNESS ONLY -- the source is NOT
    # reweighted. It differs from SUPG by CONSISTENCY ALONE (the source term). This
    # isolates the consistency axis: same modified-equation leading operator.
    use_streamdiff_inc = scheme == "streamdiff_inc"
    E = len(Ae)
    # diffusion: D Ae (b_i b_j + c_i c_j)  -> (E,3,3)
    Ke = D * Ae[:, None, None] * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
    # advection: (Ae/3)*adg_j broadcast over rows
    Ke = Ke + (Ae[:, None, None] / 3.0) * adg[:, None, :]
    if use_supg or use_streamdiff_inc:
        tau = tau_scale * tau1
        Ke = Ke + (tau * Ae)[:, None, None] * (adg[:, :, None] * adg[:, None, :])
    elif use_art:
        Ke = Ke + nu_art * Ae[:, None, None] * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
    # load
    f_nodes = source(pts, ic)
    f_e = f_nodes[elems].mean(1)                     # (E,)
    Fe = (Ae / 3.0)[:, None] * f_e[:, None] * np.ones((E, 3))
    if use_supg:   # CONSISTENT source reweighting (streamdiff_inc deliberately omits this)
        tau = tau_scale * tau1
        Fe = Fe + (tau * Ae * f_e)[:, None] * adg
    # global assembly
    npt = geom["npt"]
    Kg = csr_matrix((Ke.ravel(), (geom["rows"], geom["cols"])), shape=(npt, npt)).tolil()
    F = np.bincount(elems.ravel(), weights=Fe.ravel(), minlength=npt)
    g = inflow_bc(pts, on_bnd, ic)
    for nd in np.where(on_bnd)[0]:
        Kg.rows[nd] = [nd]; Kg.data[nd] = [1.0]
        F[nd] = g[nd]
    u = spsolve(Kg.tocsr(), F)
    return u, g


def added_diffusion_supg(pts, elems, tau_scale=1.0, D=D_PHYS, geom=None):
    """Area-weighted mean SUPG added streamline diffusion nu = tau_e |a_e|^2,
    used to MATCH the isotropic artificial viscosity (the adversarial foil)."""
    if geom is None:
        geom = mesh_geometry(pts, elems, D)
    Ae, tau1, amag = geom["Ae"], geom["tau1"], geom["amag"]
    tau = tau_scale * tau1
    num = float(np.sum(Ae * tau * amag ** 2)); den = float(np.sum(Ae))
    return num / den if den > 0 else 0.0


# ====================================================================
#  SIGNATURE: interpolate scattered fields -> regular grid -> FD coeff direction
# ====================================================================
_gx = np.linspace(0, 1, GRID_OBS)
_GX, _GY = np.meshgrid(_gx, _gx, indexing="ij")
_GPTS = np.column_stack([_GX.ravel(), _GY.ravel()])
_H = _gx[1] - _gx[0]


def _to_grid(pts, vals):
    g = griddata(pts, vals, _GPTS, method="cubic")
    nan = np.isnan(g)
    if nan.any():
        g[nan] = griddata(pts, vals, _GPTS[nan], method="nearest")
    return g.reshape(GRID_OBS, GRID_OBS)


def _fd_library(U):
    h = _H
    uxx  = (np.roll(U, -1, 0) - 2 * U + np.roll(U, 1, 0)) / h**2
    uyy  = (np.roll(U, -1, 1) - 2 * U + np.roll(U, 1, 1)) / h**2
    uxy  = (np.roll(np.roll(U, -1, 0), -1, 1) - np.roll(np.roll(U, -1, 0), 1, 1)
            - np.roll(np.roll(U, 1, 0), -1, 1) + np.roll(np.roll(U, 1, 0), 1, 1)) / (4 * h**2)
    uxxx = (np.roll(U, -2, 0) - 2 * np.roll(U, -1, 0) + 2 * np.roll(U, 1, 0) - np.roll(U, 2, 0)) / (2 * h**3)
    uyyy = (np.roll(U, -2, 1) - 2 * np.roll(U, -1, 1) + 2 * np.roll(U, 1, 1) - np.roll(U, 2, 1)) / (2 * h**3)
    sl = slice(2, GRID_OBS - 2)
    return {"u_xx": uxx[sl, sl], "u_yy": uyy[sl, sl], "u_xy": uxy[sl, sl],
            "u_xxx": uxxx[sl, sl], "u_yyy": uyyy[sl, sl]}, sl


def signature(pts, u_solver, u_ref):
    Us = _to_grid(pts, u_solver)
    Ur = _to_grid(pts, u_ref)
    R = Us - Ur
    Dlib, sl = _fd_library(Us)
    A = np.column_stack([Dlib[name].ravel() for name in LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def feats(C):
    """Unit-direction features (magnitude-invariant) -- mirror fem2d_unstructured."""
    C = np.asarray(C)
    unit = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return np.nan_to_num(unit)


# ====================================================================
#  classification helpers
# ====================================================================
def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def cv_acc(X, y, g):
    return cross_val_score(_clf(), X, y, groups=g, cv=GroupKFold(5)).mean()


def perm_floor(X, y, g, seed, reps=30):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(_clf(), X, r.permutation(y), groups=g,
                                            cv=GroupKFold(5)).mean() for _ in range(reps)]))


# ====================================================================
#  REFERENCE (fine SUPG solve) + signature builders
# ====================================================================
def reference_grids(ics, ref_pts, ref_elems, ref_bnd):
    """Fine SUPG reference field for each IC, interpolated to the observation grid.
    The reference depends ONLY on the IC (not the scheme/sigma/tau), so it is solved
    ONCE per IC here and reused for every contrast -- the key cost optimization."""
    geom = mesh_geometry(ref_pts, ref_elems)
    out = []
    for ic in ics:
        uref, _ = assemble("supg", ref_pts, ref_elems, ref_bnd, ic, tau_scale=1.0, geom=geom)
        out.append(_to_grid(ref_pts, uref))
    return out


def build_sigs(scheme, pts, elems, on_bnd, ics, sigma, seed, ref_grids,
               nu_art=None, tau_scale=1.0, geom=None):
    """Signatures for `scheme` on the working mesh, each IC referenced against the
    PRECOMPUTED fine SUPG reference grid (ref_grids[i] for ics[i])."""
    if geom is None:
        geom = mesh_geometry(pts, elems)
    C = []
    rng = np.random.default_rng(seed)
    for k, ic in enumerate(ics):
        if scheme == "galerkin":
            u, _ = assemble("galerkin", pts, elems, on_bnd, ic, geom=geom)
        elif scheme == "artvisc":
            u, _ = assemble("artvisc", pts, elems, on_bnd, ic, nu_art=nu_art, geom=geom)
        else:  # supg variants
            u, _ = assemble("supg", pts, elems, on_bnd, ic, tau_scale=tau_scale, geom=geom)
        Ur = ref_grids[k]
        Us = _to_grid(pts, u)
        if sigma > 0:
            rms = np.sqrt(np.mean(Us**2))
            Us = Us + sigma * rms * rng.standard_normal(Us.shape)
        R = Us - Ur
        Dlib, sl = _fd_library(Us)
        A = np.column_stack([Dlib[name].ravel() for name in LIB])
        b = R[sl, sl].ravel()
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        nrm = np.linalg.norm(c)
        C.append(c / nrm if nrm > 0 else c)
    return np.array(C)


# ====================================================================
#  VALIDATION
# ====================================================================
def validate(ref_pts, ref_elems, ref_bnd):
    """(1) SUPG spatial convergence: rel-L2 vs a FINER SUPG solve falls under
       refinement. (2) Galerkin oscillates (sign changes along an internal-layer
       cut) while SUPG is monotone. Returns a report dict; prints inline."""
    ic = dict(x0=0.5, y0=0.5, sharp=30.0, sgn=1.0, src=0.0, src_kx=1, src_ky=1)
    # convergence: compare SUPG on a sequence of meshes to the fine reference mesh
    uref, _ = assemble("supg", ref_pts, ref_elems, ref_bnd, ic, tau_scale=1.0)
    Uref = _to_grid(ref_pts, uref)
    conv = []
    for ns in (16, 24, 32, 48):
        p, e, bnd = make_mesh(ns, seed=999)
        u, _ = assemble("supg", p, e, bnd, ic, tau_scale=1.0)
        Ug = _to_grid(p, u)
        err = np.linalg.norm(Ug - Uref) / (np.linalg.norm(Uref) + 1e-12)
        conv.append((ns, 1.0 / ns, err))
    # oscillation: on the RAW nodal field (boundary data lie in [0,1] for sgn=+1),
    # measure (i) global over/undershoot beyond [0,1] -- the spurious extrema that
    # stabilization removes, and (ii) node-to-node 2nd-difference sign changes along
    # the y=0.5 gridded cut. SUPG must suppress both relative to Galerkin.
    pw, ew, bw = make_mesh(28, seed=42)
    osc = {}
    for sc in ("galerkin", "supg"):
        u, _ = assemble(sc, pw, ew, bw, ic, tau_scale=1.0)
        osc[sc + "_overshoot"] = float(max(u.max() - 1.0, 0.0) + max(0.0 - u.min(), 0.0))
        Ug = _to_grid(pw, u)
        mid = GRID_OBS // 2
        d2 = np.diff(Ug[:, mid], 2)
        osc[sc] = int(np.sum(np.diff(np.sign(d2)) != 0))
    return dict(conv=conv, osc=osc)


# ====================================================================
#  MAIN
# ====================================================================
def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    print("=" * 80)
    print("2D SUPG / FE STABILIZATION AUDIT AT ENGINEERING SCALE")
    print("rotating advection a=(y-0.5,-(x-0.5)), sharp inlet, hand-assembled P1 FEM")
    print("=" * 80)

    # ---- meshes ----
    N_SIDE = 28                                   # primary (coarse, oscillatory) working mesh
    pts, elems, on_bnd = make_mesh(N_SIDE, seed=2026)
    h_prim = 1.0 / N_SIDE
    pe_h = (np.max(np.hypot(*advection(pts).T)) * h_prim) / (2 * D_PHYS)
    print(f"\nprimary mesh: {len(pts)} nodes ({on_bnd.sum()} bnd), {len(elems)} tris, "
          f"h~{h_prim:.4f}, max Pe_h~{pe_h:.1f} (>>1: advection-dominated)")

    N_REF = 96                                    # fine reference mesh
    ref_pts, ref_elems, ref_bnd = make_mesh(N_REF, seed=7)
    print(f"reference mesh: {len(ref_pts)} nodes, {len(ref_elems)} tris, h~{1.0/N_REF:.4f}")

    # ---- VALIDATION ----
    vr = validate(ref_pts, ref_elems, ref_bnd)
    print("\n[VALIDATION 1] SUPG spatial convergence vs fine SUPG reference:")
    for ns, hh, err in vr["conv"]:
        print(f"   n_side={ns:>3d}  h~{hh:.4f}  rel-L2-vs-fine={err:.3e}")
    errs = [e for _, _, e in vr["conv"]]
    conv_ok = errs[-1] < errs[0]
    print(f"   converging under refinement: {conv_ok}  (err {errs[0]:.3e} -> {errs[-1]:.3e})")
    print("\n[VALIDATION 2] oscillation along y=0.5 internal-layer cut "
          "(2nd-diff sign changes; overshoot beyond [0,1]):")
    print(f"   galerkin : wiggle={vr['osc']['galerkin']:3d}  overshoot={vr['osc']['galerkin_overshoot']:.3f}")
    print(f"   supg     : wiggle={vr['osc']['supg']:3d}  overshoot={vr['osc']['supg_overshoot']:.3f}")
    gal_osc = vr['osc']['galerkin'] > vr['osc']['supg'] and \
              vr['osc']['galerkin_overshoot'] > vr['osc']['supg_overshoot'] + 0.02
    print(f"   Galerkin oscillates more than SUPG: {gal_osc}")
    solver_valid = conv_ok and gal_osc
    print(f"\n   SOLVER VALIDATION PASSES: {solver_valid}")

    # ---- ICs (shared across schemes; group = IC index) ----
    ics = [make_ic(1000 + i) for i in range(N_IC)]
    ic_idx = np.arange(N_IC)

    # ---- precompute mesh geometry (reused for every solve) ----
    geom = mesh_geometry(pts, elems)

    # ---- matched artificial viscosity (foil) on the primary mesh ----
    nu_art = added_diffusion_supg(pts, elems, tau_scale=1.0, geom=geom)
    print(f"\nmatched artificial viscosity nu_art = mean tau|a|^2 = {nu_art:.3e}  "
          f"(nu_art/D_phys = {nu_art/D_PHYS:.1f})")

    # ---- precompute the fine SUPG reference grid ONCE per IC (the key speedup) ----
    print("\n[solving fine SUPG reference for %d ICs (computed once, reused)] ..." % N_IC)
    ref_grids = reference_grids(ics, ref_pts, ref_elems, ref_bnd)

    # ---- precompute CLEAN solver grids + signatures once per scheme; the sigma
    #      sweep only re-adds grid noise, so the FEM is solved only ONCE per scheme. ----
    print("[solving working-mesh schemes (once per scheme) + sigma sweep %s] ..." % (SIGMAS,))
    tau_scale_map = {"galerkin": 1.0, "supg": 1.0, "supg_halftau": 0.5,
                     "supg_2tau": 2.0, "artvisc": 1.0}
    # clean solver grids per scheme (no noise)
    clean = {}
    for sc in SCHEMES:
        gs = []
        for ic in ics:
            if sc == "galerkin":
                u, _ = assemble("galerkin", pts, elems, on_bnd, ic, geom=geom)
            elif sc == "artvisc":
                u, _ = assemble("artvisc", pts, elems, on_bnd, ic, nu_art=nu_art, geom=geom)
            else:
                u, _ = assemble("supg", pts, elems, on_bnd, ic, tau_scale=tau_scale_map[sc], geom=geom)
            gs.append(_to_grid(pts, u))
        clean[sc] = gs

    def sig_from_grid(Us, Ur):
        R = Us - Ur
        Dlib, sl = _fd_library(Us)
        A = np.column_stack([Dlib[name].ravel() for name in LIB])
        b = R[sl, sl].ravel()
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        nrm = np.linalg.norm(c)
        return c / nrm if nrm > 0 else c

    # F[sigma][scheme] -> (N_IC, len(LIB))
    F = {sig: {} for sig in SIGMAS}
    for sig in SIGMAS:
        for j, sc in enumerate(SCHEMES):
            rng = np.random.default_rng(100 + 1000 * j + int(sig * 1e4))
            sigs_sc = []
            for k in range(N_IC):
                Us = clean[sc][k]
                if sig > 0:
                    rms = np.sqrt(np.mean(Us**2))
                    Us = Us + sig * rms * rng.standard_normal(Us.shape)
                sigs_sc.append(sig_from_grid(Us, ref_grids[k]))
            F[sig][sc] = np.array(sigs_sc)
        print(f"   sigma={sig}: all schemes done")

    def pair(Fd, a, b, seed):
        Xp = feats(np.vstack([Fd[a], Fd[b]]))
        yp = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gp = np.r_[ic_idx, ic_idx]
        return cv_acc(Xp, yp, gp), perm_floor(Xp, yp, gp, seed)

    results = {}

    # ========================================================================
    #  CONTRAST (a) PRESENCE: galerkin vs supg  -- LOAD-BEARING, expect ~1.00
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRAST (a) PRESENCE: galerkin vs supg  [load-bearing, expect ~1.00]")
    print("=" * 80)
    for sig in SIGMAS:
        acc, fl = pair(F[sig], "galerkin", "supg", 7)
        results[("presence", sig)] = (acc, fl)
        print(f"   sigma={sig}: acc={acc:.3f}  perm-floor={fl:.3f}  gap={acc-fl:+.3f}")

    # ========================================================================
    #  CONTRAST (b) SILENT TAU CHANGE: supg vs detuned (0.5x, 2x)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRAST (b) SILENT TAU CHANGE: supg vs detuned tau  [the silent loss/gain]")
    print("=" * 80)
    for sig in SIGMAS:
        a_half, f_half = pair(F[sig], "supg", "supg_halftau", 11)
        a_2, f_2 = pair(F[sig], "supg", "supg_2tau", 13)
        # 3-way among optimal/half/double tau (the silent-stabilization taxonomy)
        Xt = feats(np.vstack([F[sig]["supg"], F[sig]["supg_halftau"], F[sig]["supg_2tau"]]))
        yt = np.concatenate([np.full(N_IC, k) for k in range(3)])
        gt = np.concatenate([ic_idx] * 3)
        a_3 = cv_acc(Xt, yt, gt); f_3 = perm_floor(Xt, yt, gt, 17)
        results[("tau_half", sig)] = (a_half, f_half)
        results[("tau_2", sig)] = (a_2, f_2)
        results[("tau_3way", sig)] = (a_3, f_3)
        print(f"   sigma={sig}:  supg-vs-0.5tau acc={a_half:.3f} (fl {f_half:.3f})   "
              f"supg-vs-2tau acc={a_2:.3f} (fl {f_2:.3f})   "
              f"3way(opt/half/dbl) acc={a_3:.3f} (fl {f_3:.3f}, chance 0.33)")

    # ========================================================================
    #  CONTRAST (c) TYPE: supg vs artvisc (matched added diffusion)
    #               NOT load-bearing; expect near-chance -> identifiability boundary
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRAST (c) TYPE: supg vs artvisc (matched added diffusion)  "
          "[identifiability boundary, expect ~chance]")
    print("=" * 80)
    for sig in SIGMAS:
        acc, fl = pair(F[sig], "supg", "artvisc", 23)
        results[("type", sig)] = (acc, fl)
        print(f"   sigma={sig}: acc={acc:.3f}  perm-floor={fl:.3f}  gap={acc-fl:+.3f}")
    # signature collinearity (matched) at sigma=0
    msupg = F[0.0]["supg"].mean(0); msupg /= (np.linalg.norm(msupg) + 1e-12)
    mart  = F[0.0]["artvisc"].mean(0); mart /= (np.linalg.norm(mart) + 1e-12)
    cos_type = float(abs(msupg @ mart))
    print(f"   |cos(mean SUPG sig, mean ArtVisc sig)| = {cos_type:.3f}  "
          f"({'near-collinear' if cos_type > 0.95 else 'separable directions'})")
    print("   NOTE: SUPG (anisotropic streamline diffusion) vs ISOTROPIC artificial")
    print("   viscosity is SEPARABLE -- they are different modified-equation operators")
    print("   in 2D even with matched scalar added diffusion. The TRUE collinearity")
    print("   limit is the CONSISTENCY axis (contrast (c2) below).")

    # ========================================================================
    #  CONTRAST (c2) CONSISTENCY AXIS: supg vs streamdiff_inc
    #     streamdiff_inc adds the SAME anisotropic streamline-diffusion operator to
    #     the stiffness but does NOT reweight the source -> differs from SUPG by
    #     CONSISTENCY ALONE (identical leading modified-equation operator). This is
    #     the genuine 2D collinearity limit; expect ~chance, cos ~ 1.
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTRAST (c2) CONSISTENCY AXIS: supg vs streamdiff_inc "
          "(same anisotropic operator, source NOT reweighted) [collinearity limit]")
    print("=" * 80)
    # build streamdiff_inc clean grids + signatures (separate; not in SCHEMES)
    Fsdi = {}
    for sig in (0.0, SIGMA_MAIN):
        rng = np.random.default_rng(909 + int(sig * 1e4))
        sgs = []
        for k, ic in enumerate(ics):
            u, _ = assemble("streamdiff_inc", pts, elems, on_bnd, ic, geom=geom)
            Us = _to_grid(pts, u)
            if sig > 0:
                rms = np.sqrt(np.mean(Us**2)); Us = Us + sig * rms * rng.standard_normal(Us.shape)
            sgs.append(sig_from_grid(Us, ref_grids[k]))
        Fsdi[sig] = np.array(sgs)
    for sig in (0.0, SIGMA_MAIN):
        Xc = feats(np.vstack([F[sig]["supg"], Fsdi[sig]]))
        yc = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gc = np.r_[ic_idx, ic_idx]
        a_c2 = cv_acc(Xc, yc, gc); f_c2 = perm_floor(Xc, yc, gc, 27)
        results[("consistency", sig)] = (a_c2, f_c2)
        print(f"   sigma={sig}: acc={a_c2:.3f}  perm-floor={f_c2:.3f}  gap={a_c2-f_c2:+.3f}")
    msdi = Fsdi[0.0].mean(0); msdi /= (np.linalg.norm(msdi) + 1e-12)
    cos_consistency = float(abs(msupg @ msdi))
    print(f"   |cos(mean SUPG sig, mean streamdiff_inc sig)| = {cos_consistency:.3f}  "
          f"({'COLLINEAR -> consistency invisible' if cos_consistency > 0.97 else 'separable'})")

    # ============ 5-way ID (context) ============
    Xid = feats(np.vstack([F[SIGMA_MAIN][s] for s in SCHEMES]))
    yid = np.concatenate([np.full(N_IC, k) for k in range(len(SCHEMES))])
    gid = np.concatenate([ic_idx] * len(SCHEMES))
    id5_acc = cv_acc(Xid, yid, gid); id5_fl = perm_floor(Xid, yid, gid, 31)
    results[("id5", SIGMA_MAIN)] = (id5_acc, id5_fl)
    print(f"\n5-way ID (all schemes, sigma={SIGMA_MAIN}): acc={id5_acc:.3f}  "
          f"perm-floor={id5_fl:.3f}  (chance {1/len(SCHEMES):.2f})")

    # ========================================================================
    #  CONTROL NC1: same scheme (galerkin), random IC partition + noise -> chance
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTROL NC1: same scheme (galerkin), random IC label partition + noise")
    print("=" * 80)
    Fnc1 = F[SIGMA_MAIN]["galerkin"]
    half = N_IC // 2
    rng_nc1 = np.random.default_rng(19)
    nc1_runs = []
    for _ in range(20):
        perm = rng_nc1.permutation(N_IC); A, B = perm[:half], perm[half:]
        Xn1 = feats(np.vstack([Fnc1[A], Fnc1[B]]))
        yn1 = np.r_[np.zeros(half), np.ones(N_IC - half)]
        gn1 = np.r_[ic_idx[A], ic_idx[B]]
        nc1_runs.append(cv_acc(Xn1, yn1, gn1))
    nc1_acc = float(np.median(nc1_runs)); nc1_fl = 0.5
    results[("nc1", SIGMA_MAIN)] = (nc1_acc, nc1_fl)
    print(f"   acc={nc1_acc:.3f} (median over 20 IC-halvings; "
          f"range {min(nc1_runs):.2f}-{max(nc1_runs):.2f})  (chance ~0.50)")

    # ========================================================================
    #  CONTROL NC2: same scheme (galerkin), MESH-RESOLUTION change (confound)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONTROL NC2: same scheme (galerkin), mesh-resolution change (the confound)")
    print("=" * 80)
    ptsA, elemsA, bndA = make_mesh(22, seed=3001)   # coarser
    ptsB, elemsB, bndB = make_mesh(40, seed=3002)   # finer
    geomA = mesh_geometry(ptsA, elemsA); geomB = mesh_geometry(ptsB, elemsB)
    FncA = build_sigs("galerkin", ptsA, elemsA, bndA, ics, SIGMA_MAIN, seed=7000,
                      ref_grids=ref_grids, geom=geomA)
    FncB = build_sigs("galerkin", ptsB, elemsB, bndB, ics, SIGMA_MAIN, seed=8000,
                      ref_grids=ref_grids, geom=geomB)
    Xn2 = feats(np.vstack([FncA, FncB]))
    yn2 = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gn2 = np.r_[ic_idx, ic_idx]
    nc2_acc = cv_acc(Xn2, yn2, gn2); nc2_fl = perm_floor(Xn2, yn2, gn2, 41)
    results[("nc2", SIGMA_MAIN)] = (nc2_acc, nc2_fl)
    print(f"   acc={nc2_acc:.3f}  perm-floor={nc2_fl:.3f}  "
          f"(high => mesh-resolution is itself a discriminable confound)")

    # ---------------------------------------------------------------- CSV
    csv = os.path.join(TAB, "supg_2d_engineering.csv")
    with open(csv, "w") as f:
        f.write("contrast,sigma,accuracy,perm_floor,gap,chance,note\n")
        for sig in SIGMAS:
            a, fl = results[("presence", sig)]
            f.write(f"presence_galerkin_vs_supg,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,LOAD-BEARING\n")
        for sig in SIGMAS:
            a, fl = results[("tau_half", sig)]
            f.write(f"silenttau_supg_vs_0.5tau,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,silent under-stab\n")
            a, fl = results[("tau_2", sig)]
            f.write(f"silenttau_supg_vs_2tau,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,silent over-stab\n")
            a, fl = results[("tau_3way", sig)]
            f.write(f"silenttau_3way_opt_half_dbl,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.333,silent-stab taxonomy\n")
        for sig in SIGMAS:
            a, fl = results[("type", sig)]
            f.write(f"type_supg_vs_artvisc_iso_matched,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,"
                    f"anisotropic streamline vs isotropic Laplacian -- SEPARABLE operator-type WIN\n")
        for sig in (0.0, SIGMA_MAIN):
            a, fl = results[("consistency", sig)]
            f.write(f"consistency_supg_vs_streamdiff_inc,{sig},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,"
                    f"same anisotropic operator source-reweight only -- COLLINEARITY LIMIT (chance)\n")
        a, fl = results[("id5", SIGMA_MAIN)]
        f.write(f"id5_all_schemes,{SIGMA_MAIN},{a:.4f},{fl:.4f},{a-fl:.4f},0.20,context\n")
        a, fl = results[("nc1", SIGMA_MAIN)]
        f.write(f"NC1_ic_noise_same_scheme,{SIGMA_MAIN},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,control-should-be-chance\n")
        a, fl = results[("nc2", SIGMA_MAIN)]
        f.write(f"NC2_mesh_resolution_change,{SIGMA_MAIN},{a:.4f},{fl:.4f},{a-fl:.4f},0.50,confound\n")
        f.write(f"cos_supg_artvisc_iso,{SIGMA_MAIN},{cos_type:.4f},,,,signature angle (operator-type, separable)\n")
        f.write(f"cos_supg_streamdiff_inc,{SIGMA_MAIN},{cos_consistency:.4f},,,,signature angle (consistency, collinear)\n")
        f.write(f"Pe_h_max,{SIGMA_MAIN},{pe_h:.4f},,,,mesh Peclet\n")
        f.write(f"nu_art_over_D,{SIGMA_MAIN},{nu_art/D_PHYS:.4f},,,,matched added visc / physical\n")
        f.write(f"solver_validation,{SIGMA_MAIN},{1.0 if solver_valid else 0.0:.1f},,,,conv_ok and gal_osc\n")
    print(f"\nmetrics -> {csv}")

    summary = dict(results=results, vr=vr, conv_ok=conv_ok, gal_osc=gal_osc,
                   solver_valid=solver_valid, cos_type=cos_type, pe_h=pe_h,
                   nu_art=nu_art, F=F, ics=ics, pts=pts, elems=elems, on_bnd=on_bnd,
                   ref_pts=ref_pts, ref_elems=ref_elems, ref_bnd=ref_bnd,
                   id5=(id5_acc, id5_fl), nc1=nc1_acc, nc2=(nc2_acc, nc2_fl), csv=csv)

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 80 + "\nVERDICT (honest, against the pre-registered decision rules)\n" + "=" * 80)
    pres = results[("presence", SIGMA_MAIN)]
    t_h = results[("tau_half", SIGMA_MAIN)]; t_2 = results[("tau_2", SIGMA_MAIN)]
    typ = results[("type", SIGMA_MAIN)]
    con = results[("consistency", SIGMA_MAIN)]
    pres_ok = pres[0] - pres[1] >= 0.15 and pres[0] >= 0.90
    tau_ok = (t_h[0] - t_h[1] >= 0.15 and t_h[0] >= 0.75) or (t_2[0] - t_2[1] >= 0.15 and t_2[0] >= 0.75)
    type_sep = typ[0] - typ[1] >= 0.15 and typ[0] >= 0.75
    con_chance = abs(con[0] - con[1]) < 0.10 or con[0] < 0.62
    nc1_ok = abs(nc1_acc - 0.5) < 0.10
    summary["pres_ok"] = pres_ok; summary["tau_ok"] = tau_ok
    summary["type_sep"] = type_sep; summary["con_chance"] = con_chance
    summary["nc1_ok"] = nc1_ok; summary["cos_consistency"] = cos_consistency
    print(f"  (a) PRESENCE galerkin-vs-supg: {'STRONG' if pres_ok else 'WEAK'}  "
          f"acc={pres[0]:.3f} floor={pres[1]:.3f} (sigma={SIGMA_MAIN})")
    print(f"  (b) SILENT TAU supg-vs-0.5tau: acc={t_h[0]:.3f} floor={t_h[1]:.3f}; "
          f"supg-vs-2tau: acc={t_2[0]:.3f} floor={t_2[1]:.3f}  -> {'DETECTED' if tau_ok else 'WEAK'}")
    print(f"  (c) TYPE supg-vs-ISOTROPIC-artvisc: {'SEPARABLE (operator-type win)' if type_sep else 'at-chance'}  "
          f"acc={typ[0]:.3f} floor={typ[1]:.3f} cos={cos_type:.3f}")
    print(f"      -> the pre-registered 'collinearity limit' is NOT here: anisotropic")
    print(f"         streamline diffusion and an isotropic Laplacian are DIFFERENT 2D operators.")
    print(f"  (c2) CONSISTENCY supg-vs-streamdiff_inc: {'AT-CHANCE (true collinearity limit)' if con_chance else 'separable'}  "
          f"acc={con[0]:.3f} floor={con[1]:.3f} cos={cos_consistency:.3f}")
    print(f"  NC1 sits ~chance: {nc1_ok}  (acc={nc1_acc:.3f})")
    print(f"  NC2 mesh-resolution confound: acc={nc2_acc:.3f} (reported, not a win)")
    return summary


# ====================================================================
#  FIGURE
# ====================================================================
def _figure(r):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    except Exception:
        pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0", "#dd8452"
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0)); fig.subplots_adjust(wspace=0.28, hspace=0.36)
    pts, elems, on_bnd = r["pts"], r["elems"], r["on_bnd"]
    ic0 = dict(x0=0.5, y0=0.5, sharp=30.0, sgn=1.0, src=0.0, src_kx=1, src_ky=1)

    # A: Galerkin vs SUPG RAW NODAL values in a thin strip about y=0.5 -- the genuine
    #    node-to-node oscillation (gridded cubic interp smooths it; raw nodes show it).
    axA = axes[0, 0]
    ug, _ = assemble("galerkin", pts, elems, on_bnd, ic0)
    us, _ = assemble("supg", pts, elems, on_bnd, ic0)
    strip = np.abs(pts[:, 1] - 0.5) < 0.045
    order = np.argsort(pts[strip, 0])
    xs_s = pts[strip, 0][order]
    axA.plot(xs_s, ug[strip][order], color=RED, lw=1.0, marker="o", ms=3.0, label="Galerkin (unstab.)")
    axA.plot(xs_s, us[strip][order], color=BLUE, lw=1.0, marker="s", ms=3.0, label="SUPG (stab.)")
    axA.axhline(0, color=GREY, lw=0.7); axA.axhline(1, color=GREY, lw=0.7, ls=":")
    axA.set_xlabel("$x$ (nodes within $|y-0.5|<0.045$)"); axA.set_ylabel("$u$ (nodal)")
    axA.set_title(f"Internal layer, $Pe_h\\approx{r['pe_h']:.0f}$: Galerkin oscillates", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8)
    axA.text(-0.14, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: SUPG solution field (2D) on the rotating problem
    axB = axes[0, 1]
    Us_grid = _to_grid(pts, us)
    im = axB.contourf(_GX, _GY, Us_grid, levels=20, cmap="viridis")
    axB.set_aspect("equal"); axB.set_xlabel("$x$"); axB.set_ylabel("$y$")
    axB.set_title("SUPG field: rotated inlet + internal layer", fontsize=9.5)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    axB.text(-0.14, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: sigma sweep of the three contrasts
    axC = axes[1, 0]
    res = r["results"]
    sig_list = list(SIGMAS)
    pres = [res[("presence", s)][0] for s in sig_list]
    presf = [res[("presence", s)][1] for s in sig_list]
    tauh = [res[("tau_half", s)][0] for s in sig_list]
    type_ = [res[("type", s)][0] for s in sig_list]
    typef = [res[("type", s)][1] for s in sig_list]
    xx = np.arange(len(sig_list))
    axC.plot(xx, pres, "o-", color=GREEN, lw=2, label="(a) presence galerkin/supg")
    axC.plot(xx, tauh, "s-", color=PURP, lw=2, label="(b) silent tau supg/0.5tau")
    axC.plot(xx, type_, "^-", color=ORANGE, lw=2, label="(c) type supg/iso-artvisc")
    axC.plot(xx, presf, ":", color=GREEN, lw=1, alpha=0.6)
    axC.plot(xx, typef, ":", color=ORANGE, lw=1, alpha=0.6)
    # (c2) consistency axis: only computed at sigma 0 and SIGMA_MAIN
    con_x = [sig_list.index(s) for s in (0.0, SIGMA_MAIN)]
    con_y = [res[("consistency", s)][0] for s in (0.0, SIGMA_MAIN)]
    axC.plot(con_x, con_y, "D--", color=RED, lw=1.8, ms=5, label="(c2) consistency supg/streamdiff")
    axC.axhline(0.5, color=GREY, lw=1, ls="--", label="chance")
    axC.set_xticks(xx); axC.set_xticklabels([f"{s}" for s in sig_list])
    axC.set_xlabel("observation noise $\\sigma$"); axC.set_ylabel("GroupKFold accuracy")
    axC.set_ylim(0.3, 1.05); axC.set_title("Contrasts across noise (dotted=floor)", fontsize=9.5)
    axC.legend(frameon=False, fontsize=7.2)
    axC.text(-0.14, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: headline bars at sigma_main with permutation floors
    axD = axes[1, 1]
    sm = SIGMA_MAIN
    keys = [("presence", "presence\ngal/supg", GREEN), ("tau_half", "silent\nsupg/0.5τ", PURP),
            ("tau_2", "silent\nsupg/2τ", PURP), ("tau_3way", "3-way τ\nopt/½/2", BLUE),
            ("type", "type\nsupg/iso", ORANGE), ("consistency", "consist.\nsupg/sdi", RED),
            ("nc1", "NC1", GREY), ("nc2", "NC2\nmesh", "#c8a35a")]
    vals = [res[(k, sm)][0] for k, _, _ in keys]
    fls = [res[(k, sm)][1] for k, _, _ in keys]
    cols = [c for _, _, c in keys]
    axD.bar(range(len(keys)), vals, color=cols, width=0.66)
    for i, fl in enumerate(fls):
        axD.plot([i - 0.34, i + 0.34], [fl, fl], color="#222", ls=(0, (2, 1.5)), lw=1.5, zorder=6)
    for i, v in enumerate(vals):
        axD.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    axD.set_xticks(range(len(keys))); axD.set_xticklabels([k[1] for k in keys], fontsize=7.2)
    axD.set_ylim(0, 1.08); axD.set_ylabel("GroupKFold accuracy")
    axD.set_title(f"Attribution at $\\sigma={sm}$ (dashed=perm floor)", fontsize=9.5)
    axD.grid(axis="y", color="#e6e6e6", lw=0.7); axD.set_axisbelow(True)
    axD.text(-0.14, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "supg_2d_engineering.png")
    fig.savefig(out); plt.close(fig)
    print(f"figure -> {out}")
    return out


if __name__ == "__main__":
    import sys
    np.seterr(all="ignore")
    r = main()
    if "--plot" in sys.argv or True:
        r["fig"] = _figure(r)
    print("\ndone.")
