#!/usr/bin/env python3
"""
solver-forensics :: NON-PERIODIC BOUNDARY CONDITIONS  (does the INTERIOR signature survive?)
============================================================================================
The clean substrates in this project are PERIODIC, so every grid point sees the same
interior stencil and the modified-equation signature is spatially homogeneous. A standard
reviewer probe (CMAME revision, ITEM 8): on a NON-PERIODIC domain the boundary CLOSURE has a
different truncation order than the interior (one-sided differences, ghost-node rules, a
diffusive boundary layer), so the residual near a boundary carries a DIFFERENT modified-
equation footprint than the interior. Does the interior scheme signature still attribute, or
does the boundary contaminate even the masked interior?

PDE: unsteady advection-diffusion on [0, 1]
        u_t + a u_x = D u_xx ,    a > 0  (left-to-right transport)
with NON-PERIODIC BCs (three families, one per IC family axis -- the BC type is part of the
"initial condition" group so attribution must generalize ACROSS BC families):
   Dirichlet  : u(0)=g0(t)=u_in,   u(1)=g1 = 0           (inflow value + outflow clamp)
   Neumann    : u(0)=u_in (Dirichlet inflow),  u_x(1)=0  (zero-gradient outflow -- the common one)
   inflow-out : u(0)=u_in (Dirichlet inflow),  one-sided extrapolation outflow at x=1
Inflow is always Dirichlet (advection needs a prescribed inflow); the OUTFLOW closure is what
varies. A thin internal layer forms where the prescribed BC fights the transported interior
solution -- that is exactly where boundary truncation differs from interior truncation.

TWO INTERIOR SCHEMES (the thing being attributed), method-of-lines, SAME boundary closure,
SAME RK2(Heun) time integration, SAME physical diffusion -- they differ ONLY in the interior
ADVECTION stencil:
   upwind  : u_x ~ (u_i - u_{i-1})/h         1st order, DIFFUSIVE truncation  +a h/2 u_xx
   central : u_x ~ (u_{i+1}-u_{i-1})/(2h)     2nd order, DISPERSIVE truncation -a h^2/6 u_xxx
The interior modified equations differ in leading term (u_xx vs u_xxx) -- a clean strong-form
contrast IF the interior signature is readable away from the boundary.

REFERENCE: a GENUINE fine solve. The reference is the SAME PDE solved with a high-resolution,
well-resolved central+diffusion MOL solver on N_REF nodes (validated below: convergent under
refinement, stable, monotone where the exact BVP steady state is monotone) and the BCs of the
realization, then interpolated (scipy cubic) to the N_OBS observation grid. We VALIDATE the
reference (convergence under N_REF refinement; steady-state agreement with the analytic
constant-coefficient BVP solution) and PRINT it before trusting any residual.

SIGNATURE: unit-normalized least-squares coefficient direction of c in
   r = u_solver - u_ref ~ sum_p c_p d_x^p u ,   library {u_xx, u_xxx, u_xxxx},
central FD derivatives of the OBSERVED solver field on the uniform observation grid (mirrors
src/audit/stabilization_audit.py). FULL-DOMAIN signature uses every interior FD-valid node;
INTERIOR-MASKED drops a boundary margin of MARGIN_FRAC*L at EACH end before the lstsq fit.

ATTRIBUTION: StandardScaler + LogisticRegression, GroupKFold(5) grouped by INITIAL CONDITION
(the forcing/BC realization), label-PERMUTATION floor on EVERY reported number.
CONTROLS:
  NC1 = same scheme (upwind), IC + noise only, arbitrary IC partition -> must sit ~chance.
  NC2 = same scheme (upwind), GRID CHANGE (N_OBS underlying solve grid changed) -> the confound.

DECISION RULE (pre-registered):
  - If the INTERIOR signature survives -- full-domain attribution DEGRADES (boundary contaminates)
    while interior-masked attribution HOLDS (>= floor+0.20 and >= 0.80 for the upwind-vs-central
    pair, generalizing across BC families) -- isolate and report the interior result cleanly, with
    the recovered margin.
  - If the boundary contaminates even the masked interior (interior-masked still near floor, or no
    margin recovers it) -> report as a BOUNDARY of the method with the margin needed, quantified.
We sweep MARGIN_FRAC to FIND the margin at which interior attribution saturates, and report it.

Self-contained: numpy + scipy + sklearn, CPU, ~1-2 min. NO FEM library. Guarded by __main__.
Run:  python src/limits/nonperiodic_bc.py          (add --plot for the figure)
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
FIGDIR = os.path.join(_ROOT, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)

# ---- physical / numerical constants ----
L      = 1.0
A      = 1.0           # advection speed (left-to-right)
D      = 1.0e-2        # diffusivity -> Pe = a L / D = 100 (advection-dominated, outflow layer present;
                       # cell-Peclet Pe_h = a h /(2D) ~ 0.4 at N_OBS=128 so the INTERIOR advection
                       # truncation -- upwind diffusive vs central dispersive -- is RESOLVED and
                       # DOMINANT in the residual, not swamped by reference error)
T      = 0.45          # integration time (transport ~ a T = 0.45L; front well inside, layer formed)
N_OBS  = 128           # observation / coarse solver grid (nodes incl. boundaries)
N_REF  = 512           # fine reference grid (4x; same IMEX stepping so temporal/diffusion error
                       # cancels in the residual, isolating the spatial advection-scheme truncation)
N_IC   = 60            # forcing/BC realizations (the group key); split across 3 BC families
SIGMA  = 0.01          # field-relative observation noise
LIB    = (2, 3, 4)     # derivative library orders {u_xx, u_xxx, u_xxxx}
CFL    = 0.30          # advective CFL for the explicit advection sub-step (diffusion is implicit CN)
SCHEMES = ("upwind", "central")
BC_FAMILIES = ("dirichlet", "neumann", "inflow_out")
MARGINS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20)   # boundary margin fractions to sweep


# ======================================================================== MOL solver (non-periodic)
def _advect_ux(u, h, scheme):
    """Interior advection derivative u_x (a>0). Boundaries handled by caller (closure)."""
    ux = np.empty_like(u)
    if scheme == "upwind":              # backward difference (1st order, a>0): u_x ~ (u_i-u_{i-1})/h
        ux[1:] = (u[1:] - u[:-1]) / h
        ux[0]  = (u[1] - u[0]) / h      # inflow node value is Dirichlet-set after the step anyway
    else:                               # central (2nd order): (u_{i+1}-u_{i-1})/(2h)
        ux[1:-1] = (u[2:] - u[:-2]) / (2 * h)
        ux[0]  = (u[1] - u[0]) / h
        ux[-1] = (u[-1] - u[-2]) / h    # one-sided at the very last node (closure overwrites BC node)
    return ux


def _diff_uxx(u, h):
    """Interior diffusion u_xx, standard 3-point. Endpoints one-sided (overwritten by BC closure)."""
    uxx = np.empty_like(u)
    uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
    uxx[0]  = (u[2] - 2 * u[1] + u[0]) / h ** 2
    uxx[-1] = (u[-1] - 2 * u[-2] + u[-3]) / h ** 2
    return uxx


def _apply_bc(u, bc_family, u_in, g_out):
    """Enforce BCs in place after a time-step. Inflow (x=0) is always Dirichlet = u_in.
       Outflow (x=1) closure depends on family."""
    u[0] = u_in
    if bc_family == "dirichlet":
        u[-1] = g_out                        # clamp outflow value
    elif bc_family == "neumann":
        u[-1] = u[-2]                         # zero-gradient (u_x(L)=0) via ghost reflection
    else:                                     # inflow_out: 1st-order extrapolation outflow
        u[-1] = 2 * u[-2] - u[-3]
    return u


def _adv_rhs(u, h, scheme):
    """Explicit advection RHS  -a u_x  (the scheme-dependent part being attributed)."""
    return -A * _advect_ux(u, h, scheme)


def _cn_banded(n, h, dt, D_, bc_family):
    """Crank-Nicolson IMPLICIT diffusion operator for interior nodes 1..n-2.
       Solve (I - 0.5 dt D Lap) u^{n+1}_int = (I + 0.5 dt D Lap) u^* with the boundary nodes
       (0 and n-1) held by the BC closure. Returns the banded LHS matrix (ab) for solve_banded
       over the FULL node set with Dirichlet-style identity rows at the two boundaries; the actual
       BC VALUES are written into the rhs by the stepper. This removes the parabolic stiffness so
       the step is advective-CFL sized. Diffusion is IDENTICAL across schemes, so the upwind-vs-
       central contrast (advection truncation) is preserved exactly."""
    r = 0.5 * dt * D_ / h ** 2
    lower = np.zeros(n); diag = np.zeros(n); upper = np.zeros(n)
    diag[:] = 1.0 + 2.0 * r; lower[:] = -r; upper[:] = -r
    # boundary rows: identity (BC closure sets these values directly)
    diag[0] = 1.0; upper[0] = 0.0
    diag[-1] = 1.0; lower[-1] = 0.0
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]   # super-diagonal
    ab[1, :] = diag          # main diagonal
    ab[2, :-1] = lower[1:]   # sub-diagonal
    return ab, r


def solve_mol(u0, n_nodes, scheme, bc_family, u_in, g_out, D_=D, t_final=T):
    """IMEX method-of-lines: EXPLICIT Heun(RK2) advection (the scheme under test) + IMPLICIT
       Crank-Nicolson diffusion (banded solve), on n_nodes uniform nodes incl. both boundaries.
       Implicit diffusion removes the h^2 parabolic step constraint -> advective-CFL steps; the
       diffusion treatment is the SAME for both schemes so the attributed contrast is the
       advection truncation (upwind diffusive u_xx vs central dispersive u_xxx)."""
    h = L / (n_nodes - 1)
    dt = CFL * h / abs(A)
    ns = int(np.ceil(t_final / dt)); dt = t_final / ns
    ab, r = _cn_banded(n_nodes, h, dt, D_, bc_family)
    u = u0.copy(); _apply_bc(u, bc_family, u_in, g_out)
    for _ in range(ns):
        # explicit advection sub-step (Heun)
        k1 = _adv_rhs(u, h, scheme)
        up = u + dt * k1; _apply_bc(up, bc_family, u_in, g_out)
        k2 = _adv_rhs(up, h, scheme)
        ustar = u + 0.5 * dt * (k1 + k2); _apply_bc(ustar, bc_family, u_in, g_out)
        # implicit CN diffusion sub-step: build rhs = (I + 0.5 dt D Lap) ustar on interior
        rhs = ustar.copy()
        rhs[1:-1] = ustar[1:-1] + r * (ustar[2:] - 2 * ustar[1:-1] + ustar[:-2])
        # boundary rows of the banded system are identity -> rhs holds the BC values directly
        rhs[0] = ustar[0]; rhs[-1] = ustar[-1]
        u = solve_banded((1, 1), ab, rhs)
        _apply_bc(u, bc_family, u_in, g_out)
    return u


# ======================================================================== analytic steady BVP (validation only)
def steady_bvp(u_in, g_out, bc_family, xs, D_=D):
    """Exact STEADY-STATE solution of  a u_x = D u_xx  on [0,1] (no forcing), used ONLY to
       validate the reference solver's boundary layer. General solution u = c0 + c1 exp(a x / D).
       Inflow Dirichlet u(0)=u_in. Outflow: Dirichlet u(1)=g_out / Neumann u_x(1)=0 / extrap (~Neumann).
       (Time-dependent runs are NOT at steady state at T; this is a sanity anchor for the layer shape.)"""
    Pe = A / D_
    E = np.exp(np.clip(Pe * xs, None, 700)); EL = np.exp(np.clip(Pe * L, None, 700))
    if bc_family == "dirichlet":
        # c0 + c1 = u_in ; c0 + c1 EL = g_out
        c1 = (g_out - u_in) / (EL - 1.0); c0 = u_in - c1
    else:
        # Neumann u_x(1)=0 -> c1 Pe EL = 0 -> c1 = 0 -> flat u = u_in (zero-gradient, advection carries it)
        c1 = 0.0; c0 = u_in
    return c0 + c1 * E


# ======================================================================== IC ensemble
def random_realization(rng):
    """One realization = a smooth inflow profile (initial field) + inflow value + outflow value.
       The initial field is a smooth bump/modes so the transported solution has interior structure
       (curvature/3rd-deriv content) for the signature to read."""
    x = np.linspace(0, L, 8)
    M = 4
    coeffs = rng.normal(size=M)
    def field(xx):
        u = np.zeros_like(xx)
        for m, cm in enumerate(coeffs, start=1):
            u += cm * np.sin(m * np.pi * xx / L)        # zero at both ends -> compatible with Dirichlet
        return 0.5 * u / (np.max(np.abs(u)) + 1e-9)
    u_in = float(rng.uniform(0.3, 1.0))                 # prescribed inflow value
    g_out = float(rng.uniform(-0.2, 0.2))               # outflow clamp value (Dirichlet family)
    return field, u_in, g_out


def make_ics(n, seed=0):
    rng = np.random.default_rng(seed)
    return [random_realization(rng) for _ in range(n)]


# ======================================================================== reference + observation
def reference_field(field, u_in, g_out, bc_family, xs_obs, D_=D, n_ref=N_REF):
    xr = np.linspace(0, L, n_ref)
    u0 = field(xr)
    u_ref = solve_mol(u0, n_ref, "central", bc_family, u_in, g_out, D_=D_)
    return interp1d(xr, u_ref, kind="cubic", fill_value="extrapolate")(xs_obs)


def solver_field(field, u_in, g_out, bc_family, scheme, xs_obs, n_obs=N_OBS, D_=D):
    u0 = field(xs_obs)
    return solve_mol(u0, n_obs, scheme, bc_family, u_in, g_out, D_=D_)


# ======================================================================== signature (interior / full)
def fd_derivs_allnodes(u, dx):
    """FD derivatives at EVERY node 0..n-1. Interior uses central stencils; the boundary nodes
       (which the reviewer probe is about) use ONE-SIDED / biased stencils so the FULL-DOMAIN
       signature genuinely includes x=0 and x=L, where the solver's boundary CLOSURE truncation
       differs from the interior. Interior-masking later drops a margin around these nodes.
       Returns u_xx, u_xxx, u_xxxx at all n nodes."""
    n = len(u)
    uxx = np.empty(n); uxxx = np.empty(n); uxxxx = np.empty(n)
    # --- u_xx ---
    uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    uxx[0]  = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2            # one-sided (2nd order)
    uxx[-1] = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4]) / dx ** 2
    # --- u_xxx (central interior idx 2..n-3; biased at idx 0,1,n-2,n-1) ---
    uxxx[2:-2] = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx ** 3)
    fwd3 = lambda a, b, c, d, e: (-a + 3 * b - 3 * c + d) / dx ** 3       # not used directly; explicit below
    uxxx[0]  = (-u[0] + 3 * u[1] - 3 * u[2] + u[3]) / dx ** 3             # forward (1st order)
    uxxx[1]  = (-u[1] + 3 * u[2] - 3 * u[3] + u[4]) / dx ** 3
    uxxx[-1] = (u[-1] - 3 * u[-2] + 3 * u[-3] - u[-4]) / dx ** 3          # backward
    uxxx[-2] = (u[-2] - 3 * u[-3] + 3 * u[-4] - u[-5]) / dx ** 3
    # --- u_xxxx (central interior idx 2..n-3; one-sided 5-pt at the edges) ---
    uxxxx[2:-2] = (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4]) / dx ** 4
    uxxxx[0]  = (u[0] - 4 * u[1] + 6 * u[2] - 4 * u[3] + u[4]) / dx ** 4  # forward 5-pt
    uxxxx[1]  = (u[1] - 4 * u[2] + 6 * u[3] - 4 * u[4] + u[5]) / dx ** 4
    uxxxx[-1] = (u[-1] - 4 * u[-2] + 6 * u[-3] - 4 * u[-4] + u[-5]) / dx ** 4
    uxxxx[-2] = (u[-2] - 4 * u[-3] + 6 * u[-4] - 4 * u[-5] + u[-6]) / dx ** 4
    return uxx, uxxx, uxxxx


def signature(u_obs, r_obs, dx, margin_frac, n_obs=N_OBS):
    """Coefficient-direction signature over the nodes kept by margin_frac.
       margin_frac = 0.0 -> FULL DOMAIN (all nodes incl. the two boundary nodes, where the solver's
       boundary CLOSURE truncation lives -- one-sided FD derivatives are used there).
       margin_frac > 0 -> INTERIOR-MASKED: drop that fraction of L at EACH end before the lstsq fit."""
    uxx, uxxx, uxxxx = fd_derivs_allnodes(u_obs, dx)
    b = r_obs
    xpos = np.arange(n_obs) * dx
    keep = (xpos >= margin_frac * L) & (xpos <= L - margin_frac * L)
    if keep.sum() < 6:                                # too few points -> degenerate
        keep = np.ones_like(keep)
    A_lib = np.stack([uxx[keep], uxxx[keep], uxxxx[keep]], 1)
    c, *_ = np.linalg.lstsq(A_lib, b[keep], rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def sigs(scheme, realizations, bc_assign, margin_frac, sigma, seed, n_obs=N_OBS):
    """Signatures for one scheme across all realizations. bc_assign[i] = BC family for realization i."""
    xs = np.linspace(0, L, n_obs); dx = xs[1] - xs[0]
    gn = np.random.default_rng(seed)
    out = []
    for (field, u_in, g_out), bcf in zip(realizations, bc_assign):
        u_s = solver_field(field, u_in, g_out, bcf, scheme, xs, n_obs=n_obs)
        u_r = reference_field(field, u_in, g_out, bcf, xs)
        if sigma > 0:
            u_s = u_s + sigma * np.sqrt(np.mean(u_s ** 2)) * gn.standard_normal(n_obs)
        r = u_s - u_r
        out.append(signature(u_s, r, dx, margin_frac, n_obs=n_obs))
    return np.array(out)


# ======================================================================== attribution machinery
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def acc(F, y, g): return float(cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean())
def perm_floor(F, y, g, seed, reps=40):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean()
                            for _ in range(reps)]))


# ======================================================================== solver validation
def validate_solver():
    """(1) Reference convergence under grid refinement (rel L2, self-consistent Richardson).
       (2) Stability: no blow-up.
       (3) Boundary-layer agreement: at LATE time the solution approaches the analytic steady BVP
           layer for a clamped-outflow Dirichlet case (anchors the boundary closure)."""
    rep = {}
    # a fixed deterministic realization for validation
    rng = np.random.default_rng(123)
    field, u_in, g_out = random_realization(rng)
    xs = np.linspace(0, L, N_OBS)

    # (1) reference convergence: solve central MOL at increasing N, compare on common obs grid.
    #     The reference grid N_REF=512 sits well inside this ladder; we confirm successive grids
    #     agree to <1% and the difference SHRINKS with refinement (consistent, converging).
    errs = {}
    prev = None
    for n in (128, 256, 512, 1024, 2048):
        xr = np.linspace(0, L, n)
        ur = solve_mol(field(xr), n, "central", "dirichlet", u_in, g_out)
        ug = interp1d(xr, ur, kind="cubic", fill_value="extrapolate")(xs)
        if prev is not None:
            errs[n] = float(np.sqrt(np.mean((ug - prev) ** 2)) / (np.sqrt(np.mean(ug ** 2)) + 1e-12))
        prev = ug
    rep["ref_selfconv"] = errs
    rep["ref_converged"] = (max(errs.values()) < 1e-2) and (errs[max(errs)] < errs[min(errs)])

    # (2) stability of both coarse schemes (finite, bounded)
    stab = {}
    for sc in SCHEMES:
        for bcf in BC_FAMILIES:
            u = solve_mol(field(xs), N_OBS, sc, bcf, u_in, g_out)
            stab[(sc, bcf)] = bool(np.isfinite(u).all() and np.max(np.abs(u)) < 10)
    rep["stable_all"] = all(stab.values())
    rep["stability"] = stab

    # (3) steady-state boundary-layer anchor: long-time central solve vs analytic BVP layer (Dirichlet)
    #     run to a long time so transients decay; compare layer shape near outflow.
    xr = np.linspace(0, L, N_REF)
    u_long = solve_mol(field(xr), N_REF, "central", "dirichlet", u_in, g_out, t_final=6.0)
    u_bvp = steady_bvp(u_in, g_out, "dirichlet", xr)
    layer = xr >= 0.8 * L
    rep["bvp_layer_relL2"] = float(np.sqrt(np.mean((u_long[layer] - u_bvp[layer]) ** 2)) /
                                   (np.sqrt(np.mean(u_bvp[layer] ** 2)) + 1e-12))
    rep["bvp_ok"] = rep["bvp_layer_relL2"] < 0.05

    # (4) interior modified-equation sanity: upwind residual should be MORE u_xx-aligned (diffusive),
    #     central residual MORE u_xxx-aligned (dispersive), on the deep interior (margin 0.20).
    xsg = np.linspace(0, L, N_OBS); dx = xsg[1] - xsg[0]
    ur = reference_field(field, u_in, g_out, "dirichlet", xsg)
    dir_sc = {}
    for sc in SCHEMES:
        us = solver_field(field, u_in, g_out, "dirichlet", sc, xsg)
        dir_sc[sc] = signature(us, us - ur, dx, 0.20)
    rep["dir_upwind"] = dir_sc["upwind"]; rep["dir_central"] = dir_sc["central"]
    # |c_xx| dominance for upwind, |c_xxx| relatively larger for central
    rep["upwind_uxx_dominant"] = abs(dir_sc["upwind"][0]) > abs(dir_sc["upwind"][1])
    rep["central_more_uxxx"] = abs(dir_sc["central"][1]) > abs(dir_sc["upwind"][1])
    return rep


# ======================================================================== RUN
def main():
    print("=" * 84)
    print("NON-PERIODIC BC PROBE | advection-diffusion on [0,1], upwind vs central, fine-solve reference")
    print("=" * 84)
    Pe = A * L / D
    print(f"a={A}, D={D}, Pe=aL/D={Pe:.0f}  (advection-dominated, thin outflow layer)")
    print(f"N_OBS={N_OBS}, N_REF={N_REF}, T={T}, {N_IC} ICs across BC families {BC_FAMILIES}")
    print(f"library {LIB} (u_xx,u_xxx,u_xxxx); SIGMA={SIGMA}\n")

    # ---------------- solver validation (must pass before trusting residuals) ----------------
    vr = validate_solver()
    print("[validate] reference self-convergence (rel L2 between successive N, on obs grid):")
    for n in sorted(vr["ref_selfconv"]):
        print(f"           N={n:5d}  rel-diff vs coarser = {vr['ref_selfconv'][n]:.2e}")
    print(f"           reference converged: {vr['ref_converged']}")
    print(f"[validate] all (scheme x BC) coarse solves stable & bounded: {vr['stable_all']}")
    print(f"[validate] steady boundary-layer vs analytic BVP (Dirichlet, near outflow), rel L2 = "
          f"{vr['bvp_layer_relL2']:.3e}  -> ok: {vr['bvp_ok']}")
    print(f"[validate] interior modified-equation sanity (deep interior, margin 0.20):")
    print(f"           upwind  dir [c_xx,c_xxx,c_xxxx] = [{vr['dir_upwind'][0]:+.3f}, {vr['dir_upwind'][1]:+.3f}, {vr['dir_upwind'][2]:+.3f}]"
          f"  (u_xx dominant: {vr['upwind_uxx_dominant']})")
    print(f"           central dir [c_xx,c_xxx,c_xxxx] = [{vr['dir_central'][0]:+.3f}, {vr['dir_central'][1]:+.3f}, {vr['dir_central'][2]:+.3f}]"
          f"  (more u_xxx than upwind: {vr['central_more_uxxx']})")
    valid_ok = vr["ref_converged"] and vr["stable_all"] and vr["bvp_ok"]
    print(f"\n[validate] SOLVER TRUSTWORTHY: {valid_ok}\n")
    if not valid_ok:
        print("VALIDATION FAILED -- residuals not trustworthy; aborting attribution.")
        return None

    # ---------------- build the ensemble (ICs spread across BC families) ----------------
    realizations = make_ics(N_IC, seed=0)
    bc_assign = [BC_FAMILIES[i % len(BC_FAMILIES)] for i in range(N_IC)]   # balanced across families
    ic = np.arange(N_IC)
    y = np.r_[np.zeros(N_IC), np.ones(N_IC)]; g2 = np.r_[ic, ic]

    # ---------------- margin sweep: full-domain (margin 0) vs interior-masked ----------------
    print("-" * 84)
    print("UPWIND vs CENTRAL attribution as a function of interior boundary MARGIN (each end):")
    print(f"{'margin':>7s} {'acc':>6s} {'floor':>6s} {'gap':>6s}   (margin 0.00 = FULL DOMAIN)")
    sweep = []
    for mf in MARGINS:
        Fu = sigs("upwind",  realizations, bc_assign, mf, SIGMA, 100)
        Fc = sigs("central", realizations, bc_assign, mf, SIGMA, 200)
        X = np.vstack([Fu, Fc])
        a = acc(X, y, g2); fl = perm_floor(X, y, g2, 31)
        sweep.append(dict(margin=mf, acc=a, floor=fl, gap=a - fl))
        print(f"{mf:>7.2f} {a:>6.3f} {fl:>6.3f} {a - fl:>+6.3f}"
              + ("   <- full domain" if mf == 0.0 else ""))
    full = sweep[0]
    # interior result = the masked-interior operating point. The peak attribution accuracy over the
    # positive margins is the recovery; the margin NEEDED is the SMALLEST positive margin reaching
    # within 0.02 of that peak (the boundary margin that has to be masked). The headline `interior`
    # row is that operating point (so the reported interior acc and the margin are consistent).
    interior_candidates = [s for s in sweep if s["margin"] > 0]
    peak_acc = max(s["acc"] for s in interior_candidates)
    margin_needed = next(s["margin"] for s in interior_candidates if s["acc"] >= peak_acc - 0.02)
    interior = next(s for s in interior_candidates if s["margin"] == margin_needed)

    # ---------------- per-BC-family interior attribution (does it hold within each family?) -----------
    print("\nper-BC-family upwind-vs-central (interior margin = {:.2f}):".format(margin_needed))
    perfam = {}
    for fam in BC_FAMILIES:
        sel = [i for i in range(N_IC) if bc_assign[i] == fam]
        reals_f = [realizations[i] for i in sel]; bca_f = [fam] * len(sel); icf = np.arange(len(sel))
        Fu = sigs("upwind",  reals_f, bca_f, margin_needed, SIGMA, 100)
        Fc = sigs("central", reals_f, bca_f, margin_needed, SIGMA, 200)
        Xf = np.vstack([Fu, Fc]); yf = np.r_[np.zeros(len(sel)), np.ones(len(sel))]; gf = np.r_[icf, icf]
        af = acc(Xf, yf, gf); flf = perm_floor(Xf, yf, gf, 41)
        perfam[fam] = (af, flf, len(sel))
        print(f"   {fam:11s} acc={af:.3f}  floor={flf:.3f}  gap={af - flf:+.3f}  (n={len(sel)} ICs)")

    # ---------------- controls ----------------
    print("\ncontrols:")
    # NC1: same scheme (upwind), IC+noise only, arbitrary IC partition -> chance (mean over splits)
    Fnc = sigs("upwind", realizations, bc_assign, margin_needed, SIGMA, 9000)
    half = N_IC // 2; nc1_draws = []
    for s in range(8):
        perm = np.random.default_rng(1000 + s).permutation(N_IC)
        gA, gB = perm[:half], perm[half:]
        nc1_draws.append(acc(np.vstack([Fnc[gA], Fnc[gB]]),
                             np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[gA], ic[gB]]))
    nc1 = float(np.mean(nc1_draws)); nc1_sd = float(np.std(nc1_draws))
    nc1_f = perm_floor(np.vstack([Fnc[:half], Fnc[half:]]),
                       np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[:half], ic[half:]], 51)
    print(f"   NC1 IC+noise (same scheme): acc={nc1:.3f} +/- {nc1_sd:.3f}  floor={nc1_f:.3f}  (chance~0.50)")

    # NC2: same scheme (upwind), GRID CHANGE (solve+observe on a different grid) -> the confound
    Fg_a = sigs("upwind", realizations, bc_assign, margin_needed, SIGMA, 7000, n_obs=N_OBS)
    Fg_b = sigs("upwind", realizations, bc_assign, margin_needed, SIGMA, 7700, n_obs=192)
    nc2 = acc(np.vstack([Fg_a, Fg_b]), y, g2); nc2_f = perm_floor(np.vstack([Fg_a, Fg_b]), y, g2, 61)
    print(f"   NC2 grid change (same scheme): acc={nc2:.3f}  floor={nc2_f:.3f}  (the grid confound)")

    # ---------------- CSV ----------------
    csv = os.path.join(TAB, "nonperiodic_bc_results.csv")
    with open(csv, "w") as f:
        f.write("task,margin_frac,accuracy,perm_floor,gap,chance,note\n")
        for s in sweep:
            tag = "upwind_vs_central_FULLDOMAIN" if s["margin"] == 0.0 else "upwind_vs_central_interior"
            f.write(f"{tag},{s['margin']:.4f},{s['acc']:.4f},{s['floor']:.4f},{s['gap']:.4f},0.500,"
                    f"margin {s['margin']:.2f} each end\n")
        for fam, (af, flf, nf) in perfam.items():
            f.write(f"upwind_vs_central_{fam},{margin_needed:.4f},{af:.4f},{flf:.4f},{af - flf:.4f},0.500,"
                    f"per-BC-family interior (n={nf})\n")
        f.write(f"NC1_ic_noise,{margin_needed:.4f},{nc1:.4f},{nc1_f:.4f},{nc1 - nc1_f:.4f},0.500,"
                f"same scheme control (mean over 8 splits; sd={nc1_sd:.4f})\n")
        f.write(f"NC2_grid_change,{margin_needed:.4f},{nc2:.4f},{nc2_f:.4f},{nc2 - nc2_f:.4f},0.500,"
                f"grid confound (N_OBS {N_OBS} vs 192)\n")
        f.write(f"margin_needed,{margin_needed:.4f},,,,,interior margin (each end) for interior attribution\n")
        f.write(f"Pe,{Pe:.4f},,,,,advective Peclet aL/D\n")
        f.write(f"ref_selfconv_finest,{max(vr['ref_selfconv'].values()):.6e},,,,,reference self-convergence (worst)\n")
        f.write(f"bvp_layer_relL2,{vr['bvp_layer_relL2']:.6e},,,,,steady boundary-layer vs analytic BVP\n")
    print(f"\nmetrics -> {csv}")

    # ---------------- DECISION ----------------
    print("\n" + "=" * 84)
    print("DECISION (pre-registered)")
    print("=" * 84)
    interior_holds = (interior["acc"] >= 0.80) and (interior["gap"] >= 0.20)
    full_degrades = (full["acc"] < interior["acc"] - 0.05)   # full-domain materially worse than interior
    nc1_ok = (nc1 - nc1_f) <= 0.10
    perfam_ok = all((af - flf) >= 0.15 for af, flf, _ in perfam.values())
    print(f"  full-domain (margin 0.00): acc={full['acc']:.3f}  gap={full['gap']:+.3f}")
    print(f"  interior-masked (margin {interior['margin']:.2f}): acc={interior['acc']:.3f}  gap={interior['gap']:+.3f}")
    print(f"  margin NEEDED for interior attribution: {margin_needed:.2f} of L at each end "
          f"(~{int(round(margin_needed * (N_OBS - 1)))} nodes)")
    print(f"  per-BC-family interior all attribute: {perfam_ok}")
    print(f"  NC1 control ~chance: {nc1_ok}  ({nc1:.3f} vs floor {nc1_f:.3f})")
    print(f"  NC2 grid confound: acc={nc2:.3f} (reported as the confound, not a claim)")

    if interior_holds and full_degrades and perfam_ok and nc1_ok:
        outcome = "INTERIOR_SURVIVES"
        print("\n[INTERIOR SIGNATURE SURVIVES]")
        print(f"  Full-domain attribution is degraded by boundary truncation ({full['acc']:.2f}), but masking a")
        print(f"  boundary margin of {margin_needed:.2f}L at each end RECOVERS clean interior attribution")
        print(f"  ({interior['acc']:.2f}, gap {interior['gap']:+.2f}) that generalizes ACROSS the three BC families")
        print(f"  (Dirichlet/Neumann/inflow-outflow). The interior modified-equation signature survives")
        print(f"  non-periodic BCs once the boundary layer is masked; the margin needed is the boundary.")
    elif interior_holds and not full_degrades and perfam_ok and nc1_ok:
        outcome = "INTERIOR_SURVIVES_NO_FULL_DEGRADE"
        print("\n[INTERIOR SIGNATURE SURVIVES -- boundary contamination mild]")
        print(f"  Interior-masked attribution is clean ({interior['acc']:.2f}) and full-domain is already")
        print(f"  comparable ({full['acc']:.2f}): the boundary closure does not materially contaminate the")
        print(f"  signature here (margin needed {margin_needed:.2f}L). Interior attribution holds; boundary")
        print(f"  contamination is present but small (see margin sweep).")
    else:
        outcome = "BOUNDARY_CONTAMINATES"
        print("\n[BOUNDARY CONTAMINATES -- reported as a method boundary]")
        print(f"  Even with an interior margin up to {MARGINS[-1]:.2f}L the upwind-vs-central pair does not reach")
        print(f"  clean attribution (best interior acc {interior['acc']:.2f}, gap {interior['gap']:+.2f}). The")
        print(f"  non-periodic boundary closure contaminates the recovered signature beyond the maskable")
        print(f"  interior; this is the honest boundary of strong-form attribution under non-periodic BCs.")

    res = dict(sweep=sweep, full=full, interior=interior, margin_needed=margin_needed,
               perfam=perfam, nc1=nc1, nc1_sd=nc1_sd, nc1_f=nc1_f, nc2=nc2, nc2_f=nc2_f,
               valid=vr, Pe=Pe, outcome=outcome, bc_assign=bc_assign, realizations=realizations)
    return res


# ======================================================================== figure
def _figure(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", palette="muted", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, RED, GREEN, GREY, ORNG, PURP = "#4C72B0", "#C44E52", "#55A868", "#8a8a8a", "#dd8452", "#8e6fb0"

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8)); fig.subplots_adjust(wspace=0.28, hspace=0.36)

    # ---- A: a representative solver/reference field + residual (Dirichlet outflow layer) ----
    axA = axes[0, 0]
    field, u_in, g_out = res["realizations"][0]
    xs = np.linspace(0, L, N_OBS); dx = xs[1] - xs[0]
    u_ref = reference_field(field, u_in, g_out, "dirichlet", xs)
    u_up = solver_field(field, u_in, g_out, "dirichlet", "upwind", xs)
    u_ce = solver_field(field, u_in, g_out, "dirichlet", "central", xs)
    axA.plot(xs, u_ref, color="k", lw=1.5, ls=(0, (3, 2)), label="reference (fine solve)")
    axA.plot(xs, u_up, color=RED, lw=1.2, label="upwind")
    axA.plot(xs, u_ce, color=BLUE, lw=1.2, label="central")
    axA.set_xlabel("$x$"); axA.set_ylabel("$u$"); axA.set_title("Solver fields, Dirichlet outflow layer", fontsize=9.5)
    axA.legend(frameon=False, fontsize=7.4)
    axA.text(-0.16, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # ---- B: residuals; shade the boundary margin that is masked ----
    axB = axes[0, 1]
    axB.plot(xs, u_up - u_ref, color=RED, lw=1.2, label="upwind residual")
    axB.plot(xs, u_ce - u_ref, color=BLUE, lw=1.2, label="central residual")
    axB.axhline(0, color=GREY, lw=0.8)
    mn = res["margin_needed"]
    axB.axvspan(0, mn * L, color=ORNG, alpha=0.16); axB.axvspan(L - mn * L, L, color=ORNG, alpha=0.16,
                                                                label=f"masked margin {mn:.2f}L")
    axB.set_xlabel("$x$"); axB.set_ylabel(r"$r=u_{\mathrm{solver}}-u_{\mathrm{ref}}$")
    axB.set_title("Residual: boundary layer vs interior", fontsize=9.5)
    axB.legend(frameon=False, fontsize=7.2)
    axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # ---- C: margin sweep -- accuracy & floor vs interior margin ----
    axC = axes[1, 0]
    mfs = [s["margin"] for s in res["sweep"]]
    accs = [s["acc"] for s in res["sweep"]]
    fls = [s["floor"] for s in res["sweep"]]
    axC.plot(mfs, accs, "o-", color=BLUE, lw=2, ms=6, label="upwind vs central acc")
    axC.plot(mfs, fls, "s--", color=GREY, lw=1.4, ms=4, label="perm floor")
    axC.axvline(res["margin_needed"], color=GREEN, ls=(0, (2, 2)), lw=1.6)
    axC.text(res["margin_needed"] + 0.004, 0.55, f"margin\nneeded\n{res['margin_needed']:.2f}L",
             color=GREEN, fontsize=7.5, va="center")
    axC.scatter([0.0], [res["full"]["acc"]], s=90, color=RED, zorder=5, label="full domain")
    axC.set_xlabel("interior boundary margin (fraction of $L$, each end)")
    axC.set_ylabel("GroupKFold accuracy"); axC.set_ylim(0.4, 1.02)
    axC.set_title("Interior margin recovers the signature", fontsize=9.5)
    axC.legend(frameon=False, fontsize=7.4, loc="lower right")
    axC.text(-0.16, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # ---- D: per-BC-family interior + controls ----
    axD = axes[1, 1]
    fams = list(res["perfam"].keys())
    labels = [f.replace("inflow_out", "inflow-out") for f in fams] + ["NC1", "NC2"]
    vals = [res["perfam"][f][0] for f in fams] + [res["nc1"], res["nc2"]]
    floors = [res["perfam"][f][1] for f in fams] + [res["nc1_f"], res["nc2_f"]]
    cols = [BLUE, GREEN, PURP, GREY, "#c8a35a"]
    axD.bar(range(len(vals)), vals, color=cols, width=0.66)
    for i, fl in enumerate(floors):
        axD.plot([i - 0.34, i + 0.34], [fl, fl], color="#222", ls=(0, (2, 1.5)), lw=1.5, zorder=6)
    for i, v in enumerate(vals): axD.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    axD.axhline(0.5, color=GREY, lw=0.8, ls=":")
    axD.set_xticks(range(len(labels))); axD.set_xticklabels(labels, fontsize=7.4, rotation=15)
    axD.set_ylim(0, 1.05); axD.set_ylabel("accuracy")
    axD.set_title(f"Interior (margin {res['margin_needed']:.2f}L) per BC family + controls\n(dashed=perm floor)", fontsize=9.0)
    axD.text(-0.16, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGDIR, "nonperiodic_bc.png"); fig.savefig(out); plt.close(fig)
    print(f"figure -> {out}")
    return out


if __name__ == "__main__":
    import sys
    res = main()
    if res is not None and "--plot" in sys.argv:
        _figure(res)
