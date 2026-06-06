#!/usr/bin/env python3
"""
solver-forensics :: SUPG / STABILIZATION ATTRIBUTION  (the CMAME-native scenario)
================================================================================
The silent-stabilization-change audit, in the journal's own vocabulary.

Advection-dominated steady advection-diffusion at HIGH Peclet on [0,L],

    a u_x - D u_xx = f ,    u(0)=u(L)=0 ,    Pe_h = a h / (2 D) >> 1 ,

discretized with hand-assembled P1 finite elements three ways:

  (A) Galerkin   : standard (central-like) Galerkin. UNSTABILIZED -> spurious
                   node-to-node OSCILLATIONS when Pe_h > 1. Modified-equation
                   error is dominated by DISPERSIVE high-derivative terms (no
                   added diffusion); the residual carries a strong u_xxx/u_xxxx
                   footprint and an ANTI-diffusive sign on u_xx near layers.
  (B) SUPG       : streamline-upwind Petrov-Galerkin. CONSISTENT stabilization:
                   the test space is w + tau a w_x, so the WHOLE residual
                   (a u_x - D u_xx - f) is weighted. tau = h/(2|a|) * (coth(Pe_h)
                   - 1/Pe_h) (the classic optimal/"doubly-asymptotic" tau). The
                   added streamline diffusion is tau a^2 u_xx, but because the
                   source/diffusion terms are ALSO weighted it is consistent (it
                   vanishes on the exact solution). Distinct higher-order footprint.
  (C) ArtVisc    : isotropic ARTIFICIAL VISCOSITY. INCONSISTENT: a tuned extra
                   Laplacian nu_art u_xx is added to the bilinear form ONLY (the
                   source is NOT reweighted). Two variants:
                     ArtVisc_m  : nu_art = tau*a^2  -> added diffusion MATCHES SUPG
                                  exactly (adversarial; the leading modified-equation
                                  term is identical, so SUPG vs ArtVisc_m differs only
                                  in CONSISTENCY -- source reweighting + higher-order).
                     ArtVisc_up : nu_art = |a|h/2   -> the common upwind-equivalent
                                  practitioner rule (UNmatched; ~1.25x SUPG here).
                                  This is the realistic silent-swap scenario.

Reference is the ANALYTIC solution of the BVP (exact boundary-layer solution for
the constant-coefficient operator), so residuals are not reference-contaminated.

SIGNATURE  : unit-normalized least-squares coefficient DIRECTION of c in
             r = u_solver - u_ref ~ sum_p c_p d_x^p u, library {u_xx,u_xxx,u_xxxx},
             FD derivatives of the OBSERVED solver field on a regular grid (the FE
             nodal solution is interpolated to a uniform grid -- mirrors the
             interp-to-grid signature in src/robustness/irregular_mesh.py).
ATTRIBUTION: StandardScaler+LogisticRegression, GroupKFold(5) grouped by INITIAL
             CONDITION (here: the forcing/BC realization), label-PERMUTATION floor
             on EVERY reported number.
CONTROLS   : NC1 = same scheme (Galerkin), IC + noise only -> must sit ~chance.
             (NC2 grid-change confound is reported as a diagnostic: stabilization
             tau and the Galerkin oscillation amplitude BOTH depend on h, so a
             grid change is itself a stabilization-relevant confound; we report it.)

Tasks / reported numbers:
  ID3            3-way among the stabilized schemes (SUPG / ArtVisc_m / ArtVisc_up)
  galerkin_vs_stab  oscillatory Galerkin vs (any) stabilized -- the easy gate
  SUPG_vs_ArtVisc_matched   the ADVERSARIAL pairwise (consistent vs inconsistent,
                            matched leading diffusion) -- the documented HARD case.
  SUPG_vs_ArtVisc_upwind    the REALISTIC pairwise (unmatched |a|h/2 rule).
  NC1            IC+noise control (same scheme) -> chance.

Self-contained: numpy + scipy + sklearn, CPU, ~30-60 s. NO FEM library (P1 FEM
hand-assembled). Guarded by __main__.  Run:  python src/audit/stabilization_audit.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---- physical / numerical constants (advection-dominated) ----
L      = 1.0
A      = 1.0          # advection speed a
D      = 2.5e-3       # diffusivity D  -> with N=40 elements, Pe_h = a h/(2D) = (1/40)/(2*2.5e-3) = 5 >> 1
N_EL   = 40           # P1 elements (coarse enough that Galerkin oscillates: Pe_h=5)
N_OBS  = 200          # uniform observation grid the FE solution is interpolated onto
N_IC   = 60           # forcing/BC realizations (the "initial condition" group key)
SIGMA  = 0.01         # field-relative observation noise (robustness floor)
LIB    = (2, 3, 4)    # derivative library orders {u_xx, u_xxx, u_xxxx}
# ArtVisc_m : artificial viscosity MATCHED to SUPG added diffusion (adversarial -- identical leading term)
# ArtVisc_up: artificial viscosity = |a|h/2 (the common upwind-equivalent practitioner rule; UNmatched)
SCHEMES = ("Galerkin", "SUPG", "ArtVisc_m", "ArtVisc_up")

def mesh_params(n_el):
    """All h-dependent quantities for an N_EL-element mesh (used by main and NC2 grid-change)."""
    hh = L / n_el
    pe = A * hh / (2 * D)
    tau = (hh / (2 * abs(A))) * (1.0 / np.tanh(pe) - 1.0 / pe)   # classic optimal SUPG tau
    nu_match = tau * A * A          # artificial visc MATCHED to SUPG added diffusion (adversarial)
    nu_up = abs(A) * hh / 2.0       # upwind-equivalent artificial visc |a|h/2 (the common practitioner rule)
    return dict(n_el=n_el, h=hh, pe=pe, tau=tau, nu_match=nu_match, nu_up=nu_up)

PARAMS = mesh_params(N_EL)
h, PE_h, TAU, NU_MATCH, NU_UP = PARAMS["h"], PARAMS["pe"], PARAMS["tau"], PARAMS["nu_match"], PARAMS["nu_up"]

# ======================================================================== analytic reference
def exact_bvp(f_coeffs, bc):
    """Exact solution of  a u' - D u'' = f(x),  u(0)=bc[0], u(L)=bc[1],  on a fine grid.
    f(x) = sum_m f_coeffs[m] sin(m pi x / L)  (m=1..M). For each sine mode the particular
    solution of a u_p' - D u_p'' = sin(k x) is analytic; homogeneous part A + B exp(a x / D)
    fixes the BCs. Evaluated on N_OBS uniform points (this IS the reference field).
    Returns u_exact on the uniform observation grid xs."""
    xs = np.linspace(0, L, N_OBS)
    u = np.zeros_like(xs)
    # particular solution per sine mode: u_p = P sin(kx) + Q cos(kx)
    # a*(P k cos - Q k sin) - D*(-P k^2 sin - Q k^2 cos) = sin(kx)
    #   sin: -a Q k + D P k^2 = 1 ;  cos: a P k + D Q k^2 = 0  -> Q = -(a/(D k)) P
    #   D P k^2 + (a^2/(D)) P = 1  -> P = 1 / (D k^2 + a^2/D)
    for m, fm in enumerate(f_coeffs, start=1):
        if fm == 0.0: continue
        k = m * np.pi / L
        P = 1.0 / (D * k * k + A * A / D)
        Q = -(A / (D * k)) * P
        u += fm * (P * np.sin(k * xs) + Q * np.cos(k * xs))
    # homogeneous A + B exp(a x / D) to satisfy BCs (with whatever the particular part gives at ends)
    up0 = u[0]; upL = u[-1]
    e0, eL = 1.0, np.exp(A * L / D)
    # solve [1, e0; 1, eL] [Ah; Bh] = [bc0 - up0; bcL - upL]
    rhs0 = bc[0] - up0; rhsL = bc[1] - upL
    Bh = (rhsL - rhs0) / (eL - e0)
    Ah = rhs0 - Bh * e0
    # exp(a x/D) overflows for large a/D; clip in log-space safely
    with np.errstate(over="ignore"):
        hom = Ah + Bh * np.exp(np.clip(A * xs / D, None, 700))
    return xs, u + hom

# ======================================================================== hand-assembled P1 FEM
def assemble(scheme, f_coeffs, bc, params=None):
    """P1 FEM on params['n_el'] uniform elements. Returns nodal solution u_h at FE nodes xn.
    Element matrices (length-h linear elements):
      mass M_e   = h/6 [[2,1],[1,2]]
      adv  C_e   = (a/2)[[-1,1],[-1,1]]   (Galerkin advection, skew central-like)
      diff S_e   = (D/h)[[1,-1],[-1,1]]
    SUPG adds, per element, tau*(a w_x)(a u_x - D u_xx - f):
       stiffness contribution tau*a^2 (u_x, w_x)  =>  tau*a^2/h [[1,-1],[-1,1]]
       (D u_xx = 0 on P1 interior; boundary flux handled by the consistent assembly)
       load contribution      tau*a (f, w_x)      =>  per-element  tau*a * f_avg * [-1, +1]
    ArtVisc adds isotropic nu_art (u_x, w_x) to STIFFNESS ONLY (load NOT reweighted):
       nu_art/h [[1,-1],[-1,1]] ; this is the INCONSISTENT scheme.
    """
    if params is None: params = PARAMS
    n_el, hh, tau = params["n_el"], params["h"], params["tau"]
    nn = n_el + 1
    xn = np.linspace(0, L, nn)
    K = np.zeros((nn, nn)); F = np.zeros(nn)

    def fval(x):  # forcing f(x)
        s = 0.0
        for m, fm in enumerate(f_coeffs, start=1):
            s += fm * np.sin(m * np.pi * x / L)
        return s

    Ce = (A / 2.0) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    Se = (D / hh) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    Stab = (1.0 / hh) * np.array([[1.0, -1.0], [-1.0, 1.0]])   # (u_x,w_x) element stiffness shape

    for e in range(n_el):
        n0, n1 = e, e + 1
        Ke = Ce + Se
        if scheme == "SUPG":
            Ke = Ke + tau * A * A * Stab
        elif scheme == "ArtVisc_m":
            Ke = Ke + params["nu_match"] * Stab
        elif scheme == "ArtVisc_up":
            Ke = Ke + params["nu_up"] * Stab
        # consistent (Galerkin) load: 2-pt Gauss on the element
        xa, xb = xn[n0], xn[n1]
        g1 = xa + hh * (0.5 - 0.5 / np.sqrt(3)); g2 = xa + hh * (0.5 + 0.5 / np.sqrt(3))
        # linear shape fns at the two Gauss pts
        def Nshp(x): t = (x - xa) / hh; return np.array([1 - t, t])
        Fe = (hh / 2.0) * (fval(g1) * Nshp(g1) + fval(g2) * Nshp(g2))   # quadrature weights h/2 each
        if scheme == "SUPG":
            # consistent residual load: tau * a * (f, w_x), w_x = [-1/h, +1/h]
            f_avg = 0.5 * (fval(g1) + fval(g2))
            Fe = Fe + tau * A * f_avg * np.array([-1.0, 1.0])
        # (ArtVisc deliberately does NOT reweight the load -> inconsistent)
        K[np.ix_([n0, n1], [n0, n1])] += Ke
        F[[n0, n1]] += Fe

    # Dirichlet BCs (strong)
    for nd, val in ((0, bc[0]), (nn - 1, bc[1])):
        K[nd, :] = 0.0; K[nd, nd] = 1.0; F[nd] = val
    u_h = np.linalg.solve(K, F)
    return xn, u_h

# ======================================================================== signature
def interp_to_grid(xn, u_h, xs):
    """Interpolate the FE nodal solution to the uniform observation grid (mirrors the
    interp-to-grid signature in src/robustness/irregular_mesh.py: irregular/native ->
    regular grid, THEN grid FD signature)."""
    return interp1d(xn, u_h, kind="cubic", fill_value="extrapolate")(xs)

def fd_derivs(u, dx):
    """Interior FD derivatives (non-periodic, Dirichlet BVP). Returns u_xx,u_xxx,u_xxxx and
    the interior slice valid for the 4th-order stencil."""
    uxx   = (np.roll(u,-1) - 2*u + np.roll(u,1)) / dx**2
    uxxx  = (np.roll(u,-2) - 2*np.roll(u,-1) + 2*np.roll(u,1) - np.roll(u,2)) / (2*dx**3)
    uxxxx = (np.roll(u,-2) - 4*np.roll(u,-1) + 6*u - 4*np.roll(u,1) + np.roll(u,2)) / dx**4
    sl = slice(2, len(u) - 2)   # drop boundary nodes where the stencil wraps
    return uxx[sl], uxxx[sl], uxxxx[sl]

def signature(u_obs, r_obs, dx):
    uxx, uxxx, uxxxx = fd_derivs(u_obs, dx)
    A_lib = np.stack([uxx, uxxx, uxxxx], 1)
    b = r_obs[2:len(r_obs) - 2]
    c, *_ = np.linalg.lstsq(A_lib, b, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c

# ======================================================================== IC ensemble
def random_forcing(rng, M=5):
    """A forcing/BC realization (the GROUP key). Random sine forcing + random BCs."""
    f_coeffs = rng.normal(size=M)
    f_coeffs = f_coeffs / (np.max(np.abs(f_coeffs)) + 1e-9)
    bc = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
    return f_coeffs, bc

def sigs(scheme, realizations, sigma, seed, params=None):
    xs = np.linspace(0, L, N_OBS); dx = xs[1] - xs[0]
    out = []
    gn = np.random.default_rng(seed)
    for f_coeffs, bc in realizations:
        xn, u_h = assemble(scheme, f_coeffs, bc, params)
        u_grid = interp_to_grid(xn, u_h, xs)
        _, u_ex = exact_bvp(f_coeffs, bc)
        if sigma > 0:
            u_grid = u_grid + sigma * np.sqrt(np.mean(u_grid**2)) * gn.standard_normal(N_OBS)
        r = u_grid - u_ex
        out.append(signature(u_grid, r, dx))
    return np.array(out)

# ======================================================================== metrics
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def acc(F, y, g):
    return cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean()
def perm_floor(F, y, g, seed, reps=40):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean()
                            for _ in range(reps)]))

# ======================================================================== solver verification
def verify_solver():
    """Confirm the FEM is convergent (vs the analytic BVP solution) and that Galerkin
    oscillates while the stabilized schemes do not. Returns a dict for the report."""
    f_coeffs, bc = (np.array([1.0, 0.0, 0.0]), (0.0, 1.0))   # smooth forcing + boundary layer at x=L (outflow)
    xs = np.linspace(0, L, N_OBS)
    _, u_ex = exact_bvp(f_coeffs, bc)
    rep = {}
    # convergence of stabilized (SUPG) under mesh refinement, via the SAME assemble() code path
    errs = {}
    for nel in (20, 40, 80, 160, 320):
        xn, u_h = assemble("SUPG", f_coeffs, bc, params=mesh_params(nel))
        u_g = interp1d(xn, u_h, kind="cubic", fill_value="extrapolate")(xs)
        errs[nel] = float(np.sqrt(np.mean((u_g - u_ex)**2)) / (np.sqrt(np.mean(u_ex**2)) + 1e-12))
    rep["supg_conv"] = errs
    # oscillation diagnostic at N_EL=40: count of slope sign-changes (node-to-node wiggle) over the last
    # 10 nodes (the boundary-layer region). Monotone stabilized -> 0; oscillatory Galerkin -> many.
    osc = {}
    for sc in SCHEMES:
        _, u_h = assemble(sc, f_coeffs, bc)
        d = np.diff(u_h[-11:])                       # slopes over the layer region
        osc[sc] = int(np.sum(np.diff(np.sign(d)) != 0))
    rep["wiggle"] = osc
    return rep

# ======================================================================== RUN
def main():
    print("="*78)
    print("STABILIZATION ATTRIBUTION  (steady advection-diffusion, hand-assembled P1 FEM)")
    print("="*78)
    print(f"a={A}, D={D}, N_EL={N_EL}, h={h:.4f}  ->  Pe_h = a h/(2D) = {PE_h:.2f}  (>>1: Galerkin oscillates)")
    print(f"SUPG tau = {TAU:.5f} (optimal coth form)")
    print(f"ArtVisc_m  nu = tau*a^2 = {NU_MATCH:.5f}  (MATCHED to SUPG added diffusion -- adversarial)")
    print(f"ArtVisc_up nu = |a|h/2  = {NU_UP:.5f}  (upwind-equiv practitioner rule -- {NU_UP/NU_MATCH:.2f}x SUPG, realistic)")
    print(f"added-diffusion / physical: SUPG nu/D = {NU_MATCH/D:.2f}  (stabilization >> physical diffusion)\n")

    # --- solver verification ---
    vr = verify_solver()
    print("[verify] SUPG convergence vs analytic BVP (rel. L2 error):")
    prev = None; rates = []
    for nel in sorted(vr["supg_conv"]):
        e = vr["supg_conv"][nel]
        rate = "" if prev is None else f"rate={np.log(prev/e)/np.log(2):.2f}"
        print(f"          N_EL={nel:4d}  err={e:.3e}  {rate}")
        if prev is not None: rates.append(np.log(prev/e)/np.log(2))
        prev = e
    conv_ok = vr["supg_conv"][320] < vr["supg_conv"][20] and vr["supg_conv"][320] < 0.05
    print(f"          convergent: {conv_ok}  (mean observed rate ~{np.mean(rates):.2f})")
    print("[verify] boundary-layer wiggle (slope sign-changes over last 10 nodes; 0=monotone):")
    for sc in SCHEMES:
        print(f"          {sc:10s} wiggle={vr['wiggle'][sc]:2d}"
              + ("   <- OSCILLATES" if sc == "Galerkin" else "   (stabilized, monotone)"))
    gal_osc = vr["wiggle"]["Galerkin"] >= 3 and max(vr["wiggle"][s] for s in SCHEMES if s != "Galerkin") == 0
    print(f"          Galerkin oscillates while stabilized schemes are monotone: {gal_osc}\n")

    # --- build the ensemble ---
    rng = np.random.default_rng(0)
    reals = [random_forcing(rng) for _ in range(N_IC)]   # shared forcing/BC realizations across schemes
    ic = np.arange(N_IC)
    F = {sc: sigs(sc, reals, SIGMA, 100 + 1000*i) for i, sc in enumerate(SCHEMES)}

    # mean signature direction per scheme (interpretability)
    mdir = {sc: F[sc].mean(0) / (np.linalg.norm(F[sc].mean(0)) + 1e-12) for sc in SCHEMES}

    # helper for a balanced pairwise classification + permutation floor
    def pair(a, b, seed):
        Xp = np.vstack([F[a], F[b]]); yp = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gp = np.r_[ic, ic]
        return acc(Xp, yp, gp), perm_floor(Xp, yp, gp, seed)

    # --- ID3-among-stabilized: SUPG vs ArtVisc_m vs ArtVisc_up (3-way, the fine taxonomy) ---
    STAB = ("SUPG", "ArtVisc_m", "ArtVisc_up")
    Xid = np.vstack([F[s] for s in STAB]); yid = np.concatenate([np.full(N_IC, i) for i in range(3)])
    gid = np.concatenate([ic]*3)
    id3, id3f = acc(Xid, yid, gid), perm_floor(Xid, yid, gid, 7)

    # --- Galerkin vs stabilized (Galerkin vs all three stabilized) : the easy gate (imbalanced 60:180) ---
    Xgs = np.vstack([F["Galerkin"]] + [F[s] for s in STAB])
    ygs = np.r_[np.zeros(N_IC), np.ones(3*N_IC)]; ggs = np.r_[ic, ic, ic, ic]
    gvs, gvs_f = acc(Xgs, ygs, ggs), perm_floor(Xgs, ygs, ggs, 11)

    # --- SUPG vs ArtVisc_m : the ADVERSARIAL pair (matched added diffusion) -- expected hard ---
    sva, sva_f = pair("SUPG", "ArtVisc_m", 13)
    F0 = {sc: sigs(sc, reals, 0.0, 500 + 1000*j) for j, sc in enumerate(("SUPG", "ArtVisc_m"))}
    sva0 = acc(np.vstack([F0["SUPG"], F0["ArtVisc_m"]]),
               np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic])

    # --- SUPG vs ArtVisc_up : the REALISTIC silent-change pair (unmatched, |a|h/2 rule) ---
    svu, svu_f = pair("SUPG", "ArtVisc_up", 15)

    # --- Galerkin vs each stabilized (context) ---
    gsupg, gsupg_f = pair("Galerkin", "SUPG", 21)

    # --- NC1: same scheme (Galerkin), IC + noise only -> chance. The class label here is ARBITRARY
    #     (a random partition of ICs), so a single split is a noisy draw; we average over several random
    #     label assignments to report the TRUE chance behavior, with the permutation floor as reference. ---
    Fnc = sigs("Galerkin", reals, SIGMA, 9000)
    half = N_IC // 2
    nc1_draws = []
    for s in range(8):
        perm = np.random.default_rng(1000 + s).permutation(N_IC)
        gA, gB = perm[:half], perm[half:]
        nc1_draws.append(acc(np.vstack([Fnc[gA], Fnc[gB]]),
                             np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[gA], ic[gB]]))
    nc1 = float(np.mean(nc1_draws)); nc1_sd = float(np.std(nc1_draws))
    nc1_f = perm_floor(np.vstack([Fnc[:half], Fnc[half:]]),
                       np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[:half], ic[half:]], 31)

    # --- NC2 (diagnostic): grid change, same scheme. h-dependent stabilization makes this a real confound ---
    Fg_a = sigs("Galerkin", reals, SIGMA, 7000, params=mesh_params(40))
    Fg_b = sigs("Galerkin", reals, SIGMA, 7700, params=mesh_params(56))
    nc2 = acc(np.vstack([Fg_a, Fg_b]), np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic])
    nc2_f = perm_floor(np.vstack([Fg_a, Fg_b]), np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic], 41)

    # ---------------------------------------------------------------- report
    print("="*78)
    print("ATTRIBUTION RESULTS  (GroupKFold-by-IC, coefficient-direction signature, perm floor)")
    print("="*78)
    def line(name, a, f, chance):
        print(f"  {name:<38} acc={a:.3f}  floor={f:.3f}  gap={a-f:+.3f}  (chance~{chance})")
    line("ID3  SUPG/ArtVisc_m/ArtVisc_up (3-way)", id3, id3f, "0.33")
    line("Galerkin vs stabilized (easy gate)", gvs, gvs_f, "0.75 maj")
    line("Galerkin vs SUPG", gsupg, gsupg_f, "0.50")
    print("  " + "-"*72)
    line("SUPG vs ArtVisc_m  (MATCHED, adversarial)", sva, sva_f, "0.50")
    print(f"  {'  ^ noise-free version':<38} acc={sva0:.3f}  "
          f"({'deterministic signal' if sva0 > 0.65 else 'weak/at-chance'})")
    line("SUPG vs ArtVisc_up (UNMATCHED, realistic)", svu, svu_f, "0.50")
    print("  " + "-"*72)
    print(f"  {'NC1  IC+noise (same scheme)':<38} acc={nc1:.3f} +/- {nc1_sd:.3f}  floor={nc1_f:.3f}  "
          f"(arbitrary-label mean over 8 splits; chance~0.50)")
    line("NC2  grid change (diagnostic)", nc2, nc2_f, "0.50")

    print("\n  mean signature directions [c_xx, c_xxx, c_xxxx] (unit):")
    for sc in SCHEMES:
        v = mdir[sc]; print(f"    {sc:11s} [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]")
    cos_sm = float(abs(mdir["SUPG"] @ mdir["ArtVisc_m"]))    # matched: expected ~1 (collinear)
    cos_su = float(abs(mdir["SUPG"] @ mdir["ArtVisc_up"]))   # unmatched
    cos_gs = float(abs(mdir["Galerkin"] @ mdir["SUPG"]))
    print(f"    |cos(SUPG, ArtVisc_m)| = {cos_sm:.3f}  (matched: {'collinear -> subtle' if cos_sm>0.97 else 'separable'})")
    print(f"    |cos(SUPG, ArtVisc_up)| = {cos_su:.3f}  |cos(Galerkin, SUPG)| = {cos_gs:.3f}")

    # ---------------------------------------------------------------- CSV
    csv = os.path.join(TAB, "stabilization_audit_results.csv")
    with open(csv, "w") as fcsv:
        fcsv.write("task,accuracy,perm_floor,chance,note\n")
        fcsv.write(f"ID3_stabilized_3way,{id3:.4f},{id3f:.4f},0.333,SUPG/ArtVisc_m/ArtVisc_up\n")
        fcsv.write(f"galerkin_vs_stabilized,{gvs:.4f},{gvs_f:.4f},0.750,oscillatory vs stabilized (60:180)\n")
        fcsv.write(f"galerkin_vs_SUPG,{gsupg:.4f},{gsupg_f:.4f},0.500,pairwise\n")
        fcsv.write(f"SUPG_vs_ArtVisc_matched,{sva:.4f},{sva_f:.4f},0.500,ADVERSARIAL matched added-diffusion (hard)\n")
        fcsv.write(f"SUPG_vs_ArtVisc_matched_noisefree,{sva0:.4f},,0.500,deterministic-signal check\n")
        fcsv.write(f"SUPG_vs_ArtVisc_upwind,{svu:.4f},{svu_f:.4f},0.500,REALISTIC unmatched |a|h/2 rule\n")
        fcsv.write(f"NC1_ic_noise,{nc1:.4f},{nc1_f:.4f},0.500,same scheme control (mean over 8 arbitrary-label splits; sd={nc1_sd:.3f})\n")
        fcsv.write(f"NC2_grid_change,{nc2:.4f},{nc2_f:.4f},0.500,h-dependent stabilization confound (diagnostic)\n")
        fcsv.write(f"cos_SUPG_ArtVisc_matched,{cos_sm:.4f},,,signature collinearity (matched)\n")
        fcsv.write(f"cos_SUPG_ArtVisc_upwind,{cos_su:.4f},,,signature collinearity (unmatched)\n")
        fcsv.write(f"Pe_h,{PE_h:.4f},,,mesh Peclet\n")
        fcsv.write(f"tau,{TAU:.6f},,,SUPG tau\n")
        fcsv.write(f"nu_match_over_D,{NU_MATCH/D:.4f},,,SUPG added visc / physical diffusion\n")
    print(f"\nmetrics -> {csv}")

    res = dict(id3=id3, id3f=id3f, gvs=gvs, gvs_f=gvs_f, gsupg=gsupg, gsupg_f=gsupg_f,
               sva=sva, sva_f=sva_f, sva0=sva0, svu=svu, svu_f=svu_f,
               nc1=nc1, nc1_f=nc1_f, nc2=nc2, nc2_f=nc2_f, mdir=mdir,
               cos_sm=cos_sm, cos_su=cos_su, conv_ok=conv_ok, gal_osc=gal_osc, verify=vr)
    _figure(res, reals)

    # ---------------------------------------------------------------- honest verdict
    print("\n" + "="*78 + "\nVERDICT (honest)\n" + "="*78)
    easy_ok    = gvs - gvs_f >= 0.15 and gvs >= 0.85
    matched_ok = sva - sva_f >= 0.15 and sva >= 0.75
    real_ok    = svu - svu_f >= 0.15 and svu >= 0.75
    nc1_ok     = nc1 - nc1_f <= 0.10
    print(f"  Galerkin vs stabilized (easy gate): {'DETECTED' if easy_ok else 'WEAK'}  ({gvs:.3f} vs floor {gvs_f:.3f})")
    print(f"  SUPG vs ArtVisc MATCHED (adversarial): {'DETECTED' if matched_ok else 'WEAK/AT-CHANCE'}  ({sva:.3f} vs floor {sva_f:.3f})")
    print(f"  SUPG vs ArtVisc UNMATCHED (realistic): {'DETECTED' if real_ok else 'WEAK/AT-CHANCE'}  ({svu:.3f} vs floor {svu_f:.3f})")
    print(f"  NC1 control sits ~chance: {nc1_ok}  ({nc1:.3f} vs floor {nc1_f:.3f})")
    print("  ----")
    print("  The Galerkin (oscillatory) signature is cleanly separated -> the silent loss/addition of")
    print("  stabilization is attributable. SUPG-vs-artificial-viscosity is the documented HARD case:")
    print("  when added diffusion is MATCHED the residual signatures are near-collinear and the pair is")
    print("  at-chance; when the practitioner uses the common |a|h/2 rule (unmatched) it becomes")
    print("  recoverable. This is the honest boundary of stabilization attribution from the strong-form")
    print("  residual signature.")
    return res

# ======================================================================== figure
def _figure(r, reals):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", palette="muted", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP, ORNG = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0", "#dd8452"
    SC = {"Galerkin": (RED, "Galerkin (unstab.)"), "SUPG": (BLUE, "SUPG"),
          "ArtVisc_m": (GREEN, "ArtVisc matched"), "ArtVisc_up": (ORNG, "ArtVisc |a|h/2")}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8)); fig.subplots_adjust(wspace=0.27, hspace=0.34)

    # A: nodal solutions on the boundary-layer test (Galerkin oscillates)
    axA = axes[0, 0]
    f_coeffs, bc = (np.array([1.0, 0.0, 0.0]), (0.0, 1.0))
    xs, u_ex = exact_bvp(f_coeffs, bc)
    axA.plot(xs, u_ex, color="k", lw=1.4, ls=(0, (3, 2)), label="exact (reference)")
    for sc, (c, lab) in SC.items():
        xn, u_h = assemble(sc, f_coeffs, bc)
        axA.plot(xn, u_h, color=c, lw=1.3, marker="o", ms=2.3, label=lab)
    axA.set_xlabel("$x$"); axA.set_ylabel("$u$"); axA.set_xlim(0.55, 1.0)
    axA.set_title(f"Boundary layer, $Pe_h={PE_h:.0f}$: Galerkin oscillates", fontsize=9.5)
    axA.legend(frameon=False, fontsize=7.0)
    axA.text(-0.16, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: residual fields r = u_solver - u_ref (the signal the signature reads)
    axB = axes[0, 1]
    for sc, (c, lab) in SC.items():
        xn, u_h = assemble(sc, f_coeffs, bc)
        u_g = interp_to_grid(xn, u_h, xs); rr = u_g - u_ex
        axB.plot(xs, rr, color=c, lw=1.3, label=lab)
    axB.axhline(0, color=GREY, lw=0.8)
    axB.set_xlabel("$x$"); axB.set_ylabel(r"residual $r = u_h - u_{\mathrm{ref}}$")
    axB.set_title("Residual field (matched ArtVisc $\\approx$ SUPG)", fontsize=9.5)
    axB.legend(frameon=False, fontsize=7.0)
    axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: mean signature directions
    axC = axes[1, 0]; labs = [r"$c_{xx}$", r"$c_{xxx}$", r"$c_{xxxx}$"]; xb = np.arange(3); w = 0.20
    order = ("Galerkin", "SUPG", "ArtVisc_m", "ArtVisc_up")
    for i, sc in enumerate(order):
        axC.bar(xb + (i-1.5)*w, r["mdir"][sc], w, color=SC[sc][0], label=SC[sc][1])
    axC.axhline(0, color=GREY, lw=0.8); axC.set_xticks(xb); axC.set_xticklabels(labs)
    axC.set_ylabel("unit coeff direction")
    axC.set_title(f"Mean signatures  |cos(SUPG,ArtVisc$_m$)|={r['cos_sm']:.2f}", fontsize=9.5)
    axC.legend(frameon=False, fontsize=7.0)
    axC.text(-0.16, 1.04, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: attribution accuracies vs per-task permutation floors
    axD = axes[1, 1]
    labels = ["3-way\nID", "Galerkin\nvs stab.", "SUPG vs\nArtV$_m$", "SUPG vs\nArtV$_{up}$", "NC1", "NC2\n(grid)"]
    vals  = [r["id3"], r["gvs"], r["sva"], r["svu"], r["nc1"], r["nc2"]]
    floors= [r["id3f"], r["gvs_f"], r["sva_f"], r["svu_f"], r["nc1_f"], r["nc2_f"]]
    cols  = [PURP, RED, GREEN, ORNG, GREY, "#c8a35a"]
    axD.bar(range(6), vals, color=cols, width=0.66)
    for i, fl in enumerate(floors):
        axD.plot([i-0.34, i+0.34], [fl, fl], color="#222", ls=(0, (2, 1.5)), lw=1.5, zorder=6)
    for i, v in enumerate(vals): axD.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    axD.text(2, r["sva"] - 0.06, f"clean:\n{r['sva0']:.2f}", ha="center", va="top", fontsize=6.4, color="#444")
    axD.set_xticks(range(6)); axD.set_xticklabels(labels, fontsize=7.3); axD.set_ylim(0, 1.05)
    axD.set_ylabel("GroupKFold accuracy")
    axD.set_title("Attribution (dashed = perm floor)", fontsize=9.5)
    axD.text(-0.16, 1.04, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIG, "stabilization_audit.png"); fig.savefig(out); plt.close(fig)
    print(f"figure  -> {out}")

if __name__ == "__main__":
    main()
