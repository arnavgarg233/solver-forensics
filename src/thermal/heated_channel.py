import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
import time
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------------------------------------------------------
# Heated-channel model and nondimensionalization
# -----------------------------------------------------------------------------
# Lengths are nondimensionalized by the channel height.  Umax=1 and Ly=1 give
# Ubar=2/3, so a_th=(2/3)/Pe.  Dividing the dimensional energy equation by
# rho*cp makes the prescribed heater load g_flux=q''/(rho*cp).  On each P1
# heater edge (a,b), the exact natural-boundary contribution is
#
#       F[a] += g_flux * edge_length / 2,
#       F[b] += g_flux * edge_length / 2.
#
# The corresponding wall-normal temperature gradient is q''/k=g_flux/a_th.
LX = 3.0
LY = 1.0
UMAX = 1.0
UBAR = (2.0 / 3.0) * UMAX
# A 1.4-long localized heater keeps the length-averaged developing-flow Nu in
# the requested O(1)-O(20) range through Pe=200; the shorter [0.8,1.6] segment
# puts excessive quadrature weight on its leading-edge heat-transfer singularity.
XH0 = 0.6
XH1 = 2.0
G_FLUX = 1.0
DH = 2.0 * LY
N_IC = 60


def thermal_diffusivity(Pe, Ly=LY, Ubar=UBAR):
    """Return ``a_th=Ubar*Ly/Pe`` for the requested channel Peclet number."""
    Pe = float(Pe)
    if not np.isfinite(Pe) or Pe <= 0.0:
        raise ValueError("Pe must be finite and positive")
    return float(Ubar * Ly / Pe)


def channel_velocity(y, Ly=LY, Umax=UMAX):
    """Plane-Poiseuille streamwise velocity; the transverse velocity is zero."""
    y = np.asarray(y, dtype=float)
    return Umax * (1.0 - ((y - 0.5 * Ly) / (0.5 * Ly)) ** 2)


def make_thermal_ic(seed):
    """Create one deterministic audit realization of the thermal boundary data.

    The population varies heater strength by +/-15%, translates the fixed
    1.4-long heater by +/-0.1, and adds a small linear inlet-temperature ramp.
    The validation/demo IC overrides the ramp to zero so its energy statement is
    exactly heat input versus outlet advected enthalpy.
    """
    rng = np.random.default_rng(seed)
    centre = float(rng.uniform(1.20, 1.40))
    half_length = 0.70
    return {
        "g_flux": float(rng.uniform(0.85, 1.15)),
        "xh0": centre - half_length,
        "xh1": centre + half_length,
        "inlet_ramp": float(rng.uniform(-0.03, 0.03)),
    }


def _boundary_edges(elems):
    """Return the undirected edges belonging to exactly one triangle."""
    edges = np.vstack((elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def tag_channel_boundaries(pts, elems, xh0=XH0, xh1=XH1, Lx=LX, Ly=LY):
    """Classify channel boundary nodes and edges for a heater interval.

    Heater edges are selected by edge midpoint.  ``heater_length`` is therefore
    the exact length represented by the discrete natural load and is the length
    used in the discrete energy-balance validation.
    """
    if not (0.0 < xh0 < xh1 < Lx):
        raise ValueError("heater interval must lie strictly inside (0, Lx)")
    bedges = _boundary_edges(np.asarray(elems, dtype=int))
    ep = pts[bedges]
    tol = 1.0e-10 * max(Lx, Ly)
    inlet_m = np.all(np.abs(ep[:, :, 0]) <= tol, axis=1)
    outlet_m = np.all(np.abs(ep[:, :, 0] - Lx) <= tol, axis=1)
    bottom_m = np.all(np.abs(ep[:, :, 1]) <= tol, axis=1)
    top_m = np.all(np.abs(ep[:, :, 1] - Ly) <= tol, axis=1)
    midpoint_x = ep[:, :, 0].mean(axis=1)
    heater_m = bottom_m & (midpoint_x >= xh0 - tol) & (midpoint_x <= xh1 + tol)
    other_wall_m = (bottom_m & ~heater_m) | top_m
    recognized = inlet_m | outlet_m | bottom_m | top_m
    if not np.all(recognized):
        raise RuntimeError("Delaunay hull contains an unclassified boundary edge")

    def nodes_of(mask):
        return np.unique(bedges[mask].ravel()) if np.any(mask) else np.empty(0, dtype=int)

    heater_edges = bedges[heater_m]
    heater_length = float(np.linalg.norm(
        pts[heater_edges[:, 1]] - pts[heater_edges[:, 0]], axis=1).sum())
    return {
        "inlet_nodes": nodes_of(inlet_m),
        "heater_nodes": nodes_of(heater_m),
        "other_wall_nodes": nodes_of(other_wall_m),
        "outlet_nodes": nodes_of(outlet_m),
        "inlet_edges": bedges[inlet_m],
        "heater_edges": heater_edges,
        "other_wall_edges": bedges[other_wall_m],
        "outlet_edges": bedges[outlet_m],
        "boundary_edges": bedges,
        "heater_interval": (float(xh0), float(xh1)),
        "heater_length": heater_length,
        "Lx": float(Lx),
        "Ly": float(Ly),
    }


def make_channel_mesh(nx, ny, seed):
    """Make an unstructured P1 mesh of ``[0,3] x [0,1]``.

    Boundary nodes form an exact regular frame; only interior grid nodes are
    jittered (20% of the local spacing) before a deterministic Delaunay
    triangulation.  The returned tags use the nominal heater [0.6, 2.0].
    """
    nx, ny = int(nx), int(ny)
    if nx < 3 or ny < 2:
        raise ValueError("nx >= 3 and ny >= 2 are required")
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, LX, nx + 1)
    ys = np.linspace(0.0, LY, ny + 1)
    bottom = np.column_stack((xs, np.zeros_like(xs)))
    top = np.column_stack((xs, np.full_like(xs, LY)))
    left = np.column_stack((np.zeros(ny - 1), ys[1:-1]))
    right = np.column_stack((np.full(ny - 1, LX), ys[1:-1]))
    boundary = np.vstack((bottom, top, left, right))

    gx, gy = np.meshgrid(xs[1:-1], ys[1:-1], indexing="xy")
    interior = np.column_stack((gx.ravel(), gy.ravel()))
    dx, dy = LX / nx, LY / ny
    jitter = rng.uniform(-1.0, 1.0, size=interior.shape)
    interior += 0.20 * jitter * np.array([dx, dy])
    interior[:, 0] = np.clip(interior[:, 0], 0.60 * dx, LX - 0.60 * dx)
    interior[:, 1] = np.clip(interior[:, 1], 0.60 * dy, LY - 0.60 * dy)

    pts = np.unique(np.round(np.vstack((boundary, interior)), 12), axis=0)
    elems = Delaunay(pts).simplices.copy()
    tags = tag_channel_boundaries(pts, elems)
    return pts, elems, tags


def channel_mesh_geometry(pts, elems, a_th):
    """Precompute P1 geometry and the nominal Codina/Shakib SUPG time scale.

    ``h_e=sqrt(2*A_e)``, which equals the grid spacing for an unjittered
    right-isosceles cell triangle.  The parabolic velocity's area mean is
    integrated exactly from the first and second moments of triangle ``y``.
    """
    pts = np.asarray(pts, dtype=float)
    elems = np.asarray(elems, dtype=int)
    a_th = float(a_th)
    if a_th <= 0.0:
        raise ValueError("a_th must be positive")
    p = pts[elems]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    Ae = 0.5 * np.abs(detJ)
    if np.any(Ae <= 1.0e-14):
        raise ValueError("mesh contains a degenerate triangle")
    b = np.stack((y2 - y3, y3 - y1, y1 - y2), axis=1) / detJ[:, None]
    c = np.stack((x3 - x2, x1 - x3, x2 - x1), axis=1) / detJ[:, None]

    yv = p[:, :, 1]
    y_mean = yv.mean(axis=1)
    y2_mean = (np.sum(yv * yv, axis=1)
               + yv[:, 0] * yv[:, 1]
               + yv[:, 0] * yv[:, 2]
               + yv[:, 1] * yv[:, 2]) / 6.0
    u_e = UMAX * (4.0 * y_mean / LY - 4.0 * y2_mean / LY ** 2)
    u_e = np.maximum(u_e, 0.0)
    udg = u_e[:, None] * b
    h_e = np.sqrt(2.0 * Ae)
    tau_nom = 1.0 / np.sqrt((2.0 * np.abs(u_e) / h_e) ** 2
                            + (4.0 * a_th / h_e ** 2) ** 2)
    ii = np.repeat(elems[:, :, None], 3, axis=2)
    jj = np.repeat(elems[:, None, :], 3, axis=1)
    return {
        "elems": elems, "Ae": Ae, "b": b, "c": c, "u_e": u_e,
        "udg": udg, "h_e": h_e, "tau_nom": tau_nom,
        "rows": ii.ravel(), "cols": jj.ravel(), "npt": len(pts),
        "a_th": a_th,
    }


def matched_artificial_diffusion(geom, alpha=1.0):
    """Area-weighted ``mean_e(alpha*tau_e*|u_e|^2)`` for the ArtVisc foil."""
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    Ae = geom["Ae"]
    return float(np.sum(Ae * (alpha * geom["tau_nom"]) * geom["u_e"] ** 2)
                 / np.sum(Ae))


def _nominal_ic():
    return {"g_flux": G_FLUX, "xh0": XH0, "xh1": XH1, "inlet_ramp": 0.0}


def assemble_channel(scheme, pts, elems, tags, Pe, ic=None, alpha=1.0,
                     geom=None, nu_art=None, return_meta=False):
    """Assemble and solve the steady heated-channel P1 system.

    Element matrices, with ``u_e`` the exact element-mean Poiseuille velocity,
    are

    ``a_th*A_e*(grad N_i . grad N_j) + (A_e/3)*(u_e*dN_j/dx)``.

    SUPG adds ``alpha*tau_nom*A_e*(u_e*dN_i/dx)*(u_e*dN_j/dx)``.
    ArtVisc instead adds the isotropic diffusion ``nu_art*A_e*gradNi.gradNj``,
    where the default ``nu_art`` is the area-weighted mean SUPG streamline
    diffusion.  The volume source is zero, so there is no SUPG source term.

    Only inlet rows are constrained.  Heater, adiabatic-wall, and outlet
    conditions remain natural.  Set ``return_meta=True`` to also receive the
    IC-specific tags and assembly diagnostics.
    """
    if scheme not in ("galerkin", "supg", "artvisc"):
        raise ValueError("scheme must be 'galerkin', 'supg', or 'artvisc'")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    data = _nominal_ic() if ic is None else {**_nominal_ic(), **ic}
    a_th = thermal_diffusivity(Pe)
    if geom is None:
        geom = channel_mesh_geometry(pts, elems, a_th)
    elif not np.isclose(geom["a_th"], a_th, rtol=1.0e-13, atol=0.0):
        raise ValueError("precomputed geometry was built for a different Pe")

    active_tags = tag_channel_boundaries(
        pts, elems, data["xh0"], data["xh1"], tags["Lx"], tags["Ly"])
    Ae, b, c, udg = geom["Ae"], geom["b"], geom["c"], geom["udg"]
    grad_dot = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    Ke = a_th * Ae[:, None, None] * grad_dot
    Ke += (Ae[:, None, None] / 3.0) * udg[:, None, :]

    actual_nu_art = 0.0
    if scheme == "supg":
        tau = alpha * geom["tau_nom"]
        Ke += (tau * Ae)[:, None, None] * (udg[:, :, None] * udg[:, None, :])
    elif scheme == "artvisc":
        actual_nu_art = (matched_artificial_diffusion(geom, alpha)
                         if nu_art is None else float(nu_art))
        if actual_nu_art < 0.0:
            raise ValueError("nu_art must be nonnegative")
        Ke += actual_nu_art * Ae[:, None, None] * grad_dot

    K = csr_matrix((Ke.ravel(), (geom["rows"], geom["cols"])),
                   shape=(geom["npt"], geom["npt"])).tolil()
    F = np.zeros(geom["npt"], dtype=float)
    hedge = active_tags["heater_edges"]
    edge_length = np.linalg.norm(pts[hedge[:, 1]] - pts[hedge[:, 0]], axis=1)
    edge_load = float(data["g_flux"]) * edge_length / 2.0
    np.add.at(F, hedge[:, 0], edge_load)
    np.add.at(F, hedge[:, 1], edge_load)

    inlet = active_tags["inlet_nodes"]
    inlet_values = float(data["inlet_ramp"]) * (pts[inlet, 1] / tags["Ly"] - 0.5)
    for nd, value in zip(inlet, inlet_values):
        K.rows[int(nd)] = [int(nd)]
        K.data[int(nd)] = [1.0]
        F[int(nd)] = value
    K = K.tocsr()
    T = np.asarray(spsolve(K, F), dtype=float)
    if not np.all(np.isfinite(T)):
        raise RuntimeError("channel solve produced non-finite temperatures")
    residual = float(np.linalg.norm(K @ T - F) / (np.linalg.norm(F) + 1.0e-30))
    meta = {
        "a_th": a_th,
        "alpha": alpha if scheme != "galerkin" else 0.0,
        "nu_art": actual_nu_art,
        "heat_added": float(data["g_flux"]) * active_tags["heater_length"],
        "linear_residual": residual,
        "inlet_values": inlet_values,
        "ic": data,
    }
    return (T, active_tags, meta) if return_meta else T


def _bulk_temperatures(pts, T, x_values, Ly=LY, Umax=UMAX, order=48):
    """Velocity-weighted bulk temperatures on vertical cross-sections."""
    x_values = np.atleast_1d(np.asarray(x_values, dtype=float))
    q, w = np.polynomial.legendre.leggauss(int(order))
    yq = 0.5 * Ly * (q + 1.0)
    wy = 0.5 * Ly * w
    uq = channel_velocity(yq, Ly=Ly, Umax=Umax)
    query = np.column_stack((np.repeat(x_values, len(yq)),
                             np.tile(yq, len(x_values))))
    interp = LinearNDInterpolator(Delaunay(pts), T, fill_value=np.nan)
    Tq = np.asarray(interp(query), dtype=float)
    bad = ~np.isfinite(Tq)
    if np.any(bad):
        Tq[bad] = griddata(pts, T, query[bad], method="nearest")
    Tq = Tq.reshape(len(x_values), len(yq))
    denominator = float(np.sum(wy * uq))
    if denominator <= 0.0:
        raise RuntimeError("nonpositive channel flow rate")
    return np.sum(Tq * (wy * uq)[None, :], axis=1) / denominator


def thermal_outputs(pts, T, tags, a_th, g_flux=G_FLUX, Umax=UMAX):
    """Return heat-transfer outputs from one nodal temperature field.

    Local bulk temperature is evaluated by 48-point Gauss-Legendre integration
    of the piecewise-linear interpolant on each heater-node cross-section.
    ``Nu=(g_flux/a_th)*Dh/(Twall-Tbulk)`` because
    ``q''/k = [q''/(rho cp)]/[k/(rho cp)] = g_flux/a_th``.
    """
    pts = np.asarray(pts, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.shape != (len(pts),):
        raise ValueError("T must contain one value per mesh node")
    a_th = float(a_th)
    if a_th <= 0.0:
        raise ValueError("a_th must be positive")
    heater = np.asarray(tags["heater_nodes"], dtype=int)
    if len(heater) == 0:
        raise ValueError("heater tag contains no nodes")
    wall_T = T[heater]
    local_bulk = _bulk_temperatures(
        pts, T, pts[heater, 0], Ly=tags["Ly"], Umax=Umax)
    delta_T = wall_T - local_bulk
    with np.errstate(divide="ignore", invalid="ignore"):
        Nu_local = (float(g_flux) / a_th) * (2.0 * tags["Ly"]) / delta_T
    imax = int(np.argmax(T))
    return {
        "Twall_max": float(np.max(wall_T)),
        "Twall_mean": float(np.mean(wall_T)),
        "Nu_mean": float(np.mean(Nu_local)),
        "Tbulk_out": float(_bulk_temperatures(
            pts, T, [tags["Lx"]], Ly=tags["Ly"], Umax=Umax)[0]),
        "hotspot_xy": (float(pts[imax, 0]), float(pts[imax, 1])),
    }


def to_channel_grid(pts, vals, Lx=LX, Ly=LY):
    """Cubic scattered interpolation onto the anchor's 64x64 grid.

    The first array axis is streamwise ``x``.  Any cubic interpolation holes on
    the convex-hull boundary are filled with nearest-neighbour values, matching
    the verified anchor's fallback policy.
    """
    xg = np.linspace(0.0, Lx, A.GRID_OBS)
    yg = np.linspace(0.0, Ly, A.GRID_OBS)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    query = np.column_stack((X.ravel(), Y.ravel()))
    out = griddata(pts, vals, query, method="cubic")
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = griddata(pts, vals, query[bad], method="nearest")
    return out.reshape(A.GRID_OBS, A.GRID_OBS)


def sig_from_grid(Ts, Tr):
    R = Ts - Tr
    Dlib, sl = A._fd_library(Ts)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c); return c / nrm if nrm > 0 else c


def reference_channel_grids(ics, fine_mesh=None, Pe=100.0,
                            nx=180, ny=60, seed=7001):
    """Solve one fine nominal-SUPG reference per IC and return 64x64 grids.

    Geometry is constructed once and reused across the entire population.  A
    caller may pass ``fine_mesh=(pts, elems, tags)`` to reuse an existing mesh.
    No reference is recomputed for different candidate schemes.
    """
    if fine_mesh is None:
        ref_pts, ref_elems, ref_tags = make_channel_mesh(nx, ny, seed)
    else:
        if len(fine_mesh) != 3:
            raise ValueError("fine_mesh must be (pts, elems, tags)")
        ref_pts, ref_elems, ref_tags = fine_mesh
    geom = channel_mesh_geometry(ref_pts, ref_elems, thermal_diffusivity(Pe))
    grids = []
    for ic in ics:
        T = assemble_channel("supg", ref_pts, ref_elems, ref_tags, Pe,
                             ic=ic, alpha=1.0, geom=geom)
        grids.append(to_channel_grid(ref_pts, T, ref_tags["Lx"], ref_tags["Ly"]))
    return grids


def _wall_wiggle(pts, T):
    bottom = np.where(np.isclose(pts[:, 1], 0.0))[0]
    bottom = bottom[np.argsort(pts[bottom, 0])]
    wall = T[bottom]
    scale = float(np.ptp(wall)) + 1.0e-30
    curvature = np.diff(wall, n=2)
    return {
        "index": float(np.sum(np.abs(curvature)) / scale),
        "sign_changes": int(np.sum(np.diff(np.sign(curvature)) != 0)),
        "undershoot": float(max(-np.min(T), 0.0)),
    }


def validate_physics(primary_mesh=None):
    """Run and print the four deterministic thermal-physics validations."""
    if primary_mesh is None:
        primary_mesh = make_channel_mesh(60, 20, seed=2026)
    pts, elems, tags = primary_mesh
    ic = _nominal_ic()

    print("\nPHYSICS VALIDATION")
    print("-" * 80)

    # V1: the discrete natural heater load must leave as advected enthalpy.
    Pe_energy = 100.0
    a_energy = thermal_diffusivity(Pe_energy)
    geom_energy = channel_mesh_geometry(pts, elems, a_energy)
    T_energy, tags_energy, meta_energy = assemble_channel(
        "supg", pts, elems, tags, Pe_energy, ic=ic, alpha=1.0,
        geom=geom_energy, return_meta=True)
    out_energy = thermal_outputs(
        pts, T_energy, tags_energy, a_energy, g_flux=ic["g_flux"])
    heat_added = meta_energy["heat_added"]
    outlet_enthalpy = UBAR * LY * out_energy["Tbulk_out"]
    energy_relerr = abs(outlet_enthalpy - heat_added) / abs(heat_added)
    print("[V1] ENERGY BALANCE (Pe=100, nominal SUPG alpha=1, T_in=0)")
    print(f"     heater input g_flux*Lh : {heat_added:.12e}")
    print(f"     outlet integral u*T dy : {outlet_enthalpy:.12e}")
    print(f"     relative error          : {energy_relerr:.6e}")

    # V2: three independently jittered refinements against a much finer mesh.
    Pe_conv = 100.0
    ref_mesh = make_channel_mesh(180, 60, seed=7001)
    rp, re, rt = ref_mesh
    rg = channel_mesh_geometry(rp, re, thermal_diffusivity(Pe_conv))
    Tr = assemble_channel("supg", rp, re, rt, Pe_conv, ic=ic,
                          alpha=1.0, geom=rg)
    Gr = to_channel_grid(rp, Tr)
    convergence = []
    for ny in (10, 15, 20, 30):
        nx = 3 * ny
        cp, ce, ct = make_channel_mesh(nx, ny, seed=9000 + ny)
        cg = channel_mesh_geometry(cp, ce, thermal_diffusivity(Pe_conv))
        Tc = assemble_channel("supg", cp, ce, ct, Pe_conv, ic=ic,
                              alpha=1.0, geom=cg)
        Gc = to_channel_grid(cp, Tc)
        rel_l2 = float(np.linalg.norm(Gc - Gr) / np.linalg.norm(Gr))
        convergence.append((nx, ny, rel_l2))
    conv_errors = np.array([row[2] for row in convergence])
    convergence_ok = bool(np.all(np.diff(conv_errors) < 0.0))
    print("[V2] GRID CONVERGENCE (Pe=100, SUPG vs 180x60 SUPG reference)")
    for nx, ny, error in convergence:
        print(f"     mesh {nx:3d}x{ny:<2d}: rel-L2={error:.6e}")
    print(f"     strictly decreasing      : {convergence_ok}")

    # V3: compare raw wall-node curvature and the maximum-principle violation.
    Pe_osc = 200.0
    geom_osc = channel_mesh_geometry(pts, elems, thermal_diffusivity(Pe_osc))
    T_gal = assemble_channel("galerkin", pts, elems, tags, Pe_osc,
                             ic=ic, alpha=0.0, geom=geom_osc)
    T_supg, tags_supg, meta_supg = assemble_channel(
        "supg", pts, elems, tags, Pe_osc, ic=ic, alpha=1.0,
        geom=geom_osc, return_meta=True)
    osc_gal, osc_supg = _wall_wiggle(pts, T_gal), _wall_wiggle(pts, T_supg)
    oscillation_ok = bool(
        osc_gal["index"] > osc_supg["index"]
        and osc_gal["undershoot"] > osc_supg["undershoot"])
    print("[V3] HIGH-Pe OSCILLATION (Pe=200, raw lower-wall node sequence)")
    print(f"     galerkin: wiggle={osc_gal['index']:.6f}, "
          f"curvature-sign-changes={osc_gal['sign_changes']:2d}, "
          f"T<0 undershoot={osc_gal['undershoot']:.6e}")
    print(f"     supg    : wiggle={osc_supg['index']:.6f}, "
          f"curvature-sign-changes={osc_supg['sign_changes']:2d}, "
          f"T<0 undershoot={osc_supg['undershoot']:.6e}")
    print(f"     Galerkin > SUPG           : {oscillation_ok}")

    # V4: the nominal heat-transfer result, not the deliberately biased foils.
    nu_by_pe = {}
    cached_nominal = {
        100: (T_energy, tags_energy, meta_energy, out_energy),
        200: (T_supg, tags_supg, meta_supg, thermal_outputs(
            pts, T_supg, tags_supg, thermal_diffusivity(200), g_flux=ic["g_flux"])),
    }
    for Pe in (50, 100, 200):
        if Pe not in cached_nominal:
            a_th = thermal_diffusivity(Pe)
            geom = channel_mesh_geometry(pts, elems, a_th)
            T, active, meta = assemble_channel(
                "supg", pts, elems, tags, Pe, ic=ic, alpha=1.0,
                geom=geom, return_meta=True)
            output = thermal_outputs(pts, T, active, a_th, g_flux=ic["g_flux"])
            cached_nominal[Pe] = (T, active, meta, output)
        nu_by_pe[Pe] = cached_nominal[Pe][3]["Nu_mean"]
    nu_ok = bool(all(1.0 <= value <= 20.0 for value in nu_by_pe.values()))
    print("[V4] NOMINAL-SUPG MEAN NUSSELT")
    for Pe in (50, 100, 200):
        print(f"     Pe={Pe:3d}: Nu_mean={nu_by_pe[Pe]:.6f}")
    print(f"     all positive and in [1,20]: {nu_ok}")

    physics_ok = bool(energy_relerr < 0.10 and convergence_ok and oscillation_ok)
    print(f"PHYSICS VALIDATION PASSES: {physics_ok}")
    return {
        "energy": {"heat_added": heat_added, "outlet_enthalpy": outlet_enthalpy,
                   "relative_error": energy_relerr},
        "convergence": convergence,
        "convergence_ok": convergence_ok,
        "oscillation": {"galerkin": osc_gal, "supg": osc_supg},
        "oscillation_ok": oscillation_ok,
        "Nu_by_Pe": nu_by_pe,
        "nu_ok": nu_ok,
        "passes": physics_ok,
        "cached_nominal": cached_nominal,
    }


def run_demo(primary_mesh=None, validation=None):
    """Solve the requested 15 representative cases and write the thermal CSV."""
    if primary_mesh is None:
        primary_mesh = make_channel_mesh(60, 20, seed=2026)
    pts, elems, tags = primary_mesh
    if validation is None:
        validation = validate_physics(primary_mesh)
    ic = _nominal_ic()
    cases = (
        ("galerkin", 0.0),
        ("supg", 1.0),
        ("supg", 0.5),
        ("supg", 2.0),
        ("artvisc", 1.0),
    )
    rows = []
    for Pe in (50, 100, 200):
        a_th = thermal_diffusivity(Pe)
        geom = channel_mesh_geometry(pts, elems, a_th)
        for scheme, alpha in cases:
            if scheme == "supg" and alpha == 1.0:
                T, active_tags, meta, output = validation["cached_nominal"][Pe]
            else:
                T, active_tags, meta = assemble_channel(
                    scheme, pts, elems, tags, Pe, ic=ic, alpha=alpha,
                    geom=geom, return_meta=True)
                output = thermal_outputs(
                    pts, T, active_tags, a_th, g_flux=ic["g_flux"])
            rows.append({
                "Pe": Pe,
                "scheme": scheme,
                "alpha": alpha,
                "Twall_max": output["Twall_max"],
                "Twall_mean": output["Twall_mean"],
                "Nu_mean": output["Nu_mean"],
                "Tbulk_out": output["Tbulk_out"],
                "hotspot_x": output["hotspot_xy"][0],
                "hotspot_y": output["hotspot_xy"][1],
            })

    columns = ("Pe", "scheme", "alpha", "Twall_max", "Twall_mean",
               "Nu_mean", "Tbulk_out", "hotspot_x", "hotspot_y")
    csv_path = os.path.join(TAB, "heated_channel.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.10f}" if isinstance(value, float) else value)
                for key, value in row.items()
            })

    print("\nDEMO THERMAL OUTPUTS (representative IC)")
    print("-" * 112)
    print(" Pe  scheme     alpha   Twall_max  Twall_mean    Nu_mean  Tbulk_out  hotspot")
    for row in rows:
        print(f"{row['Pe']:3d}  {row['scheme']:<9s} {row['alpha']:5.1f}  "
              f"{row['Twall_max']:10.5f}  {row['Twall_mean']:10.5f}  "
              f"{row['Nu_mean']:9.5f}  {row['Tbulk_out']:9.5f}  "
              f"({row['hotspot_x']:.3f},{row['hotspot_y']:.3f})")

    print("\nDETUNED SUPG SILENT HEAT-TRANSFER BIAS (relative to alpha=1)")
    print("-" * 80)
    print(" Pe  alpha    Nu_mean   delta_Nu[%]  Twall_max  delta_Twall_max[%]")
    for Pe in (50, 100, 200):
        nominal = next(row for row in rows
                       if row["Pe"] == Pe and row["scheme"] == "supg"
                       and row["alpha"] == 1.0)
        for alpha in (0.5, 2.0):
            row = next(row for row in rows
                       if row["Pe"] == Pe and row["scheme"] == "supg"
                       and row["alpha"] == alpha)
            dnu = 100.0 * (row["Nu_mean"] / nominal["Nu_mean"] - 1.0)
            dtw = 100.0 * (row["Twall_max"] / nominal["Twall_max"] - 1.0)
            print(f"{Pe:3d}  {alpha:5.1f}  {row['Nu_mean']:9.5f}  {dnu:+11.4f}  "
                  f"{row['Twall_max']:10.5f}  {dtw:+18.4f}")
    print(f"\nCSV -> {csv_path}")
    return rows, csv_path


def main():
    started = time.perf_counter()
    print("=" * 80)
    print("2D HEATED LAMINAR CHANNEL: CONVECTIVE HEAT-TRANSFER FOUNDATION")
    print("=" * 80)
    print(f"domain=[0,{LX}]x[0,{LY}], Umax={UMAX}, Ubar={UBAR:.12f}")
    print(f"localized heater=[{XH0},{XH1}], g_flux={G_FLUX}, Dh={DH}")
    print("P1 heater edge load: F_i += g_flux*edge_length/2 at each endpoint")
    print("a_th=Ubar*Ly/Pe; inlet T=0; other walls/outlet are natural")
    print(f"signature grid={A.GRID_OBS}x{A.GRID_OBS}; audit IC population N_IC={N_IC}")
    primary_mesh = make_channel_mesh(60, 20, seed=2026)
    print(f"working mesh: {len(primary_mesh[0])} nodes, {len(primary_mesh[1])} triangles")
    validation = validate_physics(primary_mesh)
    rows, csv_path = run_demo(primary_mesh, validation)
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    return {"validation": validation, "rows": rows,
            "csv": csv_path, "runtime_seconds": elapsed}


if __name__ == "__main__":
    main()
