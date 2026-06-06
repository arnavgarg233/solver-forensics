import os, sys, numpy as np
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT,"src","audit")); import supg_2d_engineering as A
TAB=os.path.join(_ROOT,"results","tables"); os.makedirs(TAB,exist_ok=True)

"""Steady conjugate thermal foundation for a five-cell prismatic module.

The model is deliberately a steady, two-dimensional longitudinal-section
assessment.  It is a fast, defensible fixed-operating-point model whose coolant
subproblem is strongly advection dominated, making it suitable for the SUPG
stabilization audit built on top of this foundation.

PROPERTY NOTE -- every value below is a *representative, literature-sourced,
replaceable* engineering property.  They are generic Li-ion, aluminium, and
liquid-water values; no value is fitted to, or claimed to reproduce, a specific
paper or experiment.  In SI units:

    cell:     k=20.0 W/(m K) [in-plane/vertical effective], rho*cp=2.50e6 J/(m^3 K)
    gap:      k=0.3 W/(m K), no volumetric generation
    cold plate: k=237 W/(m K), rho*cp=2.42e6 J/(m^3 K)
    coolant:  k=0.6 W/(m K), rho*cp=4.18e6 J/(m^3 K)
    sources:  1C=2.0e4, 2C=6.0e4, 3C=1.2e5 W/m^3

The physical equation is

    div(k grad(T)) + q''' - rho_cp_f (u . grad(T)) = 0.

For assembly it is divided everywhere by the constant coolant rho_cp_f.  This
does not change T or the conjugate interface fluxes, but gives the coolant
advection and SUPG matrices the canonical engineering form requested here:

    k/rho_cp_f * A * grad(N_i).grad(N_j)
    + A/3 * u.grad(N_j)
    + alpha*tau*A*(u.grad(N_i))*(u.grad(N_j)).

Physical W per unit out-of-plane depth are reconstructed for the energy balance.
All external boundaries except the coolant inlet are natural adiabatic/outflow
boundaries.  The top is therefore insulated by construction: high 3C predicted
temperatures are a screening result, not a claim of safe battery operation.

Continuous-Galerkin P1 fields are not locally flux-conservative at a mixed
Dirichlet/adiabatic corner: directly fixing the coolant/plate inlet vertex can
create a finite mesh-dependent reaction at a zero-measure point.  To enforce the
specified global physical balance without changing any element diffusion,
advection, or SUPG operator, the default solve appends one scalar Lagrange
constraint using the exact parabolic outlet enthalpy functional.  This is a
conservative flux closure; its multiplier is reported in solve metadata.
"""

import csv
import time

from scipy.interpolate import griddata
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------------------------------------------------------
# Geometry and representative, literature-sourced, replaceable properties
# -----------------------------------------------------------------------------
MM = 1.0e-3
HC = 3.0 * MM
HP = 3.0 * MM
WC = 27.5 * MM
HCELL = 91.0 * MM
GAP = 2.0 * MM
NCELL = 5
LX = NCELL * WC + (NCELL - 1) * GAP
LY = HC + HP + HCELL

# All values in this block are representative, literature-sourced, replaceable.
KB = 20.0                # cell conductivity [W/(m K)]: in-plane (along-height) effective value
                         # for a prismatic jelly roll (~20-30 in-plane vs ~0.5-1 through-plane;
                         # the dominant heat path to the base cold plate is the in-plane/vertical
                         # direction). Representative, literature-sourced, replaceable.
RHOCP_B = 2.50e6         # cell volumetric heat capacity [J/(m^3 K)]
KG = 0.3                 # filler-gap conductivity [W/(m K)]
KP = 237.0               # aluminium conductivity [W/(m K)]
RHOCP_P = 2.42e6         # aluminium volumetric heat capacity [J/(m^3 K)]
KF = 0.6                 # liquid-water conductivity [W/(m K)]
RHOCP_F = 4.18e6         # liquid-water volumetric heat capacity [J/(m^3 K)]
Q_BY_CRATE = {"1C": 2.0e4, "2C": 6.0e4, "3C": 1.2e5}  # W/m^3
UBAR_NOMINAL = 0.10      # representative, replaceable coolant bulk velocity [m/s]
T_IN = 25.0              # coolant inlet temperature [degC]
ATH_F = KF / RHOCP_F
SIGNATURE_WINDOW = (0.0, LX, 0.0, HC + HP)
N_IC = 24

CELL_SPANS = tuple(
    (i * (WC + GAP), i * (WC + GAP) + WC) for i in range(NCELL)
)
K_BY_REGION = {"coolant": KF, "plate": KP, "cell": KB, "gap": KG}

# Re-export the audit helpers so later battery analyses use the exact same
# feature/classification implementation as the engineering anchor.
feats = A.feats
cv_acc = A.cv_acc
perm_floor = A.perm_floor
_clf = A._clf


def coolant_velocity(y, Ubar=UBAR_NOMINAL):
    """Plane-Poiseuille coolant velocity in m/s; zero outside [0, HC]."""
    Ubar = float(Ubar)
    if not np.isfinite(Ubar) or Ubar <= 0.0:
        raise ValueError("Ubar must be finite and positive")
    y = np.asarray(y, dtype=float)
    Umax = 1.5 * Ubar
    inside = (y >= 0.0) & (y <= HC)
    value = Umax * (1.0 - ((y - 0.5 * HC) / (0.5 * HC)) ** 2)
    return np.where(inside, np.maximum(value, 0.0), 0.0)


def module_peclet(Ubar=UBAR_NOMINAL):
    """Long-channel coolant Peclet number Ubar*LX/a_th,f."""
    return float(Ubar) * LX / ATH_F


def region(x, y):
    """Return material region(s) at coordinates; cells occupy the upper band."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    ans = np.full(x.shape, "gap", dtype="<U8")
    tol = 2.0e-13
    ans[y <= HC + tol] = "coolant"
    ans[(y > HC + tol) & (y <= HC + HP + tol)] = "plate"
    upper = y > HC + HP + tol
    for x0, x1 in CELL_SPANS:
        ans[upper & (x >= x0 - tol) & (x <= x1 + tol)] = "cell"
    return str(ans) if ans.ndim == 0 else ans


def material_regions(pts, elems):
    """Element material map evaluated at P1-triangle centroids."""
    centroids = np.asarray(pts, dtype=float)[np.asarray(elems, dtype=int)].mean(axis=1)
    return region(centroids[:, 0], centroids[:, 1])


def _boundary_edges(elems):
    """Undirected edges occurring in exactly one P1 triangle."""
    edges = np.vstack((elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def tag_module_boundaries(pts, elems, Lx=LX, Ly=LY, Hc=HC):
    """Tag coolant inlet/outlet and all external natural-boundary edges.

    Only left-boundary coolant nodes are Dirichlet.  All remaining exterior
    edges intentionally stay natural: adiabatic solid walls and a natural
    advective coolant outflow.
    """
    pts = np.asarray(pts, dtype=float)
    bedges = _boundary_edges(np.asarray(elems, dtype=int))
    ep = pts[bedges]
    tol = 1.0e-10 * max(Lx, Ly)
    left = np.all(np.abs(ep[:, :, 0]) <= tol, axis=1)
    right = np.all(np.abs(ep[:, :, 0] - Lx) <= tol, axis=1)
    bottom = np.all(np.abs(ep[:, :, 1]) <= tol, axis=1)
    top = np.all(np.abs(ep[:, :, 1] - Ly) <= tol, axis=1)
    inlet = left & (np.max(ep[:, :, 1], axis=1) <= Hc + tol)
    outlet_coolant = right & (np.min(ep[:, :, 1], axis=1) >= -tol) & (
        np.max(ep[:, :, 1], axis=1) <= Hc + tol)
    recognized = left | right | bottom | top
    if not np.all(recognized):
        raise RuntimeError("module mesh contains an unclassified external boundary edge")

    def nodes_of(mask):
        return np.unique(bedges[mask].ravel()) if np.any(mask) else np.empty(0, dtype=int)

    return {
        "inlet_nodes": nodes_of(inlet),
        "inlet_edges": bedges[inlet],
        "outlet_nodes": nodes_of(right),
        "coolant_outlet_edges": bedges[outlet_coolant],
        "natural_edges": bedges[~inlet],
        "boundary_edges": bedges,
        "Lx": float(Lx), "Ly": float(Ly), "Hc": float(Hc),
        "window": SIGNATURE_WINDOW,
    }


def _append_segment(coords, start, width, count):
    """Append one interface-aligned coordinate segment without duplicate joins."""
    count = int(count)
    if count < 1:
        raise ValueError("every material segment needs at least one interval")
    return np.r_[coords, start + width * np.arange(1, count + 1) / count]


def make_module_mesh(n_cell_x=16, n_cell_y=None, n_cool=None, n_plate=None,
                     seed=2026, nx=None, ny=None):
    """Build a deterministic conforming P1 mesh of the full conjugate rectangle.

    Material interfaces are exact coordinate lines rather than centroid-smeared
    cuts.  Alternating diagonal orientation (seed-controlled phase) preserves a
    simple vectorized P1 layout while avoiding one preferred diagonal direction.
    ``n_cell_x`` is the number of intervals across each 27.5-mm cell; ``ny`` is
    accepted as an alias for vertical cell intervals to mirror mesh-builder APIs.
    """
    if nx is not None:
        n_cell_x = nx
    if ny is not None:
        n_cell_y = ny
    n_cell_x = int(n_cell_x)
    if n_cell_x < 4:
        raise ValueError("n_cell_x must be at least 4")
    dx_cell = WC / n_cell_x
    if n_cell_y is None:
        n_cell_y = max(12, int(round(HCELL / dx_cell)))
    if n_cool is None:
        n_cool = max(4, int(round(HC / dx_cell)))
    if n_plate is None:
        n_plate = max(4, int(round(HP / dx_cell)))
    n_cell_y, n_cool, n_plate = int(n_cell_y), int(n_cool), int(n_plate)

    xs = np.array([0.0])
    x0 = 0.0
    n_gap = max(1, int(round(n_cell_x * GAP / WC)))
    for cell in range(NCELL):
        xs = _append_segment(xs, x0, WC, n_cell_x)
        x0 += WC
        if cell < NCELL - 1:
            xs = _append_segment(xs, x0, GAP, n_gap)
            x0 += GAP
    ys = np.array([0.0])
    ys = _append_segment(ys, 0.0, HC, n_cool)
    ys = _append_segment(ys, HC, HP, n_plate)
    ys = _append_segment(ys, HC + HP, HCELL, n_cell_y)
    xs[-1], ys[-1] = LX, LY  # protect exact boundary tags from roundoff.

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((X.ravel(), Y.ravel()))
    nxp, nyp = len(xs), len(ys)
    ii, jj = np.meshgrid(np.arange(nxp - 1), np.arange(nyp - 1), indexing="xy")
    a = (jj * nxp + ii).ravel()
    b = a + 1
    c = a + nxp
    d = c + 1
    parity = (ii.ravel() + jj.ravel() + int(seed)) % 2
    elems = np.empty((2 * len(a), 3), dtype=int)
    first = parity == 0
    elems[0::2][first] = np.column_stack((a[first], b[first], c[first]))
    elems[1::2][first] = np.column_stack((b[first], d[first], c[first]))
    elems[0::2][~first] = np.column_stack((a[~first], b[~first], d[~first]))
    elems[1::2][~first] = np.column_stack((a[~first], d[~first], c[~first]))

    tags = tag_module_boundaries(pts, elems)
    tags.update({
        "xs": xs, "ys": ys, "n_cell_x": n_cell_x, "n_cell_y": n_cell_y,
        "n_cool": n_cool, "n_plate": n_plate,
        "regions": material_regions(pts, elems),
    })
    return pts, elems, tags


def module_mesh_geometry(pts, elems):
    """Precompute static P1 geometry/material data once per module mesh.

    The velocity and SUPG time scale depend on Ubar, so they are evaluated by
    ``_flow_terms`` per operating point; triangle geometry, material map, sparse
    index pattern, and exact triangle y-moments remain reusable across all solves.
    """
    pts = np.asarray(pts, dtype=float)
    elems = np.asarray(elems, dtype=int)
    p = pts[elems]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    Ae = 0.5 * np.abs(detJ)
    if np.any(Ae <= 1.0e-16):
        raise ValueError("module mesh contains a degenerate triangle")
    b = np.stack((y2 - y3, y3 - y1, y1 - y2), axis=1) / detJ[:, None]
    c = np.stack((x3 - x2, x1 - x3, x2 - x1), axis=1) / detJ[:, None]
    regions = material_regions(pts, elems)
    k = np.array([K_BY_REGION[name] for name in regions], dtype=float)
    coolant = regions == "coolant"
    cell = regions == "cell"
    centroids = p.mean(axis=1)
    cell_index = np.full(len(elems), -1, dtype=int)
    tol = 2.0e-13
    for index, (x0, x1_) in enumerate(CELL_SPANS):
        hit = cell & (centroids[:, 0] >= x0 - tol) & (centroids[:, 0] <= x1_ + tol)
        cell_index[hit] = index
    if np.any(cell & (cell_index < 0)):
        raise RuntimeError("cell element could not be assigned to one of five cells")
    yv = p[:, :, 1]
    y_mean = yv.mean(axis=1)
    y2_mean = (np.sum(yv * yv, axis=1)
               + yv[:, 0] * yv[:, 1] + yv[:, 0] * yv[:, 2]
               + yv[:, 1] * yv[:, 2]) / 6.0
    ii = np.repeat(elems[:, :, None], 3, axis=2)
    jj = np.repeat(elems[:, None, :], 3, axis=1)
    return {
        "elems": elems, "Ae": Ae, "b": b, "c": c, "p": p,
        "regions": regions, "k": k, "kappa": k / RHOCP_F,
        "coolant": coolant, "cell": cell, "cell_index": cell_index,
        "y_mean": y_mean, "y2_mean": y2_mean, "h_e": np.sqrt(2.0 * Ae),
        "grad_dot": b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :],
        "rows": ii.ravel(), "cols": jj.ravel(), "npt": len(pts),
    }


def _flow_terms(geom, Ubar):
    """Exact area-mean Poiseuille velocity and nominal coolant SUPG tau."""
    Ubar = float(Ubar)
    if not np.isfinite(Ubar) or Ubar <= 0.0:
        raise ValueError("Ubar must be finite and positive")
    Umax = 1.5 * Ubar
    u = Umax * (4.0 * geom["y_mean"] / HC - 4.0 * geom["y2_mean"] / HC ** 2)
    u = np.where(geom["coolant"], np.maximum(u, 0.0), 0.0)
    h_e = geom["h_e"]
    tau = np.zeros_like(u)
    active = geom["coolant"]
    tau[active] = 1.0 / np.sqrt(
        (2.0 * np.abs(u[active]) / h_e[active]) ** 2
        + (4.0 * ATH_F / h_e[active] ** 2) ** 2
    )
    return {"u_e": u, "adg": u[:, None] * geom["b"], "tau_nom": tau,
            "Umax": Umax, "Ubar": Ubar}


def matched_artificial_diffusion(geom, Ubar=UBAR_NOMINAL, alpha=1.0):
    """Area-weighted mean coolant ``alpha*tau*|u|^2`` [m^2/s]."""
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    flow = _flow_terms(geom, Ubar)
    mask = geom["coolant"]
    denominator = float(np.sum(geom["Ae"][mask]))
    if denominator <= 0.0:
        raise RuntimeError("mesh has no coolant elements")
    return float(np.sum(geom["Ae"][mask] * alpha * flow["tau_nom"][mask]
                        * flow["u_e"][mask] ** 2) / denominator)


def make_module_ic(seed):
    """Deterministic operating perturbation for an audit population.

    The requested population varies source amplitude by +/-10%, each of the five
    cell sources by +/-5%, bulk flow by +/-10%, and a zero-flow-mean inlet ramp
    by +/-1 K.  It is intentionally an operating-condition population, not
    fabricated experimental data.
    """
    rng = np.random.default_rng(seed)
    return {
        "q_scale": float(rng.uniform(0.90, 1.10)),
        "cell_jitter": rng.uniform(0.95, 1.05, NCELL),
        "ubar_scale": float(rng.uniform(0.90, 1.10)),
        "inlet_ramp": float(rng.uniform(-1.0, 1.0)),
    }


def _nominal_ic():
    return {"q_scale": 1.0, "cell_jitter": np.ones(NCELL),
            "ubar_scale": 1.0, "inlet_ramp": 0.0}


def _merged_ic(ic):
    data = _nominal_ic()
    if ic is not None:
        data.update(ic)
    data["q_scale"] = float(data["q_scale"])
    data["ubar_scale"] = float(data["ubar_scale"])
    data["inlet_ramp"] = float(data["inlet_ramp"])
    data["cell_jitter"] = np.asarray(data["cell_jitter"], dtype=float)
    if data["cell_jitter"].shape != (NCELL,):
        raise ValueError("cell_jitter must contain five factors")
    if data["q_scale"] <= 0.0 or data["ubar_scale"] <= 0.0:
        raise ValueError("q_scale and ubar_scale must be positive")
    return data


def _q_base_value(q_base):
    if isinstance(q_base, str):
        if q_base not in Q_BY_CRATE:
            raise ValueError("q_base string must be one of 1C, 2C, 3C")
        q_base = Q_BY_CRATE[q_base]
    q_base = float(q_base)
    if not np.isfinite(q_base) or q_base < 0.0:
        raise ValueError("q_base must be finite and nonnegative")
    return q_base


def _cell_generation(geom, q_base, ic):
    """Elementwise q''' with source only in cell material."""
    q = np.zeros_like(geom["Ae"])
    cell = geom["cell"]
    q[cell] = (q_base * ic["q_scale"]
               * ic["cell_jitter"][geom["cell_index"][cell]])
    return q


def assemble_module(scheme, pts, elems, tags=None, q_base=Q_BY_CRATE["2C"],
                    Ubar=UBAR_NOMINAL, ic=None, alpha=1.0, geom=None,
                    nu_art=None, return_meta=False, energy_closure=True):
    """Assemble and solve the monolithic steady P1 conjugate thermal system.

    ``scheme`` is ``galerkin``, ``supg``, or ``artvisc``.  Advection and SUPG
    are deliberately nonzero only for coolant triangles; q''' is deliberately
    nonzero only for cell triangles.  The artificial-viscosity foil receives an
    isotropic coolant-only increment matched to the nominal SUPG added diffusion.
    """
    if scheme not in ("galerkin", "supg", "artvisc"):
        raise ValueError("scheme must be 'galerkin', 'supg', or 'artvisc'")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if tags is None:
        tags = tag_module_boundaries(pts, elems)
    if geom is None:
        geom = module_mesh_geometry(pts, elems)
    q_base = _q_base_value(q_base)
    data = _merged_ic(ic)
    actual_ubar = float(Ubar) * data["ubar_scale"]
    flow = _flow_terms(geom, actual_ubar)
    Ae, adg = geom["Ae"], flow["adg"]
    Ke = (geom["kappa"] * Ae)[:, None, None] * geom["grad_dot"]
    # P1 advection: int(N_i)=A/3.  u is zero in every non-coolant element.
    Ke += (Ae[:, None, None] / 3.0) * adg[:, None, :]

    actual_nu_art = 0.0
    if scheme == "supg":
        tau = alpha * flow["tau_nom"]
        Ke += (tau * Ae)[:, None, None] * (adg[:, :, None] * adg[:, None, :])
    elif scheme == "artvisc":
        actual_nu_art = (matched_artificial_diffusion(geom, actual_ubar, alpha)
                          if nu_art is None else float(nu_art))
        if actual_nu_art < 0.0:
            raise ValueError("nu_art must be nonnegative")
        Ke += ((actual_nu_art * geom["coolant"]) * Ae)[:, None, None] * geom["grad_dot"]

    q_e = _cell_generation(geom, q_base, data)
    Fe = (Ae * q_e / (3.0 * RHOCP_F))[:, None] * np.ones((1, 3))
    F = np.bincount(geom["elems"].ravel(), weights=Fe.ravel(),
                    minlength=geom["npt"]).astype(float)
    K = csr_matrix((Ke.ravel(), (geom["rows"], geom["cols"])),
                   shape=(geom["npt"], geom["npt"])).tolil()

    inlet = np.asarray(tags["inlet_nodes"], dtype=int)
    if len(inlet) == 0:
        raise RuntimeError("module mesh has no coolant inlet nodes")
    # This profile has exactly zero velocity-weighted mean relative to T_IN.
    inlet_values = T_IN + data["inlet_ramp"] * (2.0 * pts[inlet, 1] / HC - 1.0)
    for node, value in zip(inlet, inlet_values):
        K.rows[int(node)] = [int(node)]
        K.data[int(node)] = [1.0]
        F[int(node)] = value
    K = K.tocsr()
    total_generation = float(np.sum(q_e * Ae))
    outlet_functional = _edge_temperature_functional(
        pts, tags["coolant_outlet_edges"], actual_ubar, geom["npt"])
    inlet_functional = _edge_temperature_functional(
        pts, tags["inlet_edges"], actual_ubar, geom["npt"])
    inlet_profile = np.zeros(geom["npt"], dtype=float)
    inlet_profile[inlet] = inlet_values
    inlet_enthalpy = float(inlet_functional @ inlet_profile)
    energy_multiplier = 0.0
    if energy_closure:
        # The constraint is the physical conservation identity in the same
        # globally rho_cp_f-scaled units as K and F:
        # int_out u*T dy = sum(q''' A)/rho_cp_f + int_in u*T_in dy.
        ccol = csr_matrix(outlet_functional[:, None])
        augmented = bmat([[K, ccol], [ccol.T, None]], format="csr")
        rhs = np.r_[F, total_generation / RHOCP_F + inlet_enthalpy]
        solved = np.asarray(spsolve(augmented, rhs), dtype=float)
        T, energy_multiplier = solved[:-1], float(solved[-1])
        closure_residual = K @ T + outlet_functional * energy_multiplier - F
        residual = float(np.linalg.norm(closure_residual)
                         / (np.linalg.norm(F) + 1.0e-30))
    else:
        T = np.asarray(spsolve(K, F), dtype=float)
        residual = float(np.linalg.norm(K @ T - F) / (np.linalg.norm(F) + 1.0e-30))
    if not np.all(np.isfinite(T)):
        raise RuntimeError("module solve produced non-finite temperatures")
    meta = {
        "q_base": q_base, "q_e": q_e,
        "total_generation": total_generation,
        "Ubar": actual_ubar, "Umax": flow["Umax"], "Pe": module_peclet(actual_ubar),
        "alpha": alpha if scheme != "galerkin" else 0.0,
        "nu_art": actual_nu_art, "linear_residual": residual,
        "inlet_values": inlet_values, "ic": data,
        "energy_closure": bool(energy_closure),
        "energy_multiplier": energy_multiplier,
        "outlet_flow_integral": float(np.sum(outlet_functional)),
    }
    return (T, geom["regions"].copy(), meta) if return_meta else T


def _outlet_edges_from_points(pts):
    """Infer coolant outlet P1 edges for a structured interface-aligned mesh."""
    pts = np.asarray(pts, dtype=float)
    tol = 1.0e-10 * max(LX, LY)
    nodes = np.where((np.abs(pts[:, 0] - LX) <= tol) & (pts[:, 1] <= HC + tol))[0]
    nodes = nodes[np.argsort(pts[nodes, 1])]
    if len(nodes) < 2:
        raise RuntimeError("could not identify coolant outlet edges")
    return np.column_stack((nodes[:-1], nodes[1:]))


def _edge_temperature_functional(pts, edges, Ubar, npt=None):
    """Return c with ``c @ T = integral_edges u(y)*T ds`` for P1 traces."""
    pts = np.asarray(pts, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if npt is None:
        npt = len(pts)
    c = np.zeros(int(npt), dtype=float)
    xi, w = np.polynomial.legendre.leggauss(4)
    shape0, shape1 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
    for n0, n1 in edges:
        p0, p1 = pts[n0], pts[n1]
        length = float(np.linalg.norm(p1 - p0))
        y = shape0 * p0[1] + shape1 * p1[1]
        uq = coolant_velocity(y, Ubar)
        factor = 0.5 * length
        c[n0] += float(np.sum(w * uq * shape0) * factor)
        c[n1] += float(np.sum(w * uq * shape1) * factor)
    return c


def _outlet_integrals(pts, T, Ubar, edges=None):
    """Exactly-enough Gauss integration of P1 T times parabolic outlet velocity."""
    pts, T = np.asarray(pts, dtype=float), np.asarray(T, dtype=float)
    if edges is None:
        edges = _outlet_edges_from_points(pts)
    edges = np.asarray(edges, dtype=int)
    xi, w = np.polynomial.legendre.leggauss(4)
    int_u = 0.0
    int_uT = 0.0
    for n0, n1 in edges:
        p0, p1 = pts[n0], pts[n1]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 0.0:
            continue
        shape0, shape1 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
        y = shape0 * p0[1] + shape1 * p1[1]
        tq = shape0 * T[n0] + shape1 * T[n1]
        uq = coolant_velocity(y, Ubar)
        factor = 0.5 * length
        int_u += float(np.sum(w * uq) * factor)
        int_uT += float(np.sum(w * uq * tq) * factor)
    if int_u <= 0.0:
        raise RuntimeError("nonpositive coolant outlet flow integral")
    return int_u, int_uT


def _cell_node_masks(pts):
    """Five node masks used for cell extrema and per-cell mean temperatures."""
    pts = np.asarray(pts, dtype=float)
    tol = 1.0e-10 * max(LX, LY)
    upper = pts[:, 1] >= HC + HP - tol
    return [upper & (pts[:, 0] >= x0 - tol) & (pts[:, 0] <= x1 + tol)
            for x0, x1 in CELL_SPANS]


def thermal_outputs_module(pts, T, regions, tags=None, Ubar=UBAR_NOMINAL,
                           total_generation=None, T_in=T_IN):
    """Return thermal outputs and the coolant enthalpy energy-balance diagnostic.

    ``regions`` is accepted explicitly because analyses commonly retain the
    element material vector alongside T.  The output extrema follow the request
    exactly: all cell nodes for Tmax/dTmax and one nodal mean per physical cell
    for sigma_cell.  Supply ``total_generation`` from ``assemble_module`` for
    an energy-error value; otherwise it is reported as NaN rather than guessed.
    """
    pts, T = np.asarray(pts, dtype=float), np.asarray(T, dtype=float)
    if T.shape != (len(pts),):
        raise ValueError("T must contain one temperature per mesh node")
    if tags is None:
        outlet_edges = _outlet_edges_from_points(pts)
    else:
        outlet_edges = np.asarray(tags["coolant_outlet_edges"], dtype=int)
    int_u, int_uT = _outlet_integrals(pts, T, Ubar, outlet_edges)
    Tout = int_uT / int_u
    Qcool = RHOCP_F * (int_uT - float(T_in) * int_u)
    masks = _cell_node_masks(pts)
    if any(not np.any(mask) for mask in masks):
        raise RuntimeError("one or more cell node masks are empty")
    cell_nodes = np.any(np.column_stack(masks), axis=1)
    cell_T = T[cell_nodes]
    cell_means = np.array([np.mean(T[mask]) for mask in masks], dtype=float)
    if total_generation is None:
        energy_err = float("nan")
    else:
        total_generation = float(total_generation)
        energy_err = (abs(total_generation - Qcool) / abs(total_generation)
                      if total_generation > 0.0 else float("nan"))
    return {
        "Tmax": float(np.max(cell_T)),
        "dTmax": float(np.ptp(cell_T)),
        "sigma_cell": float(np.std(cell_means)),
        "Tout": float(Tout), "Qcool": float(Qcool),
        "energy_err": float(energy_err),
        "cell_means": cell_means,
        "outlet_flow_integral": float(int_u),
    }


def to_module_grid(pts, vals, window=SIGNATURE_WINDOW):
    """Cubic scattered interpolation to the audit's regular coolant/plate grid.

    The first grid axis is x.  Cubic boundary holes are filled with nearest
    values, matching the heated-channel and anchor fallback policy.
    """
    if len(window) != 4:
        raise ValueError("window must be (xmin, xmax, ymin, ymax)")
    xmin, xmax, ymin, ymax = map(float, window)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("window bounds must increase")
    xg = np.linspace(xmin, xmax, A.GRID_OBS)
    yg = np.linspace(ymin, ymax, A.GRID_OBS)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    query = np.column_stack((X.ravel(), Y.ravel()))
    out = griddata(pts, vals, query, method="cubic")
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = griddata(pts, vals, query[bad], method="nearest")
    return out.reshape(A.GRID_OBS, A.GRID_OBS)


def sig_from_grid(Ts,Tr):
    R=Ts-Tr; Dlib,sl=A._fd_library(Ts)
    Amat=np.column_stack([Dlib[n].ravel() for n in A.LIB]); b=R[sl,sl].ravel()
    c,*_=np.linalg.lstsq(Amat,b,rcond=None); nrm=np.linalg.norm(c); return c/nrm if nrm>0 else c


def reference_module_grids(ics, fine_mesh=None, q_base=Q_BY_CRATE["2C"],
                           Ubar=UBAR_NOMINAL, n_cell_x=26, seed=7001,
                           window=SIGNATURE_WINDOW):
    """Return one fine nominal-SUPG coolant/plate grid per requested IC.

    Static mesh geometry is constructed once.  Each IC retains its requested
    source, per-cell, flow, and inlet perturbation, while reference stabilization
    stays nominal SUPG (alpha=1) for every solve.
    """
    if fine_mesh is None:
        pts, elems, tags = make_module_mesh(n_cell_x=n_cell_x, seed=seed)
    else:
        if len(fine_mesh) != 3:
            raise ValueError("fine_mesh must be (pts, elems, tags)")
        pts, elems, tags = fine_mesh
    geom = module_mesh_geometry(pts, elems)
    grids = []
    for ic in ics:
        T = assemble_module("supg", pts, elems, tags, q_base=q_base,
                            Ubar=Ubar, ic=ic, alpha=1.0, geom=geom)
        grids.append(to_module_grid(pts, T, window))
    return grids


def _coolant_wiggle(pts, T):
    """High-frequency node-to-node curvature on the coolant mid-height row."""
    pts, T = np.asarray(pts, dtype=float), np.asarray(T, dtype=float)
    target = 0.5 * HC
    yvals = np.unique(pts[pts[:, 1] <= HC + 1.0e-12, 1])
    row_y = yvals[np.argmin(np.abs(yvals - target))]
    tol = 1.0e-11 * max(LX, LY)
    nodes = np.where(np.abs(pts[:, 1] - row_y) <= tol)[0]
    nodes = nodes[np.argsort(pts[nodes, 0])]
    line = T[nodes]
    span = float(np.ptp(line)) + 1.0e-14
    curvature = np.diff(line, n=2)
    coolant_nodes = T[pts[:, 1] <= HC + tol]
    return {
        "wiggle": float(np.sum(np.abs(curvature)) / span),
        "sign_changes": int(np.sum(np.diff(np.sign(curvature)) != 0)),
        "undershoot": float(max(T_IN - np.min(coolant_nodes), 0.0)),
        "row_y": float(row_y),
    }


def _solve_output(scheme, pts, elems, tags, geom, q_base, Ubar, ic=None,
                  alpha=1.0):
    """Small internal helper that keeps every validation solve physically aligned."""
    T, regions, meta = assemble_module(
        scheme, pts, elems, tags, q_base=q_base, Ubar=Ubar, ic=ic,
        alpha=alpha, geom=geom, return_meta=True)
    output = thermal_outputs_module(
        pts, T, regions, tags=tags, Ubar=meta["Ubar"],
        total_generation=meta["total_generation"])
    return T, regions, meta, output


def validate_physics(primary_mesh=None):
    """Run and print the four deterministic physics gates required for the module."""
    if primary_mesh is None:
        primary_mesh = make_module_mesh(n_cell_x=16, seed=2026)
    pts, elems, tags = primary_mesh
    geom = module_mesh_geometry(pts, elems)
    nominal_ic = _nominal_ic()

    print("\nPHYSICS VALIDATION")
    print("-" * 88)
    print(f"nominal Ubar={UBAR_NOMINAL:.4f} m/s, Umax={1.5 * UBAR_NOMINAL:.4f} m/s, "
          f"Pe=Ubar*Lx/a_th,f={module_peclet(UBAR_NOMINAL):.3e}")

    # V1: global discrete test function means source must leave as fluid enthalpy.
    T_nom, regions_nom, meta_nom, out_nom = _solve_output(
        "supg", pts, elems, tags, geom, Q_BY_CRATE["2C"], UBAR_NOMINAL,
        nominal_ic, alpha=1.0)
    energy_ok = bool(out_nom["energy_err"] < 0.01)
    print("[V1] ENERGY BALANCE (2C, nominal SUPG alpha=1)")
    print(f"     total cell generation [W/m] : {meta_nom['total_generation']:.10e}")
    print(f"     coolant Qcool [W/m]         : {out_nom['Qcool']:.10e}")
    print(f"     relative error               : {out_nom['energy_err']:.6e}  (<1%: {energy_ok})")

    # V2: independently assembled interface-aligned refinements versus a fine mesh.
    levels = (8, 12, 16)
    fine_level = 24
    convergence = []
    for level in levels + (fine_level,):
        mp, me, mt = make_module_mesh(n_cell_x=level, seed=2026)
        mg = module_mesh_geometry(mp, me)
        _, _, _, out = _solve_output("supg", mp, me, mt, mg, Q_BY_CRATE["2C"],
                                     UBAR_NOMINAL, nominal_ic, alpha=1.0)
        convergence.append({"level": level, "nodes": len(mp), "triangles": len(me),
                            "Tmax": out["Tmax"], "dTmax": out["dTmax"],
                            "Tout": out["Tout"]})
    reference = convergence[-1]
    for row in convergence:
        for key in ("Tmax", "dTmax", "Tout"):
            row[f"err_{key}"] = abs(row[key] - reference[key]) / max(abs(reference[key]), 1.0)
    coarse_rows = convergence[:-1]
    temperature_errors = {
        key: np.array([row[f"err_{key}"] for row in coarse_rows])
        for key in ("Tmax", "dTmax", "Tout")
    }
    thermo_decreasing = all(np.all(np.diff(temperature_errors[key]) < 0.0)
                           for key in ("Tmax", "dTmax"))
    # Tout is energy-constrained and should be mesh-invariant to numerical precision.
    tout_converged = bool(np.max(temperature_errors["Tout"]) < 1.0e-5)
    convergence_ok = bool(thermo_decreasing and tout_converged)
    print("[V2] MESH CONVERGENCE (nominal SUPG; 24-per-cell reference)")
    print("     ncellx   nodes  triangles       Tmax       dTmax       Tout"
          "    rel-to-fine(Tmax,dTmax,Tout)")
    for row in convergence:
        print(f"     {row['level']:6d} {row['nodes']:7d} {row['triangles']:10d} "
              f"{row['Tmax']:10.4f} {row['dTmax']:11.4f} {row['Tout']:10.5f}  "
              f"({row['err_Tmax']:.3e}, {row['err_dTmax']:.3e}, {row['err_Tout']:.3e})")
    print(f"     Tmax/dTmax errors decrease: {thermo_decreasing}; "
          f"Tout mesh-invariant: {tout_converged}")

    # V3 uses the requested nominal operating point on the working mesh.
    T_gal, _, _, out_gal = _solve_output(
        "galerkin", pts, elems, tags, geom, Q_BY_CRATE["2C"], UBAR_NOMINAL,
        nominal_ic, alpha=0.0)
    osc_gal = _coolant_wiggle(pts, T_gal)
    osc_supg = _coolant_wiggle(pts, T_nom)
    oscillation_ok = bool(osc_gal["wiggle"] > osc_supg["wiggle"])
    print("[V3] COOLANT OSCILLATION (2C, Ubar=0.1 m/s, coolant-midline nodes)")
    print(f"     galerkin: wiggle={osc_gal['wiggle']:.6f}, "
          f"curvature-sign-changes={osc_gal['sign_changes']}, "
          f"T<25 undershoot={osc_gal['undershoot']:.6e} C")
    print(f"     supg    : wiggle={osc_supg['wiggle']:.6f}, "
          f"curvature-sign-changes={osc_supg['sign_changes']}, "
          f"T<25 undershoot={osc_supg['undershoot']:.6e} C")
    print(f"     Galerkin wiggle > SUPG: {oscillation_ok}")

    # V4: source-rate and flow trends on exactly the same physical mesh.
    rate_cache = {"2C": (T_nom, regions_nom, meta_nom, out_nom)}
    for crate in ("1C", "3C"):
        rate_cache[crate] = _solve_output(
            "supg", pts, elems, tags, geom, Q_BY_CRATE[crate], UBAR_NOMINAL,
            nominal_ic, alpha=1.0)
    _, _, _, out_high_flow = _solve_output(
        "supg", pts, elems, tags, geom, Q_BY_CRATE["2C"], 0.20,
        nominal_ic, alpha=1.0)
    Tmax_by_rate = {crate: rate_cache[crate][3]["Tmax"] for crate in Q_BY_CRATE}
    trends = bool(Tmax_by_rate["1C"] < Tmax_by_rate["2C"] < Tmax_by_rate["3C"]
                  and out_high_flow["Tmax"] < Tmax_by_rate["2C"])
    # This is a numerical-plausibility screening band, never a safety threshold.
    plausible = bool(all(25.0 < value < 350.0 for value in Tmax_by_rate.values()))
    trends_ok = bool(trends and plausible)
    print("[V4] PHYSICAL TRENDS (nominal SUPG)")
    for crate in ("1C", "2C", "3C"):
        print(f"     {crate}: Tmax={Tmax_by_rate[crate]:.4f} C")
    print(f"     2C, Ubar=0.20 m/s: Tmax={out_high_flow['Tmax']:.4f} C")
    print(f"     C-rate rising / higher-flow cooling: {trends}; "
          f"screening range 25<Tmax<350 C: {plausible}")

    physics_ok = bool(energy_ok and convergence_ok and oscillation_ok and trends_ok)
    print(f"PHYSICS VALIDATION PASSES: {physics_ok}")
    return {
        "energy": {"generation": meta_nom["total_generation"], "Qcool": out_nom["Qcool"],
                   "relative_error": out_nom["energy_err"], "ok": energy_ok},
        "convergence": convergence, "convergence_ok": convergence_ok,
        "oscillation": {"galerkin": osc_gal, "supg": osc_supg, "ok": oscillation_ok,
                        "galerkin_output": out_gal},
        "trends": {"Tmax_by_rate": Tmax_by_rate, "high_flow_Tmax": out_high_flow["Tmax"],
                   "ok": trends_ok},
        "passes": physics_ok, "rate_cache": rate_cache,
    }


def run_demo(primary_mesh=None, validation=None):
    """Solve 15 requested operating-point/scheme cases and write the CSV table."""
    if primary_mesh is None:
        primary_mesh = make_module_mesh(n_cell_x=16, seed=2026)
    pts, elems, tags = primary_mesh
    geom = module_mesh_geometry(pts, elems)
    if validation is None:
        validation = validate_physics(primary_mesh)
    nominal_ic = _nominal_ic()
    cases = (("galerkin", 0.0), ("supg", 1.0), ("supg", 0.5),
             ("supg", 2.0), ("artvisc", 1.0))
    rows = []
    for crate in ("1C", "2C", "3C"):
        for scheme, alpha in cases:
            if scheme == "supg" and alpha == 1.0 and crate in validation["rate_cache"]:
                T, regions, meta, output = validation["rate_cache"][crate]
            else:
                T, regions, meta, output = _solve_output(
                    scheme, pts, elems, tags, geom, Q_BY_CRATE[crate], UBAR_NOMINAL,
                    nominal_ic, alpha=alpha)
            rows.append({
                "Crate": crate, "scheme": scheme, "alpha": float(alpha),
                "Tmax": output["Tmax"], "dTmax": output["dTmax"],
                "sigma_cell": output["sigma_cell"], "Tout": output["Tout"],
                "Qcool": output["Qcool"], "energy_err": output["energy_err"],
            })

    columns = ("Crate", "scheme", "alpha", "Tmax", "dTmax", "sigma_cell",
               "Tout", "Qcool", "energy_err")
    csv_path = os.path.join(TAB, "battery_module.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.10f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    print("\nDEMO THERMAL OUTPUTS (representative nominal operating point)")
    print("-" * 112)
    print("Crate  scheme     alpha       Tmax      dTmax  sigma_cell       Tout"
          "        Qcool   energy_err")
    for row in rows:
        print(f"{row['Crate']:>4s}  {row['scheme']:<9s} {row['alpha']:5.1f} "
              f" {row['Tmax']:10.4f} {row['dTmax']:10.4f} {row['sigma_cell']:11.5f}"
              f" {row['Tout']:10.5f} {row['Qcool']:12.5f} {row['energy_err']:.3e}")

    print("\nDETUNED SUPG SILENT THERMAL BIAS (relative to alpha=1)")
    print("-" * 92)
    print("Crate  alpha       Tmax  delta Tmax [C]      dTmax  delta dTmax [C]")
    for crate in ("1C", "2C", "3C"):
        nominal = next(row for row in rows if row["Crate"] == crate
                       and row["scheme"] == "supg" and row["alpha"] == 1.0)
        for alpha in (0.5, 2.0):
            row = next(row for row in rows if row["Crate"] == crate
                       and row["scheme"] == "supg" and row["alpha"] == alpha)
            print(f"{crate:>4s}  {alpha:5.1f} {row['Tmax']:10.4f} "
                  f"{row['Tmax'] - nominal['Tmax']:+17.6f} "
                  f"{row['dTmax']:10.4f} {row['dTmax'] - nominal['dTmax']:+18.6f}")
    print(f"\nCSV -> {csv_path}")
    return rows, csv_path


# Convenient aliases for analyses that mirror the heated-channel naming.
make_mesh = make_module_mesh
assemble = assemble_module
to_grid = to_module_grid


def main():
    """Run the validated foundation demo without plotting or external data access."""
    started = time.perf_counter()
    print("=" * 88)
    print("BATTERY THERMAL MANAGEMENT: STEADY 2D FIVE-CELL CONJUGATE FOUNDATION")
    print("=" * 88)
    print(f"domain=[0,{LX:.6f}] x [0,{LY:.6f}] m; coolant={HC/MM:.1f} mm, "
          f"plate={HP/MM:.1f} mm, five cells={WC/MM:.1f} x {HCELL/MM:.1f} mm")
    print("properties are representative, literature-sourced, replaceable; no experimental match claimed")
    print(f"cell k={KB:g}, rho*cp={RHOCP_B:.3e}; gap k={KG:g}; plate k={KP:g}, "
          f"rho*cp={RHOCP_P:.3e}; water k={KF:g}, rho*cp={RHOCP_F:.3e}")
    print(f"q''' [W/m^3]: 1C={Q_BY_CRATE['1C']:.3e}, 2C={Q_BY_CRATE['2C']:.3e}, "
          f"3C={Q_BY_CRATE['3C']:.3e}; Tin={T_IN:.1f} C")
    print(f"a_th,f={ATH_F:.6e} m^2/s; nominal Ubar={UBAR_NOMINAL:.3f} m/s; "
          f"Pe={module_peclet():.3e}")
    print("BCs: coolant inlet Dirichlet; all other external boundaries natural/adiabatic; "
          "outlet enthalpy is integrated with parabolic u(y)")
    print(f"signature window=[0,{LX:.6f}] x [0,{HC + HP:.6f}] m; "
          f"grid={A.GRID_OBS}x{A.GRID_OBS}; audit IC population N={N_IC}")
    primary_mesh = make_module_mesh(n_cell_x=16, seed=2026)
    print(f"working mesh: {len(primary_mesh[0])} nodes, {len(primary_mesh[1])} triangles")
    validation = validate_physics(primary_mesh)
    rows, csv_path = run_demo(primary_mesh, validation)
    elapsed = time.perf_counter() - started
    print(f"RUNTIME_SECONDS: {elapsed:.3f}")
    return {"validation": validation, "rows": rows, "csv": csv_path,
            "runtime_seconds": elapsed}


if __name__ == "__main__":
    main()
