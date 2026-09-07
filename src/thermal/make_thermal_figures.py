#!/usr/bin/env python3
"""
Publication figures for the heated-channel (ICHMT) case.

Three figures, written to figures/:
  fig_channel_field.png       - the heated-channel setup: 2D SUPG temperature field with the
                                wall heater marked, and the lower-wall temperature trace showing
                                Galerkin node-to-node oscillation vs the monotone SUPG field (high Pe).
  fig_silent_bias.png         - silent stabilization bias: mean Nusselt and peak wall temperature
                                vs the SUPG strength alpha, across Pe. The thermal outputs move only
                                ~1%, i.e. a detuning is thermally "silent".
  fig_signature_vs_thermal.png - detection sensitivity vs detuning for the modified-equation
                                signature vs conventional thermal outputs (max wall T, mean Nu,
                                full-field distance): the signature crosses 95% while the thermal
                                outputs stay flat -> it detects the numerical change first.

Self-contained: imports the validated heated_channel foundation and reads the committed
results/tables/thermal_detection_limit.csv. Deterministic. CPU. Run:
    python src/thermal/make_thermal_figures.py
"""
import os, sys, csv
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
import heated_channel as HC
TAB = os.path.join(_ROOT, "results", "tables")
FIGS = os.path.join(_ROOT, "figures"); os.makedirs(FIGS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
except Exception:
    pass
plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, GREEN, RED, GREY, PURP, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0", "#dd8452"


def _vertical_profile(pts, T, x_cut, ny=64, Ly=1.0, Lx=3.0):
    """Temperature along a vertical line x=x_cut, from the 64x64 interpolated grid."""
    grid = HC.to_channel_grid(pts, T, Lx, Ly)
    gx = np.linspace(0, Lx, grid.shape[0])
    gy = np.linspace(0, Ly, grid.shape[1])
    col = int(np.argmin(np.abs(gx - x_cut)))
    return gy, grid[col, :]


def fig_channel_field():
    Pe = 200.0
    ic = HC.make_thermal_ic(1000)
    # Panel A: fine-mesh SUPG field (clean, physical). Panel B: a COARSE mesh where the
    # unstabilized Galerkin oscillation (spurious T<0 at the plume edge) is visible; SUPG removes it.
    fp, fe, ftags = HC.make_channel_mesh(90, 30, seed=2026)
    fgeom = HC.channel_mesh_geometry(fp, fe, HC.thermal_diffusivity(Pe))
    T_supg_fine = HC.assemble_channel("supg", fp, fe, ftags, Pe, ic=ic, alpha=1.0, geom=fgeom)
    grid = HC.to_channel_grid(fp, T_supg_fine, ftags["Lx"], ftags["Ly"])
    grid = np.clip(grid, 0.0, None)              # T>=0 for display (kills cubic-interp speckles)
    gx = np.linspace(0, ftags["Lx"], grid.shape[0])
    gy = np.linspace(0, ftags["Ly"], grid.shape[1])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")

    cp, ce, ctags = HC.make_channel_mesh(30, 10, seed=2026)   # coarse: Galerkin oscillates
    cgeom = HC.channel_mesh_geometry(cp, ce, HC.thermal_diffusivity(Pe))
    T_gal = HC.assemble_channel("galerkin", cp, ce, ctags, Pe, ic=ic, alpha=1.0, geom=cgeom)
    T_sup = HC.assemble_channel("supg", cp, ce, ctags, Pe, ic=ic, alpha=1.0, geom=cgeom)
    x_cut = float(cp[int(np.argmin(T_gal)), 0])               # station of worst Galerkin undershoot
    yg, Pg = _vertical_profile(cp, T_gal, x_cut, Ly=ctags["Ly"], Lx=ctags["Lx"])
    _, Ps = _vertical_profile(cp, T_sup, x_cut, Ly=ctags["Ly"], Lx=ctags["Lx"])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.2, 6.4),
                                   gridspec_kw={"height_ratios": [1.05, 1.0]})
    fig.subplots_adjust(hspace=0.5)
    im = axA.contourf(GX, GY, grid, levels=24, cmap="inferno")
    axA.set_aspect("equal"); axA.set_xlabel("$x$"); axA.set_ylabel("$y$")
    axA.set_title(f"SUPG temperature field, $Pe={Pe:.0f}$ (heater on lower wall)", fontsize=10)
    hx = fp[np.asarray(ftags["heater_nodes"], int), 0]
    axA.plot([hx.min(), hx.max()], [0, 0], color="cyan", lw=4, solid_capstyle="butt",
             label="heater ($q''$)")
    axA.legend(loc="upper right", frameon=True, fontsize=8)
    fig.colorbar(im, ax=axA, fraction=0.024, pad=0.02, label="$T$")
    axA.text(-0.08, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    axB.plot(Pg, yg, color=RED, lw=1.4, marker="o", ms=3.2, label="Galerkin (unstabilized)")
    axB.plot(Ps, yg, color=BLUE, lw=1.6, marker="s", ms=3.2, label="SUPG (stabilized)")
    axB.axvline(0, color=GREY, lw=0.9, ls="--")
    xmin = min(Pg.min(), 0) - 0.5
    axB.axvspan(xmin, 0, color=RED, alpha=0.06)
    axB.text(min(Pg.min(), 0) * 0.5, 0.9, "spurious $T<0$\n(unphysical)", color=RED,
             fontsize=8, ha="center", va="top")
    axB.set_xlim(left=xmin)
    axB.set_ylabel("$y$"); axB.set_xlabel("temperature $T$")
    axB.set_title(f"Vertical profile at $x\\approx{x_cut:.2f}$ (coarse mesh): "
                  f"Galerkin undershoots, SUPG stays physical", fontsize=9.5)
    axB.legend(frameon=False, fontsize=8, loc="upper right")
    axB.text(-0.08, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIGS, "fig_channel_field.png")
    fig.savefig(out); plt.close(fig); print(f"  -> {out}")


def fig_silent_bias():
    alphas = np.round(np.linspace(0.3, 2.0, 18), 4)
    Pes = [50.0, 100.0, 200.0]
    ic = HC.make_thermal_ic(1000)
    pts, elems, tags = HC.make_channel_mesh(60, 20, seed=2026)
    colors = {50.0: BLUE, 100.0: GREEN, 200.0: RED}
    Nu = {Pe: [] for Pe in Pes}; Tw = {Pe: [] for Pe in Pes}
    for Pe in Pes:
        a_th = HC.thermal_diffusivity(Pe)
        geom = HC.channel_mesh_geometry(pts, elems, a_th)
        for a in alphas:
            T = HC.assemble_channel("supg", pts, elems, tags, Pe, ic=ic, alpha=float(a), geom=geom)
            o = HC.thermal_outputs(pts, T, tags, a_th)
            Nu[Pe].append(o["Nu_mean"]); Tw[Pe].append(o["Twall_max"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9)); fig.subplots_adjust(wspace=0.3)
    for Pe in Pes:
        nu = np.array(Nu[Pe]); tw = np.array(Tw[Pe])
        nu0 = nu[np.argmin(np.abs(alphas - 1.0))]; tw0 = tw[np.argmin(np.abs(alphas - 1.0))]
        axA.plot(alphas, 100 * (nu / nu0 - 1), "-o", color=colors[Pe], ms=3, label=f"$Pe={Pe:.0f}$")
        axB.plot(alphas, 100 * (tw / tw0 - 1), "-o", color=colors[Pe], ms=3, label=f"$Pe={Pe:.0f}$")
    for ax, name in ((axA, "mean Nusselt"), (axB, "peak wall $T$")):
        ax.axvline(1.0, color=GREY, lw=1, ls="--")
        ax.set_xlabel(r"SUPG strength $\alpha=\tau/\tau_{\rm nom}$")
        ax.set_ylabel(f"{name} deviation from $\\alpha=1$ [%]")
        ax.legend(frameon=False, fontsize=8)
    axA.set_title("Silent bias in Nusselt number", fontsize=10)
    axB.set_title("Silent bias in peak wall temperature", fontsize=10)
    axA.text(-0.16, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")
    axB.text(-0.16, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    out = os.path.join(FIGS, "fig_silent_bias.png")
    fig.savefig(out); plt.close(fig); print(f"  -> {out}")


def _read_curves(Pe, sigma, direction):
    """Return {detector: (deltas, sensitivities)} from the committed CSV."""
    path = os.path.join(TAB, "thermal_detection_limit.csv")
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["type"] != "curve":
                continue
            if abs(float(row["Pe"]) - Pe) > 1e-6 or abs(float(row["sigma"]) - sigma) > 1e-9:
                continue
            if row["direction"] != direction:
                continue
            d = data.setdefault(row["detector"], ([], []))
            d[0].append(float(row["delta_alpha"])); d[1].append(float(row["sensitivity"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in data.items()}


def fig_signature_vs_thermal():
    Pe, sigma = 100.0, 0.01
    styles = {"signature": (BLUE, "o", "modified-equation signature"),
              "thermal_pair": ("#9467bd", "v", "joint wall $T$ and mean Nusselt"),
              "Twall_max": (RED, "s", "peak wall $T$"),
              "Nu_mean": (ORANGE, "^", "mean Nusselt"),
              "fullfield_T": (GREY, "D", "full-field $T$ distance")}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True); fig.subplots_adjust(wspace=0.12)
    for ax, direction in zip(axes, ("weaken", "strengthen")):
        curves = _read_curves(Pe, sigma, direction)
        for det, (col, mk, lab) in styles.items():
            if det not in curves:
                continue
            dd, ss = curves[det]
            order = np.argsort(dd)
            ax.plot(dd[order], ss[order], "-", marker=mk, color=col, ms=4, lw=1.8,
                    label=lab if direction == "weaken" else None)
        ax.axhline(0.95, color="k", lw=1, ls=(0, (3, 2)))
        ax.text(0.02, 0.965, "95% sensitivity", fontsize=7.5, transform=ax.get_yaxis_transform())
        ax.set_xlabel(r"detuning $|\Delta\alpha|$")
        ax.set_ylim(-0.02, 1.03)
        ax.set_title(f"{direction} ($Pe={Pe:.0f}$, $\\sigma={sigma}$)", fontsize=10)
    axes[0].set_ylabel("detection sensitivity at the 5% split-conformal level")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.suptitle("Detection of stabilization detuning against matched thermal baselines",
                 fontsize=11, y=1.02)
    out = os.path.join(FIGS, "fig_signature_vs_thermal.png")
    fig.savefig(out); plt.close(fig); print(f"  -> {out}")


def main():
    print("Generating heated-channel (ICHMT) publication figures:")
    fig_channel_field()
    fig_silent_bias()
    fig_signature_vs_thermal()
    print("done.")


if __name__ == "__main__":
    main()
