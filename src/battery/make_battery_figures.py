#!/usr/bin/env python3
"""Battery screening figure for the solver-forensics limits result.

Panel A shows the battery signature sensitivity against weakening at three noise
levels. Panel B compares a subtle detuning with complete Galerkin replacement in
the same battery model at one percent noise. The figure makes no Péclet-number or
causal attenuation claim.

Reads the committed CSVs; deterministic. Run:
    python src/battery/make_battery_figures.py
"""
import os, sys, csv
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
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
BLUE, GREEN, RED, GREY, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#dd8452"


def _battery_signature_curves():
    """{sigma: (deltas, sens)} for signature, weaken, 2C_nominal."""
    out = {}
    with open(os.path.join(TAB, "battery_detection.csv")) as f:
        for r in csv.DictReader(f):
            if (r["type"] == "curve" and r["detector"] == "signature"
                    and r["direction"] == "weaken" and r["operating_point"] == "2C_nominal"):
                s = float(r["sigma"])
                d = out.setdefault(s, ([], []))
                d[0].append(float(r["delta_alpha"])); d[1].append(float(r["sensitivity"]))
    return {s: (np.array(v[0]), np.array(v[1])) for s, v in out.items()}


def _control_rate(row):
    with open(os.path.join(TAB, "battery_controls.csv")) as f:
        for record in csv.DictReader(f):
            if int(record["row"]) == int(row):
                return float(record["sensitivity"])
    return float("nan")


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.1)); fig.subplots_adjust(wspace=0.3)

    curves = _battery_signature_curves()
    cmap = {0.0: GREEN, 0.01: ORANGE, 0.05: RED}
    for s in sorted(curves):
        dd, ss = curves[s]
        order = np.argsort(dd)
        axA.plot(dd[order], ss[order], "-o", ms=4, color=cmap.get(s, GREY), label=f"$\\sigma={s}$")
    axA.axhline(0.90, color="k", lw=1, ls=(0, (3, 2)))
    axA.text(0.02, 0.915, "90% target", fontsize=7.5, transform=axA.get_yaxis_transform())
    axA.set_xlabel(r"stabilization detuning $|\Delta\alpha|$ (weakening)")
    axA.set_ylabel("signature firing rate")
    axA.set_ylim(-0.03, 1.05)
    axA.set_title("Battery screening across observation-noise levels", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8, loc="center left")
    axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    detuning = _control_rate(1)
    galerkin = _control_rate(2)
    labels = ["subtle detuning\n($\\alpha=0.5$)", "Galerkin\nreplacement"]
    vals = [detuning, galerkin]
    colors = [RED, GREEN]
    axB.bar([0, 1], vals, color=colors, width=0.6)
    axB.text(0, detuning + 0.03, f"{detuning:.3f}\nmissed", ha="center", fontsize=8.5, color=RED)
    axB.text(1, galerkin + 0.03, f"{galerkin:.3f}\ndetected", ha="center", fontsize=8.5, color=GREEN)
    axB.axhline(0.90, color="k", lw=1, ls=(0, (3, 2)))
    axB.set_xticks([0, 1]); axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylim(0, 1.15)
    axB.set_ylabel(r"firing rate at $\sigma=0.01$")
    axB.set_title("Small and wholesale numerical changes", fontsize=9.5)
    axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    fig.suptitle("Battery screening: subtle detuning is missed at one percent noise",
                 fontsize=10.5, y=1.02)
    out = os.path.join(FIGS, "fig_battery_regime.png")
    fig.savefig(out); plt.close(fig); print(f"  -> {out}")


if __name__ == "__main__":
    main()
