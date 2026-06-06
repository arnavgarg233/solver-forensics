#!/usr/bin/env python3
"""
Honest battery-regime figure for the solver-forensics limits story.

CORRECTED (the earlier "Peclet ceiling" reading was falsified by the same-geometry
Peclet sweep, src/thermal/pe_sweep.py): in the OPEN heated channel, subtle-detuning
detection stays at sensitivity 1.0 all the way to Pe=1e5 at 1% noise. So the battery
null is NOT a Peclet effect. It is an APPLICATION / OBSERVABILITY limit: in the conjugate
battery the safety-relevant output (peak cell temperature) is dominated by solid
conduction, so the coolant-borne stabilization signal is diluted and buried by noise.

Panel A: battery signature detection sensitivity vs detuning at three noise levels
         (2C nominal, weakening) -- detects at sigma=0, collapses at sigma>=1%.
Panel B: SAME Pe (~1e5), OPPOSITE outcome -- the open channel detects (sens~1.0) while
         the conjugate battery does not: the limit is observability, not Peclet.

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


def _battery_sens(delta=0.5, sigma=0.01):
    with open(os.path.join(TAB, "battery_detection.csv")) as f:
        for r in csv.DictReader(f):
            if (r["type"] == "curve" and r["detector"] == "signature"
                    and r["direction"] == "weaken" and r["operating_point"] == "2C_nominal"
                    and abs(float(r["sigma"]) - sigma) < 1e-9
                    and abs(float(r["delta_alpha"]) - delta) < 1e-6):
                return float(r["sensitivity"])
    return 0.0


def _channel_sens(pe=100000.0, alpha=0.5, noise=0.01):
    with open(os.path.join(TAB, "pe_sweep.csv")) as f:
        for r in csv.DictReader(f):
            if (abs(float(r["Pe"]) - pe) < 1e-3 and abs(float(r["alpha"]) - alpha) < 1e-6
                    and abs(float(r["noise"]) - noise) < 1e-9):
                return float(r["detection_sensitivity"])
    return float("nan")


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.1)); fig.subplots_adjust(wspace=0.3)

    # Panel A: the battery noise cliff
    curves = _battery_signature_curves()
    cmap = {0.0: GREEN, 0.01: ORANGE, 0.05: RED}
    for s in sorted(curves):
        dd, ss = curves[s]
        order = np.argsort(dd)
        axA.plot(dd[order], ss[order], "-o", ms=4, color=cmap.get(s, GREY), label=f"$\\sigma={s}$")
    axA.axhline(0.90, color="k", lw=1, ls=(0, (3, 2)))
    axA.text(0.02, 0.915, "90% sensitivity target", fontsize=7.5, transform=axA.get_yaxis_transform())
    axA.set_xlabel(r"stabilization detuning $|\Delta\alpha|$ (weakening)")
    axA.set_ylabel("signature detection sensitivity")
    axA.set_ylim(-0.03, 1.05)
    axA.set_title("Battery ($Pe\\approx10^5$): the coolant signal is buried at $\\sigma\\geq1\\%$", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8, loc="center left")
    axA.text(-0.13, 1.04, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # Panel B: SAME Pe, opposite outcome -> observability limit, NOT Peclet
    ch = _channel_sens(100000.0, 0.5, 0.01)
    bat = _battery_sens(0.5, 0.01)
    labels = ["open channel\n(same $Pe\\approx10^5$)", "conjugate battery\n($Pe\\approx10^5$)"]
    vals = [ch, bat]
    colors = [GREEN, RED]
    axB.bar([0, 1], vals, color=colors, width=0.6)
    axB.text(0, ch + 0.03, f"{ch:.2f}\n(detected)", ha="center", fontsize=8.5, color=GREEN)
    axB.text(1, bat + 0.03, f"{bat:.2f}\n(undetected)", ha="center", fontsize=8.5, color=RED)
    axB.axhline(0.90, color="k", lw=1, ls=(0, (3, 2)))
    axB.set_xticks([0, 1]); axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylim(0, 1.15)
    axB.set_ylabel(r"detection sensitivity, $\alpha=0.5$, $\sigma=0.01$")
    axB.set_title("Same Peclet, opposite outcome: an observability limit", fontsize=9.5)
    axB.text(-0.13, 1.04, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")
    axB.text(0.5, 0.55, "not a Peclet ceiling:\nthe channel detects at the\nsame $Pe$. The battery's\nsafety output is solid-\nconduction-dominated,\nso the coolant signal is diluted.",
             ha="center", va="center", fontsize=7.6, color=GREY, transform=axB.transAxes,
             bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))

    fig.suptitle("The battery null is an observability limit, not a Peclet effect "
                 "(controlled sweep: channel detects to $Pe=10^5$)", fontsize=10.5, y=1.02)
    out = os.path.join(FIGS, "fig_battery_regime.png")
    fig.savefig(out); plt.close(fig); print(f"  -> {out}")


if __name__ == "__main__":
    main()
