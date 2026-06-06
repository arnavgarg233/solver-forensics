#!/usr/bin/env python3
"""
solver-forensics :: MECHANICS DEMONSTRATION (elastic wave / structural dynamics)
================================================================================
Forensic attribution of the discretization choice in computational mechanics, on the
1D elastic-wave / elastodynamics equation

    u_tt = c^2 u_xx        (periodic rod, u_t(.,0) = 0),

whose reference is the analytic modal solution u(x,t) = Re IFFT[ FFT(u0) cos(c|k|t) ] -
exact, so there is no reference error. We attribute three textbook structural-dynamics
discretizations that differ only in the MASS MATRIX and the TIME INTEGRATOR, each with a
distinct, known numerical-dispersion / numerical-dissipation signature:

  lumped_CD      lumped-mass + central-difference (explicit)  -> dispersive, phase LAG
  consistent_FEM consistent linear-FEM mass + central-diff    -> dispersive, phase LEAD
  newmark_damped Newmark-beta (gamma=0.6) numerical damping    -> dissipative (amplitude decay)

The forensic signature is the unit-normalized modified-equation coefficient direction
recovered from the residual r = u_solver - u_ref on the library {u_xx, u_xxx, u_xxxx}
(derivatives of the observed solver field, matching the 1D/2D audits). Tasks:
  ID3            3-way scheme identification (accuracy vs permutation floor)
  dissipation    damped (newmark) vs non-dissipative (the two central-difference schemes)
  mass_matrix    lumped vs consistent mass - the fine same-order distinction
  NC1 / NC2      initial-condition+noise and grid-resolution controls

Self-contained: numpy + scikit-learn, analytic reference, deterministic. CPU, ~10-20 s.
Run:  python src/mechanics/wave_attribution.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB, FIGS = os.path.join(_ROOT, "results", "tables"), os.path.join(_ROOT, "figures")

L, C, T = 1.0, 1.0, 1.0
N_DET, NC2_GRIDS, N_OBS, N_IC = 128, (96, 160), 128, 60     # observe at native resolution
LIB_ORDERS, SIGMA = (2, 3, 4), 0.01
SCHEMES = ("lumped_CD", "consistent_FEM", "newmark_damped")
DISSIPATIVE = {"newmark_damped"}

def _k(N): return 2 * np.pi * np.fft.fftfreq(N, d=L / N)

def exact(u0, t):
    k = _k(len(u0)); return np.real(np.fft.ifft(np.fft.fft(u0) * np.cos(C * np.abs(k) * t)))

def _omega2(N, scheme):
    k = _k(N); dx = L / N
    Khat = C ** 2 * (2 - 2 * np.cos(k * dx)) / dx ** 2          # linear-FEM / central-difference stiffness
    Mhat = (2 + np.cos(k * dx)) / 3 if scheme == "consistent_FEM" else np.ones_like(k)
    return Khat / Mhat                                          # semidiscrete squared frequency Omega_k^2

def run(scheme, N, u0):
    dx = L / N; Om2 = _omega2(N, scheme)
    dt = 0.5 * dx / C; ns = int(round(T / dt)); dt = T / ns
    uh = np.fft.fft(u0)
    if scheme in ("lumped_CD", "consistent_FEM"):              # explicit central difference (leapfrog)
        um1 = uh; u = (1 - 0.5 * dt ** 2 * Om2) * uh           # Taylor first step (v0 = 0)
        for _ in range(ns - 1):
            u, um1 = (2 - dt ** 2 * Om2) * u - um1, u
        return np.real(np.fft.ifft(u))
    g, b = 0.6, 0.25 * (0.6 + 0.5) ** 2                        # Newmark-beta with numerical damping
    u = uh.copy(); v = np.zeros_like(uh); a = -Om2 * u
    for _ in range(ns):
        un = (u + dt * v + dt ** 2 * (0.5 - b) * a) / (1 + b * dt ** 2 * Om2)
        an = -Om2 * un
        v = v + dt * ((1 - g) * a + g * an); u, a = un, an
    return np.real(np.fft.ifft(u))

def observe(field, ex, sigma, M, seed):
    g = np.random.default_rng(seed)
    nz = sigma * np.sqrt(np.mean(field ** 2)) * g.standard_normal(field.shape) if sigma > 0 else 0.0
    # Fourier resample to a common M-point observation grid (handles non-integer ratios)
    def resample(u):
        Fh = np.fft.rfft(u); out = np.zeros(M // 2 + 1, complex); m = min(len(Fh), len(out))
        out[:m] = Fh[:m]; return np.fft.irfft(out, n=M) * (M / len(u))
    return resample(field + nz), resample(field + nz - ex)

def signature(u_obs, r_obs):
    M = len(u_obs); k = _k(M); uh = np.fft.fft(u_obs)
    A = np.stack([np.real(np.fft.ifft(uh * (1j * k) ** p)) for p in LIB_ORDERS], 1)
    c, *_ = np.linalg.lstsq(A, r_obs, rcond=None)
    n = np.linalg.norm(c); return c / n if n > 0 else c

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def acc(F, y, g): return cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean()
def perm_floor(F, y, g, seed, reps=30):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean() for _ in range(reps)]))

def random_ic(N, rng, n_modes=5):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 6) * x / L + rng.uniform(0, 2 * np.pi))
    return u / (np.std(u) + 1e-9)

def sigs(scheme, N, ics, sigma, seed):
    out = []
    for i, u0 in enumerate(ics):
        uf = run(scheme, N, u0); ex = exact(u0, T)
        uo, ro = observe(uf, ex, sigma, N, seed + i)        # observe at native grid N
        out.append(signature(uo, ro))
    return np.array(out)

def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(0); ics = [random_ic(N_DET, rng) for _ in range(N_IC)]
    ic = np.arange(N_IC); half = N_IC // 2
    print(f"mechanics: elastic-wave scheme attribution | u_tt=c^2 u_xx, c={C}, T={T}, N={N_DET}, "
          f"{N_IC} ICs, schemes {SCHEMES}\n")

    F = {s: sigs(s, N_DET, ics, SIGMA, 100 + 1000 * i) for i, s in enumerate(SCHEMES)}

    # ID3: 3-way scheme identification
    X = np.vstack([F[s] for s in SCHEMES]); y = np.concatenate([np.full(N_IC, i) for i in range(3)])
    g = np.concatenate([ic] * 3)
    id3, id3f = acc(X, y, g), perm_floor(X, y, g, 7)
    print(f"ID3   3-way scheme identification:   acc={id3:.3f}   floor={id3f:.3f}   (chance 0.33)")

    # dissipation: damped (newmark) vs the two central-difference schemes - IMBALANCED 60:120,
    # so the permutation floor (~0.667 majority baseline) is reported, not an implied 0.50 chance.
    Fdiss = np.vstack([F["newmark_damped"], F["lumped_CD"], F["consistent_FEM"]])
    yd = np.r_[np.zeros(N_IC), np.ones(2 * N_IC)]; gd = np.r_[ic, ic, ic]
    diss, diss_f = acc(Fdiss, yd, gd), perm_floor(Fdiss, yd, gd, 13)
    print(f"diss  dissipative vs non-dissipative: acc={diss:.3f}   floor={diss_f:.3f}   (Newmark damping detectable; floor is the 60:120 majority baseline)")

    # mass_matrix: lumped vs consistent (both central-difference) - the FINE same-order distinction.
    Fmass = np.vstack([F["lumped_CD"], F["consistent_FEM"]]); ym = np.r_[np.zeros(N_IC), np.ones(N_IC)]; gm = np.r_[ic, ic]
    mass, mass_f = acc(Fmass, ym, gm), perm_floor(Fmass, ym, gm, 17)
    F0 = {s: sigs(s, N_DET, ics, 0.0, 200 + 1000 * j) for j, s in enumerate(("lumped_CD", "consistent_FEM"))}
    mass0 = acc(np.vstack([F0["lumped_CD"], F0["consistent_FEM"]]), ym, gm)
    print(f"mass  lumped vs consistent mass:      acc={mass:.3f}   floor={mass_f:.3f}   (fine limit; weak & noise-mediated)")
    print(f"      lumped vs consistent @ sigma=0:  acc={mass0:.3f}   (≈chance -> NOT a clean deterministic signature; the distinction rides on noise)")

    # NC1 (IC + noise, same scheme) and NC2 (grid change, same scheme)
    F1 = sigs("lumped_CD", N_DET, ics, SIGMA, 5000)
    nc1 = acc(np.vstack([F1[:half], F1[half:]]), np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[:half], ic[half:]])
    ics_a = [random_ic(NC2_GRIDS[0], rng) for _ in range(N_IC)]
    ics_b = [random_ic(NC2_GRIDS[1], rng) for _ in range(N_IC)]
    Fa = sigs("lumped_CD", NC2_GRIDS[0], ics_a, SIGMA, 7000)
    Fb = sigs("lumped_CD", NC2_GRIDS[1], ics_b, SIGMA, 9000)
    nc2 = acc(np.vstack([Fa, Fb]), np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic])
    print(f"\nNC1   IC + noise control (same scheme):   acc={nc1:.3f}   [chance ~0.50]")
    print(f"NC2   grid change {NC2_GRIDS[0]} vs {NC2_GRIDS[1]} (snapshot): acc={nc2:.3f}   [high = grid confound]")

    with open(os.path.join(TAB, "wave_attribution_results.csv"), "w") as f:
        f.write("key,value,floor\n")
        f.write(f"id3,{id3:.4f},{id3f:.4f}\ndissipation,{diss:.4f},{diss_f:.4f}\nmass_matrix,{mass:.4f},{mass_f:.4f}\n"
                f"mass_sigma0,{mass0:.4f},\nnc1,{nc1:.4f},\nnc2,{nc2:.4f},\n")
    print(f"\nartifacts -> {os.path.join(TAB,'wave_attribution_results.csv')}")
    return dict(id3=id3, id3f=id3f, diss=diss, diss_f=diss_f, mass=mass, mass_f=mass_f, mass0=mass0, nc1=nc1, nc2=nc2)

def _figure(r):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try: import seaborn as sns; sns.set_theme(context="paper", style="white", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, PURP = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#8e6fb0"
    SC = {"lumped_CD": (BLUE, "lumped CD"), "consistent_FEM": (GREEN, "consistent FEM"), "newmark_damped": (RED, "Newmark damped")}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6)); fig.subplots_adjust(wspace=0.28, hspace=0.36)

    # A: numerical dispersion (analytic semidiscrete) - lumped lags, consistent leads
    axA = axes[0, 0]; th = np.linspace(0.02, np.pi, 300)
    axA.plot(th / np.pi, np.sqrt(2 - 2 * np.cos(th)) / th, color=BLUE, lw=2, label="lumped mass (CD)")
    axA.plot(th / np.pi, np.sqrt((2 - 2 * np.cos(th)) * 3 / (2 + np.cos(th))) / th, color=GREEN, lw=2, label="consistent FEM mass")
    axA.axhline(1.0, color=GREY, ls=(0, (1, 2)), lw=1.2); axA.text(0.03, 1.005, "exact", fontsize=7.5, color=GREY)
    axA.set_xlabel(r"$k\Delta x/\pi$"); axA.set_ylabel(r"$\omega_h/\omega$"); axA.set_ylim(0.55, 1.30)
    axA.set_title("Numerical dispersion: lumped lags, consistent leads", fontsize=9.5)
    axA.legend(frameon=False, fontsize=8); axA.grid(True, color="#e3e3e3", lw=0.8); axA.set_axisbelow(True)
    axA.text(-0.17, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # representative wave-packet IC (localized -> dispersion vs damping is visible)
    N = 128; x = np.linspace(0, L, N, endpoint=False)
    u0 = np.exp(-(((x - 0.5)) ** 2) / (2 * 0.06 ** 2)) * np.cos(2 * np.pi * 12 * x); u0 /= np.std(u0)
    ex = exact(u0, T); res = {s: run(s, N, u0) - ex for s in SC}

    # B: residual fields r(x)
    axB = axes[0, 1]
    for s, (c, lab) in SC.items():
        rr = res[s]; axB.plot(x, rr / (np.max(np.abs(rr)) + 1e-12), color=c, lw=1.3, label=lab)
    axB.set_xlabel("$x$"); axB.set_ylabel(r"residual $r$ (normalized)"); axB.set_xlim(0, 1); axB.set_ylim(-1.25, 1.25)
    axB.set_title("Residual field: dispersive ripple vs damping", fontsize=9.5)
    axB.legend(frameon=False, fontsize=7.5); axB.grid(True, color="#e3e3e3", lw=0.8); axB.set_axisbelow(True)
    axB.text(-0.17, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: residual spectra |FFT(r)| - where each scheme errs
    axC = axes[1, 0]; k = np.arange(N // 2 + 1)
    for s, (c, lab) in SC.items():
        sp = np.abs(np.fft.rfft(res[s])); axC.semilogy(k, sp / (sp.max() + 1e-12), color=c, lw=1.3, label=lab)
    axC.set_xlabel("wavenumber index $k$"); axC.set_ylabel(r"$|\hat r_k|$ (normalized)"); axC.set_xlim(0, N // 2); axC.set_ylim(1e-4, 1.5)
    axC.set_title("Residual spectrum: dispersive error is high-$k$", fontsize=9.5)
    axC.legend(frameon=False, fontsize=7.5); axC.grid(True, which="major", color="#e3e3e3", lw=0.8); axC.set_axisbelow(True)
    axC.text(-0.17, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    # D: attribution accuracies with per-task permutation floors
    axD = axes[1, 1]
    labels = ["3-way\nID", "dissip.\n(damping)", "mass\nmatrix", "NC1", "NC2\n(grid)"]
    vals = [r["id3"], r["diss"], r["mass"], r["nc1"], r["nc2"]]; cols = [BLUE, GREEN, PURP, GREY, RED]
    floors = [r["id3f"], r["diss_f"], r["mass_f"], 0.5, 0.5]
    axD.bar(range(5), vals, color=cols, width=0.66)
    for i, fl in enumerate(floors):
        axD.plot([i - 0.34, i + 0.34], [fl, fl], color="#333", ls=(0, (2, 1.5)), lw=1.4, zorder=6)
    for i, v in enumerate(vals): axD.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    axD.text(2, r["mass"] - 0.07, f"σ=0:\n{r['mass0']:.2f}", ha="center", va="top", fontsize=6.6, color="#555")
    axD.set_xticks(range(5)); axD.set_xticklabels(labels, fontsize=8); axD.set_ylim(0, 1.05); axD.set_ylabel("accuracy")
    axD.set_title("Attribution (dashed = floor per task)", fontsize=9.5)
    axD.grid(axis="y", color="#e3e3e3", lw=0.8); axD.set_axisbelow(True)
    axD.text(-0.17, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
    out = os.path.join(FIGS, "fig_mechanics.png"); os.makedirs(FIGS, exist_ok=True); fig.savefig(out); plt.close(fig)
    print(f"figure -> {out}")

if __name__ == "__main__":
    import sys
    r = main()
    if "--plot" in sys.argv: _figure(r)
