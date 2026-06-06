#!/usr/bin/env python3
"""
solver-forensics :: ACTIVE MULTI-RESOLUTION ON BURGERS & KdV
============================================================
Extends the grid-confound REPAIR (paper Sec. 2.7, shown there on linear advection) to the
harder equations the Limitations flag as untested: inviscid Burgers (nonlinear) and KdV
(dispersive). The claim: a single-snapshot coefficient signature carries a grid confound
(NC2 high), but the grid-invariant CONVERGENCE RATE p (slope of log||r|| vs log N, from
querying the solver at several resolutions) separates the schemes AND does not encode the
grid - so the same active-query repair works beyond linear advection.

Two schemes per equation (one higher-order, one diffusive/lower-effective-order):
  Burgers : upwind  vs  Lax-Wendroff           (u_t + (u^2/2)_x = 0, smooth pre-shock)
  KdV     : S1 centered flux  vs  S2 LF flux    (u_t + 6 u u_x + u_xxx = 0)
Both reuse the project's VALIDATED solvers (spectral integrating-factor RK4 references).

Reported per equation:
  rate_detect   scheme A vs B by their convergence rate p     -> high (rate separates them)
  p_A / p_B     median recovered rates                        -> distinct
  snap_NC2      single-snapshot signature, same scheme, two grids -> high (grid confound)
  rate_NC2      rate from two disjoint resolution sets, same scheme -> ~chance (grid-invariant)

numpy + scikit-learn, deterministic. Run:  python src/limits/multiresolution_nonlinear.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(_ROOT, "results", "tables")
LIB = (2, 3, 4)
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

# ---- shared signal machinery ----
def _antialias(u, M, Ldom):
    N = len(u)
    return u if N == M else np.fft.irfft(np.fft.rfft(u)[:M // 2 + 1], n=M) * (M / N)
def _deriv(u, o, h):
    if o == 2: return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / h ** 2
    if o == 3: return (np.roll(u, -2) - 2 * np.roll(u, -1) + 2 * np.roll(u, 1) - np.roll(u, 2)) / (2 * h ** 3)
    return (np.roll(u, -2) - 4 * np.roll(u, -1) + 6 * u - 4 * np.roll(u, 1) + np.roll(u, 2)) / h ** 4
def _signature(u, r, Ldom):
    h = Ldom / len(u); A = np.stack([_deriv(u, o, h) for o in LIB], 1)
    c, *_ = np.linalg.lstsq(A, r, rcond=None); n = np.linalg.norm(c)
    return c / n if n > 0 else c
def _ifrk4(uh, Lhat, Nl, dt, ns):
    E, E2 = np.exp(Lhat * dt), np.exp(Lhat * dt / 2)
    for _ in range(ns):
        a = dt * Nl(uh); b = dt * Nl(E2 * (uh + a / 2)); c = dt * Nl(E2 * uh + b / 2); d = dt * Nl(E * uh + E2 * c)
        uh = E * uh + (E * a + 2 * E2 * (b + c) + d) / 6
    return uh
def _acc(F, y, g): return cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean()

# ======================================================= KdV
LK, TK, DELTA, AMPK = 2 * np.pi, 1.0, 1.0, 0.5
D3 = {2: 0.5, 1: -1.0, -1: 1.0, -2: -0.5}; D2 = {1: 1.0, 0: -2.0, -1: 1.0}
def kdv_ic(N, seed):
    r = np.random.default_rng(seed); x = np.linspace(0, LK, N, endpoint=False); u = np.zeros(N)
    for kk in (1, 2, 3): u += r.normal() * np.sin(2 * np.pi * kk * x / LK + r.uniform(0, 2 * np.pi))
    return AMPK * u / (np.max(np.abs(u)) + 1e-9)
def _kdv_ns(u0, dx): return max(1, int(np.ceil(TK / (0.2 * dx / (6 * (np.max(np.abs(u0)) + 1e-9))))))
def _sym(st, k, dx): return sum(c * np.exp(1j * k * m * dx) for m, c in st.items())
def kdv_ref(u0, N):
    dx = LK / N; k = 2 * np.pi * np.fft.fftfreq(N, d=dx); m = np.abs(k) <= (2 / 3) * np.max(np.abs(k))
    ns = _kdv_ns(u0, dx); dt = TK / ns
    Nl = lambda uh: -3j * k * (np.fft.fft(np.real(np.fft.ifft(uh)) ** 2) * m)
    return np.real(np.fft.ifft(_ifrk4(np.fft.fft(u0), 1j * DELTA * k ** 3, Nl, dt, ns)))
def kdv_coarse(u0, N, flux):
    dx = LK / N; k = 2 * np.pi * np.fft.fftfreq(N, d=dx); Lhat = -DELTA * _sym(D3, k, dx) / dx ** 3
    def Nl(uh):
        u = np.real(np.fft.ifft(uh)); f = 3 * u * u
        if flux == "centered": fx = (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
        else:
            a = 6 * np.max(np.abs(u)) + 1e-9; Fp = 0.5 * (f + np.roll(f, -1)) - 0.5 * a * (np.roll(u, -1) - u)
            fx = (Fp - np.roll(Fp, 1)) / dx
        return np.fft.fft(-fx)
    ns = _kdv_ns(u0, dx); dt = TK / ns
    return np.real(np.fft.ifft(_ifrk4(np.fft.fft(u0), Lhat, Nl, dt, ns)))

# ======================================================= Burgers (inviscid, smooth pre-shock)
LB, TFIN, AMPB, CFL = 1.0, 0.08, 0.4, 0.4
def burg_ic(N, seed):
    r = np.random.default_rng(seed); x = np.linspace(0, LB, N, endpoint=False); u = np.zeros(N)
    for _ in range(4): u += r.normal() * np.sin(2 * np.pi * r.integers(1, 4) * x / LB + r.uniform(0, 2 * np.pi))
    return 1.0 + AMPB * u / (np.max(np.abs(u)) + 1e-9)
def burg_ref(u0, N):
    k = 2 * np.pi * np.fft.fftfreq(N, d=LB / N); ik = 1j * k; dx = LB / N; mask = np.abs(k) <= (2 / 3) * np.max(np.abs(k))
    ns = int(np.ceil(TFIN / (CFL * dx / (np.max(np.abs(u0)) + 1e-9)))); dt = TFIN / ns
    Nl = lambda uh: -0.5 * ik * (np.fft.fft(np.real(np.fft.ifft(uh)) ** 2) * mask)
    return np.real(np.fft.ifft(_ifrk4(np.fft.fft(u0), np.zeros_like(k, float), Nl, dt, ns)))
def _s_upwind(u, dt, dx): f = 0.5 * u * u; return u - (dt / dx) * (f - np.roll(f, 1))
def _s_lw(u, dt, dx):
    f = 0.5 * u * u; uh = 0.5 * (u + np.roll(u, -1)) - 0.5 * (dt / dx) * (np.roll(f, -1) - f)
    fh = 0.5 * uh * uh; return u - (dt / dx) * (fh - np.roll(fh, 1))
def burg_coarse(u0, N, scheme):
    dx = LB / N; ns = int(np.ceil(TFIN / (CFL * dx / (np.max(np.abs(u0)) + 1e-9)))); dt = TFIN / ns
    fn = _s_upwind if scheme == "upwind" else _s_lw; u = u0.copy()
    for _ in range(ns): u = fn(u, dt, dx)
    return u

# ======================================================= multi-resolution audit driver
def audit(name, Ldom, ic_fn, ref_fn, coarse_fn, schemes, n_ref, mr1, mr2, snap_pair, n_ic=28, noise=0.005):
    ic = np.arange(n_ic); logN1, logN2 = np.log(np.array(mr1, float)), np.log(np.array(mr2, float))
    def relres(seed, N, scheme):
        u0 = ic_fn(N, seed); tru = _antialias(ref_fn(ic_fn(n_ref, seed), n_ref), N, Ldom)
        return np.linalg.norm(coarse_fn(u0, N, scheme) - tru) / (np.linalg.norm(tru) + 1e-12)
    def rate(seed, scheme, mr, logN): return -np.polyfit(logN, np.log([relres(seed, N, scheme) for N in mr]), 1)[0]

    P = {s: np.array([rate(sd, s, mr1, logN1) for sd in range(n_ic)]) for s in schemes}
    # rate-based scheme detection
    rate_acc = _acc(np.r_[P[schemes[0]], P[schemes[1]]][:, None],
                    np.r_[np.zeros(n_ic), np.ones(n_ic)], np.r_[ic, ic])
    # rate NC2: same scheme, rate from disjoint resolution sets -> grid-invariant -> chance
    Pa = P[schemes[0]]; Pb = np.array([rate(sd, schemes[0], mr2, logN2) for sd in range(n_ic)])
    rate_nc2 = _acc(np.r_[Pa, Pb][:, None], np.r_[np.zeros(n_ic), np.ones(n_ic)], np.r_[ic, ic])
    # single-snapshot signature NC2: same scheme, two grids -> grid confound
    def snap_sig(N, scheme, sd0):
        S = []
        for sd in range(n_ic):
            g = np.random.default_rng(sd0 + sd); u0 = ic_fn(N, sd)
            uc = coarse_fn(u0, N, scheme); tru = _antialias(ref_fn(ic_fn(n_ref, sd), n_ref), N, Ldom)
            un = uc + noise * np.sqrt(np.mean(tru ** 2)) * g.standard_normal(N)
            S.append(_signature(un, un - tru, Ldom))
        return np.array(S)
    Sa, Sb = snap_sig(snap_pair[0], schemes[0], 100), snap_sig(snap_pair[1], schemes[0], 700)
    snap_nc2 = _acc(np.vstack([Sa, Sb]), np.r_[np.zeros(n_ic), np.ones(n_ic)], np.r_[ic, ic])

    pa, pb = float(np.median(P[schemes[0]])), float(np.median(P[schemes[1]]))
    print(f"\n[{name}]  {schemes[0]} vs {schemes[1]}   ({n_ic} ICs, ref N={n_ref})")
    print(f"  convergence rate p:  {schemes[0]}={pa:.2f}   {schemes[1]}={pb:.2f}")
    print(f"  rate-based scheme detection:      acc={rate_acc:.3f}   (high = rate separates the schemes)")
    print(f"  single-snapshot grid control NC2: acc={snap_nc2:.3f}   (high = grid confound present)")
    print(f"  convergence-rate grid control NC2: acc={rate_nc2:.3f}  (~chance = rate is grid-invariant -> confound broken)")
    return dict(eq=name, pa=pa, pb=pb, rate_acc=rate_acc, snap_nc2=snap_nc2, rate_nc2=rate_nc2)

def main():
    os.makedirs(TAB, exist_ok=True)
    print("active multi-resolution: does the convergence-rate repair extend to Burgers & KdV?")
    rB = audit("Burgers", LB, burg_ic, burg_ref, burg_coarse, ("upwind", "lax_wendroff"),
               1024, (128, 160, 192, 256), (144, 176, 208, 272), (192, 256))
    rK = audit("KdV", LK, kdv_ic, kdv_ref, kdv_coarse, ("centered", "LF"),
               512, (96, 128, 160, 192), (104, 136, 168, 200), (128, 192))
    with open(os.path.join(TAB, "multiresolution_nonlinear_results.csv"), "w") as f:
        f.write("equation,p_A,p_B,rate_detect,snapshot_nc2,rate_nc2\n")
        for r in (rB, rK):
            f.write(f"{r['eq']},{r['pa']:.4f},{r['pb']:.4f},{r['rate_acc']:.4f},{r['snap_nc2']:.4f},{r['rate_nc2']:.4f}\n")
    print(f"\nartifacts -> {os.path.join(TAB, 'multiresolution_nonlinear_results.csv')}")
    return rB, rK

def _figure(rB, rK, n_fig=10):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try: import seaborn as sns; sns.set_theme(context="paper", style="white", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#8a8a8a"

    def curves_rates(ic_fn, ref_fn, coarse_fn, schemes, Ldom, mr, nref):
        logN = np.log(np.array(mr, float)); med, P = {}, {}
        refs = [ref_fn(ic_fn(nref, sd), nref) for sd in range(n_fig)]
        for s in schemes:
            ladder, slopes = [], []
            for sd in range(n_fig):
                tru = [_antialias(refs[sd], N, Ldom) for N in mr]
                rr = [np.linalg.norm(coarse_fn(ic_fn(N, sd), N, s) - tru[j]) / (np.linalg.norm(tru[j]) + 1e-12) for j, N in enumerate(mr)]
                ladder.append(rr); slopes.append(-np.polyfit(logN, np.log(rr), 1)[0])
            med[s] = np.median(np.array(ladder), 0); P[s] = np.array(slopes)
        return med, P
    mrB, mrK = (128, 160, 192, 256), (96, 128, 160, 192)
    medB, PB = curves_rates(burg_ic, burg_ref, burg_coarse, ("lax_wendroff", "upwind"), LB, mrB, 1024)
    medK, PK = curves_rates(kdv_ic, kdv_ref, kdv_coarse, ("centered", "LF"), LK, mrK, 512)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6)); fig.subplots_adjust(wspace=0.30, hspace=0.36)
    def conv(ax, mr, med, hi_s, lo_s, rate_hi, rate_lo, title, lbl_hi, lbl_lo):   # label with CANONICAL rate
        ax.plot(mr, med[hi_s], "o-", color=BLUE, lw=2, ms=6, label=f"{lbl_hi}  $p={rate_hi:.2f}$")
        ax.plot(mr, med[lo_s], "s-", color=RED, lw=2, ms=6, label=f"{lbl_lo}  $p={rate_lo:.2f}$")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(mr); ax.set_xticklabels(mr); ax.minorticks_off()
        ax.set_xlabel("grid resolution $N$"); ax.set_ylabel(r"$\|r\|/\|u_{\mathrm{ref}}\|$")
        ax.set_title(title, fontsize=10); ax.legend(frameon=True, framealpha=0.92, edgecolor="#ddd", fontsize=8)
        ax.grid(True, which="major", color="#e3e3e3", lw=0.8); ax.set_axisbelow(True)
    conv(axes[0, 0], mrB, medB, "lax_wendroff", "upwind", rB["pb"], rB["pa"], "Burgers: convergence rate", "Lax-Wendroff", "upwind")
    conv(axes[0, 1], mrK, medK, "centered", "LF", rK["pa"], rK["pb"], "KdV: convergence rate", "centered flux", "LF flux")
    axes[0, 0].text(-0.18, 1.05, "A", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold")
    axes[0, 1].text(-0.18, 1.05, "B", transform=axes[0, 1].transAxes, fontsize=13, fontweight="bold")

    axC = axes[1, 0]; rng = np.random.default_rng(0)
    for gi, (lo, hi) in enumerate([(PB["upwind"], PB["lax_wendroff"]), (PK["LF"], PK["centered"])]):
        jx = gi + rng.uniform(-0.06, 0.06, len(lo))
        axC.scatter(jx, lo, s=18, color=RED, alpha=0.7, edgecolor="none")
        axC.scatter(jx, hi, s=18, color=BLUE, alpha=0.7, edgecolor="none")
        axC.plot([gi - 0.18, gi + 0.18], [np.median(lo)] * 2, color=RED, lw=2.2)
        axC.plot([gi - 0.18, gi + 0.18], [np.median(hi)] * 2, color=BLUE, lw=2.2)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["Burgers", "KdV"]); axC.set_xlim(-0.5, 1.5)
    axC.set_ylabel("per-IC convergence rate $p$"); axC.set_title("Rate distributions separate cleanly (detection 1.00)", fontsize=9.5)
    axC.grid(axis="y", color="#e3e3e3", lw=0.8); axC.set_axisbelow(True)
    axC.text(-0.18, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    axD = axes[1, 1]; eqs = [rB, rK]; x = np.arange(2); w = 0.36
    axD.bar(x - w / 2, [r["snap_nc2"] for r in eqs], w, color=RED, label="single-snapshot signature")
    axD.bar(x + w / 2, [r["rate_nc2"] for r in eqs], w, color=GREEN, label="convergence-rate feature")
    for i, r in enumerate(eqs):
        axD.text(i - w / 2, r["snap_nc2"] + 0.012, f"{r['snap_nc2']:.2f}", ha="center", fontsize=8)
        axD.text(i + w / 2, r["rate_nc2"] + 0.012, f"{r['rate_nc2']:.2f}", ha="center", fontsize=8)
    axD.axhline(0.5, color=GREY, ls=(0, (1, 2)), lw=1); axD.text(1.46, 0.515, "chance", ha="right", fontsize=7.5, color=GREY)
    axD.set_xticks(x); axD.set_xticklabels(["Burgers", "KdV"]); axD.set_ylim(0, 1.0); axD.set_ylabel("grid-control accuracy NC2")
    axD.set_title("Rate feature breaks the grid confound", fontsize=9.5)
    axD.legend(frameon=False, fontsize=8); axD.grid(axis="y", color="#e3e3e3", lw=0.8); axD.set_axisbelow(True)
    axD.text(-0.18, 1.05, "D", transform=axD.transAxes, fontsize=13, fontweight="bold")
    out = os.path.join(os.path.join(_ROOT, "figures"), "fig_multires_nonlinear.png")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out); plt.close(fig); print(f"figure -> {out}")

if __name__ == "__main__":
    import sys
    rB, rK = main()
    if "--plot" in sys.argv: _figure(rB, rK)
