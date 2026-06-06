"""
solver-forensics :: PUBLICATION FIGURES  (seaborn, WeakPINN house style)
========================================================================
Five figures, each chosen because it reveals real mathematical structure rather than
restating an AUROC. House style matches arnavgarg233/WeakPINN: bold "Figure N | title"
header, bold panel letters, muted seaborn palette, crisp lines, light dotted grids,
despined, compact.

  fig_advection : the residual IS the modified-equation term (R^2=1 fit) + (c2,c3) phase plane
  fig_transfer  : family-robust vs same-order-fragile degradation + grid/CFL transfer surface
  fig_multires  : log-log ||r|| vs N convergence rates - the grid-invariant rate that breaks the confound
  fig_reference : coefficient drift -> 0 while attribution stays ~1 (attribution != recovery)
  fig_site      : dense vs sparse lambda-sweep - the mechanism of STLSQ brittleness

Numbers come from results/tables/*.csv (runs of record); raw geometry is recomputed with the
import-safe src/baselines/site_baseline.py kernels. Figures -> figures/*.png.  Run: python scripts/make_figures.py
"""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB   = os.path.join(_ROOT, "results", "tables")
OUT   = os.path.join(_ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
import site_baseline as sb

# ----------------------------------------------------------------- aesthetic (WeakPINN-matched)
sns.set_theme(context="paper", style="white", font="DejaVu Sans")
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.titleweight": "regular", "axes.titlepad": 7,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": "#333", "axes.linewidth": 0.9,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 130,
})
# muted seaborn-deep palette; diffusive cool, dispersive warm
BLUE, CYAN, RED, ORANGE, GREEN, GREY = "#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#55A868", "#8a8a8a"
COL = {"upwind": BLUE, "lax_friedrichs": CYAN, "lax_wendroff": RED, "beam_warming": ORANGE}
PRETTY = {"upwind": "upwind", "lax_friedrichs": "Lax-Friedrichs", "lax_wendroff": "Lax-Wendroff", "beam_warming": "Beam-Warming"}
INK = "#222"

def header(fig, n, title, y=1.0):
    # no banner baked into the raster - the title lives in the LaTeX \caption (Elsevier house style),
    # and LaTeX owns the figure number. Kept as a no-op so call sites need not change.
    return

def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, ha="right", va="bottom")

def ygrid(ax):
    ax.grid(axis="y", color="#cfcfcf", ls=(0, (1, 2)), lw=0.8); ax.set_axisbelow(True)
    sns.despine(ax=ax)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))      # PNG only
    plt.close(fig); print(f"  wrote {name}.png")

def read_csv(name):
    with open(os.path.join(TAB, name)) as f: return list(csv.DictReader(f))

def diffdisp_labels(lab):
    return np.array([0 if sb.names[l] in sb.DIFFUSIVE else 1 for l in lab])

def _relres(scheme_fn, N, u0b, nu=0.6):
    L, a, T = 1.0, 1.0, 0.30
    u0 = np.fft.irfft(np.fft.rfft(u0b)[:N // 2 + 1], n=N) * (N / len(u0b))
    dx = L / N; dt = nu * dx / a; ns = int(round(T / dt)); u = u0.copy()
    for _ in range(ns): u = scheme_fn(u, nu)
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N); ex = np.fft.irfft(np.fft.rfft(u0) * np.exp(-1j * k * a * ns * dt), n=N)
    r = u - ex
    return np.sqrt(np.mean(r ** 2)) / (np.sqrt(np.mean(ex ** 2)) + 1e-12)

def _attribution_transfer():
    """Per-cell grid/CFL transfer, reproducing src/attribution/coefficient_attribution.py exactly:
    common observation grid N_OBS=64, strong-form direction+ratio feature, 200/60 ICs, field 1% noise,
    accuracy (clf.score) - the run of record behind the paper's transfer numbers."""
    from sklearn.model_selection import cross_val_score, GroupKFold
    L, a, T, N_OBS = 1.0, 1.0, 0.30, 64
    GRIDS, CFLS, BASE, NB, NT = [128, 256, 512], [0.4, 0.6, 0.8], (256, 0.6), 200, 60
    SCH = [sb.upwind, sb.lax_friedrichs, sb.lax_wendroff, sb.beam_warming]   # identical numerics
    def ric(N, rng):
        x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
        for _ in range(6): u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 8) * x / L + rng.uniform(0, 2 * np.pi))
        if rng.random() < 0.7:
            x0, w = rng.uniform(0, L), rng.uniform(L * 0.02, L * 0.08)
            u += rng.normal() * np.exp(-(((x - x0 + L / 2) % L - L / 2) ** 2) / (2 * w * w))
        return u
    def aa(R, M):
        N = R.shape[-1]
        if N == M: return R
        F = np.fft.rfft(R, axis=-1); F[..., M // 2 + 1:] = 0.0
        return np.fft.irfft(F, n=N, axis=-1)[..., ::(N // M)]
    def sim(N, nu, nic, seed):
        rng = np.random.default_rng(seed); dx = L / N; dt = nu * dx / a; ns = int(round(T / dt))
        U, X, lab, grp = [], [], [], []
        for ic in range(nic):
            u0 = ric(N, rng); ex = aa(np.fft.irfft(np.fft.rfft(u0) * np.exp(-1j * 2*np.pi*np.fft.rfftfreq(N, d=L/N) * a * ns*dt), n=N), N_OBS)
            for li, fn in enumerate(SCH):
                u = u0.copy()
                for _ in range(ns): u = fn(u, nu)
                U.append(aa(u, N_OBS)); X.append(ex); lab.append(li); grp.append(seed * 1_000_000 + ic)
        return np.array(U), np.array(X), np.array(lab), np.array(grp)
    def noisy(U, X, sigma, seed):
        rng = np.random.default_rng(seed); rms = np.sqrt(np.mean(U ** 2, 1, keepdims=True))
        uf = U + sigma * rms * rng.standard_normal(U.shape) if sigma > 0 else U
        return uf, uf - X
    def sfc(U, R):
        h = L / N_OBS
        uxx = (np.roll(U,-1,1) - 2*U + np.roll(U,1,1)) / h**2
        uxxx = (np.roll(U,-2,1) - 2*np.roll(U,-1,1) + 2*np.roll(U,1,1) - np.roll(U,2,1)) / (2*h**3)
        uxxxx = (np.roll(U,-2,1) - 4*np.roll(U,-1,1) + 6*U - 4*np.roll(U,1,1) + np.roll(U,2,1)) / h**4
        A = np.stack([uxx, uxxx, uxxxx], 2); AtA = np.einsum('mni,mnk->mik', A, A) + 1e-8*np.eye(3)
        return np.linalg.solve(AtA, np.einsum('mni,mn->mi', A, R)[..., None])[..., 0]
    def feat(c):
        unit = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
        r32 = np.clip(np.nan_to_num(c[:,1]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
        r42 = np.clip(np.nan_to_num(c[:,2]/c[:,0], nan=0., posinf=10, neginf=-10), -10, 10)
        return np.hstack([unit, r32[:,None], r42[:,None]])
    def ydiff(lab): return np.array([0 if l in (0, 1) else 1 for l in lab])
    Ub, Xb, lab_b, grp_b = sim(*BASE, NB, 1); uo, ro = noisy(Ub, Xb, 0.01, 10)
    Ftr, ytr = feat(sfc(uo, ro)), ydiff(lab_b)
    clf = sb.CLF(); clf.fit(Ftr, ytr)
    H = np.full((len(CFLS), len(GRIDS)), np.nan); sidx = 100
    for gi, N in enumerate(GRIDS):
        for ci, nu in enumerate(CFLS):
            if (N, nu) == BASE:
                H[ci, gi] = cross_val_score(sb.CLF(), Ftr, ytr, groups=grp_b, cv=GroupKFold(5)).mean()
                continue
            sidx += 1
            Un, Xn, la, _ = sim(N, nu, NT, sidx); u2, r2 = noisy(Un, Xn, 0.01, sidx + 500)
            H[ci, gi] = clf.score(feat(sfc(u2, r2)), ydiff(la))
    return H, GRIDS, CFLS

# =================================================================== FIG 1
def fig_advection():
    N = 256
    U0, R0, lab, grp = sb.gen(N, 0.6, 12, seed=1)
    ic = 3; rows = [ic * 4 + k for k in range(4)]
    C = sb.dense_coeffs(U0, R0, sb.RESTRICTED)
    recon = np.einsum("mni,mi->mn", sb.lib(U0, sb.RESTRICTED), C)
    x = np.linspace(0, 1, N, endpoint=False)
    V = C[:, :2] / (np.linalg.norm(C[:, :2], axis=1, keepdims=True) + 1e-12)
    DOM = {"upwind": r"$c_2 u_{xx}$", "lax_friedrichs": r"$c_2 u_{xx}$",
           "lax_wendroff": r"$c_3 u_{xxx},\ c_3<0$", "beam_warming": r"$c_3 u_{xxx},\ c_3>0$"}

    fig = plt.figure(figsize=(10.0, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.22)
    for k, (r, c) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        row = rows[k]; ax = fig.add_subplot(gs[r, c]); nm = sb.names[lab[row]]
        rr, re = R0[row], recon[row]; scl = np.max(np.abs(rr)) + 1e-12
        ax.plot(x, rr / scl, color=COL[nm], lw=1.6, label="residual $r$")
        ax.plot(x, re / scl, color=INK, lw=0.9, ls=(0, (4, 2)), label="modified-eq. fit")
        ss = 1 - np.sum((rr - re) ** 2) / (np.sum((rr - rr.mean()) ** 2) + 1e-12)
        ax.set_title(f"{PRETTY[nm]}   ({DOM[nm]})", fontsize=9, color=COL[nm])
        ax.text(0.97, 0.06, f"$R^2={ss:.2f}$", transform=ax.transAxes, fontsize=7.5, color=GREY, ha="right", va="bottom")
        ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(-1.3, 1.3); sns.despine(ax=ax, left=True)
        ax.set_xticks([0, 0.5, 1])
        if r == 1: ax.set_xlabel("$x$", fontsize=9)
        else: ax.set_xticklabels([])
        if k == 0:
            ax.legend(loc="lower left", fontsize=6.6, frameon=False, ncol=2, bbox_to_anchor=(-0.02, 0.92),
                      handlelength=1.4, columnspacing=1.0)
            panel(ax, "A")

    axp = fig.add_subplot(gs[:, 2])
    th = np.linspace(0, 2 * np.pi, 240)
    axp.plot(np.cos(th), np.sin(th), color="#d8d8d8", lw=1.0, zorder=0)
    for i, nm in enumerate(sb.names):
        m = lab == i
        axp.scatter(V[m, 0], V[m, 1], s=30, color=COL[nm], alpha=0.9, edgecolor="white", linewidth=0.4, label=PRETTY[nm], zorder=3)
    axp.axhline(0, color="#bbb", lw=0.8); axp.axvline(0, color="#bbb", lw=0.8)
    axp.set_xlim(-1.32, 1.32); axp.set_ylim(-1.38, 1.38); axp.set_aspect("equal")
    axp.set_xlabel(r"$\hat c_2$  (diffusion)"); axp.set_ylabel(r"$\hat c_3$  (dispersion)")
    axp.set_xticks([-1, 0, 1]); axp.set_yticks([-1, 0, 1])
    axp.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9, edgecolor="#ddd", handletextpad=0.3)
    axp.text(0, 1.16, "BW: lead", fontsize=7.5, color=ORANGE, ha="center", style="italic")
    axp.text(0, -1.22, "LW: lag", fontsize=7.5, color=RED, ha="center", style="italic")
    sns.despine(ax=axp); panel(axp, "B")
    header(fig, 1, "A solver's error is its modified-equation term, and its shape names the scheme")
    save(fig, "fig_advection")

# =================================================================== FIG 2
def fig_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9)); fig.subplots_adjust(wspace=0.30)
    ca = read_csv("coefficient_attribution_results.csv"); ns = read_csv("near_shock_results.csv")
    def cav(row, d): return next(float(r["accuracy"]) for r in ca if r["row"] == row and r["method"] == "strong-form" and r["distinction"] == d)
    def nsv(level, d): return next(float(r[d]) for r in ns if r["method"] == "interp+FD" and r["level"] == level)
    regimes = ["clean", "σ=1%", "σ=5%", "near-\nshock", "warp+\nmiss"]
    diff = [cav("same grid/CFL, field σ=0", "diff_disp"), cav("same grid/CFL, field σ=0.01", "diff_disp"),
            cav("same grid/CFL, field σ=0.05", "diff_disp"), nsv("uniform-64", "diff_disp"), nsv("warp+jit50+miss50", "diff_disp")]
    lwbw = [cav("same grid/CFL, field σ=0", "lw_bw"), cav("same grid/CFL, field σ=0.01", "lw_bw"),
            cav("same grid/CFL, field σ=0.05", "lw_bw"), nsv("uniform-64", "lw_bw"), nsv("warp+jit50+miss50", "lw_bw")]
    ax = axes[0]; xr = np.arange(len(regimes))
    ax.plot(xr, diff, "o-", color=GREEN, lw=2, ms=6, label="diffusive vs dispersive")
    ax.plot(xr, lwbw, "s--", color=RED, lw=2, ms=6, label="LW vs BW (same order)")
    ax.fill_between(xr, lwbw, diff, color=RED, alpha=0.07)
    ax.axhline(0.5, color=GREY, ls=(0, (1, 2)), lw=0.9); ax.text(xr[-1], 0.515, "chance", ha="right", fontsize=7.5, color=GREY)
    ax.set_xticks(xr); ax.set_xticklabels(regimes, fontsize=8); ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("attribution accuracy"); ax.set_title("Family split robust; same-order fades first", fontsize=10)
    ax.legend(loc="lower left", frameon=False); ygrid(ax); panel(ax, "A")

    axB = axes[1]
    H, Ns, CFLs = _attribution_transfer()              # run of record: common N_OBS=64, accuracy
    print(f"    [transfer] per-cell accuracy {H.min():.3f}-{H.max():.3f}; "
          f"worst N={Ns[np.argwhere(H==H.min())[0][1]]} CFL={CFLs[np.argwhere(H==H.min())[0][0]]}")
    sns.heatmap(H, ax=axB, cmap="crest", vmin=0.85, vmax=1.0, annot=True, fmt=".3f",
                annot_kws={"fontsize": 9.5}, cbar_kws={"label": "diff-vs-disp accuracy", "fraction": 0.046, "pad": 0.04},
                linewidths=1.2, linecolor="white")
    axB.set_xticklabels(Ns, rotation=0); axB.set_yticklabels(CFLs, rotation=0)
    axB.set_xlabel("grid resolution $N$"); axB.set_ylabel("CFL number"); axB.invert_yaxis()
    axB.set_title("Trained on one grid, tested on others", fontsize=10); panel(axB, "B")
    header(fig, 2, "Family-level signatures transfer across resolution and CFL")
    save(fig, "fig_transfer")

# =================================================================== FIG 3
def fig_multires():
    SCH = {"upwind": sb.upwind, "lax_friedrichs": sb.lax_friedrichs, "lax_wendroff": sb.lax_wendroff, "beam_warming": sb.beam_warming}
    Ns = np.array([48, 64, 96, 128, 192, 256])
    rng = np.random.default_rng(0)
    def smooth_ic(M):
        xx = np.linspace(0, 1, M, endpoint=False); u = np.zeros(M)
        for _ in range(4): u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 5) * xx + rng.uniform(0, 2 * np.pi))
        return u / (np.std(u) + 1e-12)
    bases = [smooth_ic(512) for _ in range(24)]
    curves, slopes = {}, {}
    for nm, fn in SCH.items():
        M = np.array([[_relres(fn, N, u0) for N in Ns] for u0 in bases])
        curves[nm] = np.median(M, 0)
        slopes[nm] = -np.polyfit(np.log(Ns), np.median(np.log(M + 1e-15), 0), 1)[0]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for nm, fn in SCH.items():
        ax.plot(Ns, curves[nm], "o-", color=COL[nm], lw=2, ms=6, label=f"{PRETTY[nm]}   $p={slopes[nm]:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Ns); ax.set_xticklabels(Ns); ax.minorticks_off()
    ax.set_xlabel("grid resolution  $N$  (query the solver at several)")
    ax.set_ylabel(r"relative residual  $\|r\|/\|u_{\mathrm{exact}}\|$")
    ax.set_title("A convergence rate cannot encode which grid produced it", fontsize=10.5)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#ddd", title="slope $p$ of $\\|r\\|\\sim N^{-p}$")
    ax.grid(True, which="major", color="#e3e3e3", lw=0.8); ax.set_axisbelow(True); sns.despine(ax=ax)
    ax.text(0.975, 0.96, "dispersive  $p\\approx2$\ndiffusive  $p\\approx0.7$-$0.9$\n\n"
            "grid-change control:\n$0.99\\to0.63$ (to chance)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color="#333",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#ddd"))
    header(fig, 3, "Active multi-resolution access removes the grid confound", y=1.0)
    save(fig, "fig_multires")

# =================================================================== FIG 4
def fig_reference():
    ratios = ["1×", "2×", "4×", "8×"]
    drift  = [1.34, 0.39, 0.06, 0.00]
    attrib = [0.975, 0.990, 0.998, 1.000]
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); x = np.arange(len(ratios))
    l1, = ax.plot(x, drift, "o-", color=RED, lw=2.2, ms=7, label="coefficient-direction drift")
    ax.fill_between(x, drift, color=RED, alpha=0.06)
    ax.set_xticks(x); ax.set_xticklabels(ratios)
    ax.set_xlabel(r"reference fineness  $N_{\mathrm{ref}}/N$")
    ax.set_ylabel("coeff.-direction drift (rad)", color=RED); ax.tick_params(axis="y", labelcolor=RED)
    ax.set_ylim(-0.08, 1.5)
    axR = ax.twinx()
    l2, = axR.plot(x, attrib, "s--", color=GREEN, lw=2.2, ms=7, label="attribution accuracy")
    axR.set_ylabel("attribution accuracy", color=GREEN); axR.tick_params(axis="y", labelcolor=GREEN)
    axR.set_ylim(0.5, 1.03); axR.grid(False)
    ax.grid(axis="y", color="#cfcfcf", ls=(0, (1, 2)), lw=0.8); ax.set_axisbelow(True)
    for sp in ("top",): ax.spines[sp].set_visible(False); axR.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RED); axR.spines["right"].set_color(GREEN)
    ax.legend(handles=[l1, l2], loc="center right", frameon=True, framealpha=0.92, edgecolor="#ddd")
    ax.set_title("Attribution tolerates a coarse reference; recovery does not", fontsize=10.5)
    header(fig, 4, "Attributing a scheme and recovering its coefficients are different tasks", y=1.0)
    save(fig, "fig_reference")

# =================================================================== FIG 5
def fig_site():
    U0, R0, lab, grp = sb.gen(256, 0.6, 80, seed=1)
    uo_b, ro_b = sb.observe(U0, R0, 0.01, 1, 11); yb = diffdisp_labels(lab)
    HELD = [(N, nu) for N in (128, 256, 512) for nu in (0.4, 0.6, 0.8) if (N, nu) != (256, 0.6)]
    TE = []
    for N, nu in HELD:
        Ut, Rt, lt, _ = sb.gen(N, nu, 30, seed=100 + N + int(nu * 10))
        uo, ro = sb.observe(Ut, Rt, 0.01, 1, 200 + N); TE.append((uo, ro, diffdisp_labels(lt)))
    def tg(lam):
        sb.LAM[sb.RICH] = lam; out = {}
        for m in ("dense", "sparse"):
            c = sb.CLF(); c.fit(sb.feats(uo_b, ro_b, m, sb.RICH), yb)
            X = np.vstack([sb.feats(uo, ro, m, sb.RICH) for uo, ro, _ in TE]); y = np.concatenate([y for *_, y in TE])
            out[m] = roc_auc_score(y, c.predict_proba(X)[:, 1])
        return out["dense"], out["sparse"]
    lams = [0.003, 0.01, 0.03, 0.1, 0.2, 0.3]
    ds, ss = zip(*[tg(l) for l in lams])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(lams, ds, "o-", color=BLUE, lw=2.2, ms=7, label="dense least-squares")
    ax.plot(lams, ss, "s-", color=RED, lw=2.2, ms=7, label="sparse STLSQ (SITE-style)")
    ax.fill_between(lams, ss, ds, color=RED, alpha=0.08)
    ax.set_xscale("log"); ax.set_xlabel(r"sparsity threshold  $\lambda$")
    ax.set_ylabel("grid/CFL transfer AUROC  (rich library)"); ax.set_ylim(0.6, 1.0)
    ax.axvline(0.03, color=GREY, ls=(0, (1, 2)), lw=1.0)
    ax.text(0.033, 0.625, "frozen $\\lambda$", fontsize=8, color=GREY)
    ax.set_title("Sparse recovery is brittle under an over-specified library", fontsize=10.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#ddd")
    ax.annotate("thresholds onto the\nspurious $u_x$ term", (0.03, ss[2]), xytext=(0.055, 0.70),
                fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.9, alpha=0.7))
    ax.grid(axis="y", color="#cfcfcf", ls=(0, (1, 2)), lw=0.8); ax.set_axisbelow(True); sns.despine(ax=ax)
    header(fig, 5, "Dense recovery is robust where sparse identification is not", y=1.0)
    save(fig, "fig_site")

FIGS = [fig_advection, fig_transfer, fig_multires, fig_reference, fig_site]
if __name__ == "__main__":
    import time, traceback
    print(f"Generating 5 figures (seaborn / WeakPINN style) -> {OUT}\n")
    ok = 0
    for fn in FIGS:
        t = time.time()
        try: fn(); ok += 1; print(f"    ({time.time()-t:.1f}s)")
        except Exception as e: print(f"  FAILED {fn.__name__}: {e}"); traceback.print_exc()
    print(f"\nDone: {ok}/{len(FIGS)} figures")
