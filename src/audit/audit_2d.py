#!/usr/bin/env python3
"""
2D solver-forensics audit on linear advection-diffusion.

Mirrors the 1D open-solver audit (paper Sec. 2.6-2.7) in two dimensions:
two solver configs differ ONLY in the advection discretization (first-order
upwind vs centered). The modified-equation coefficient direction recovered
from the residual r = u_solver - u_ref attributes the scheme. Controls:
  NC1  initial-condition + noise   (should sit at chance)
  NC2  grid resolution             (single-snapshot grid confound)
Extension: a grid-invariant convergence-rate feature (Sec. 2.7 analogue).

Reference is the EXACT advection-diffusion solution via Fourier (periodic box),
so there is no reference-convergence error to control here.

Deterministic (fixed seeds). Deps: numpy, scipy-free, scikit-learn.
Run:  python audit_2d.py
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------- problem ----------------
AX, AY = 1.0, 0.7          # advection velocity (a_x, a_y)
NU     = 5e-3              # physical diffusion
T      = 0.35             # final time
SEED0  = 0
LIB    = ['u_xx', 'u_yy', 'u_xxx', 'u_yyy', 'u_xxxx', 'u_yyyy']


def _wavenumbers(N):
    k = 2 * np.pi * fftfreq(N, d=1.0 / N)      # domain length 1, dx = 1/N
    KX, KY = np.meshgrid(k, k, indexing='ij')
    return KX, KY


def make_ic(N, seed, n_modes=6):
    """Smooth periodic IC: random low-wavenumber Fourier superposition.
    Mode parameters depend only on `seed`, so make_ic(N, seed) is the SAME
    continuous field sampled at resolution N (needed for convergence rates)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(N) / N
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    u = np.zeros((N, N))
    for _ in range(n_modes):
        kx = int(rng.integers(1, 5)); ky = int(rng.integers(0, 5))
        ph = rng.uniform(0, 2 * np.pi); amp = rng.uniform(0.3, 1.0)
        u += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + ph)
    u -= u.mean()
    s = u.std()
    return u / s if s > 0 else u


def reference(u0, t):
    """Exact advection-diffusion solution via Fourier."""
    KX, KY = _wavenumbers(u0.shape[0])
    u0h = fft2(u0)
    mult = np.exp(-1j * (AX * KX + AY * KY) * t) * np.exp(-NU * (KX ** 2 + KY ** 2) * t)
    return np.real(ifft2(u0h * mult))


# ---- finite-difference operators (periodic, np.roll) ----
def _dx_up(u, h, a):
    return (u - np.roll(u, 1, 0)) / h if a >= 0 else (np.roll(u, -1, 0) - u) / h
def _dy_up(u, h, a):
    return (u - np.roll(u, 1, 1)) / h if a >= 0 else (np.roll(u, -1, 1) - u) / h
def _dx_ce(u, h):
    return (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * h)
def _dy_ce(u, h):
    return (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * h)
def _lap(u, h):
    return ((np.roll(u, -1, 0) - 2 * u + np.roll(u, 1, 0)) +
            (np.roll(u, -1, 1) - 2 * u + np.roll(u, 1, 1))) / h ** 2


def _rhs(u, h, scheme):
    if scheme == 'upwind':
        adv = AX * _dx_up(u, h, AX) + AY * _dy_up(u, h, AY)
    else:  # 'central'
        adv = AX * _dx_ce(u, h) + AY * _dy_ce(u, h)
    return -adv + NU * _lap(u, h)


def solve(u0, scheme, T=T):
    """RK4 in time (accurate, so the residual is dominated by spatial truncation)."""
    N = u0.shape[0]; h = 1.0 / N
    dt = min(0.4 * h / (abs(AX) + abs(AY)), 0.2 * h ** 2 / NU)
    nst = int(np.ceil(T / dt)); dt = T / nst
    u = u0.copy()
    for _ in range(nst):
        k1 = _rhs(u, h, scheme)
        k2 = _rhs(u + 0.5 * dt * k1, h, scheme)
        k3 = _rhs(u + 0.5 * dt * k2, h, scheme)
        k4 = _rhs(u + dt * k3, h, scheme)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def _spectral_derivs(u):
    KX, KY = _wavenumbers(u.shape[0])
    uh = fft2(u)
    D = lambda px, py: np.real(ifft2(uh * (1j * KX) ** px * (1j * KY) ** py))
    return {'u_xx': D(2, 0), 'u_yy': D(0, 2), 'u_xxx': D(3, 0),
            'u_yyy': D(0, 3), 'u_xxxx': D(4, 0), 'u_yyyy': D(0, 4)}


def signature(u_solver, u_ref):
    """Unit-normalized modified-equation coefficient direction from the residual.
    The derivative library is built from the OBSERVED solver field (not the reference),
    matching the 1D strong-form method, so the 2D audit is a faithful analogue of Sec. 2."""
    r = (u_solver - u_ref).ravel()
    D = _spectral_derivs(u_solver)
    A = np.column_stack([D[name].ravel() for name in LIB])
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def residual_norm(u_solver, u_ref):
    return np.linalg.norm((u_solver - u_ref).ravel()) / np.linalg.norm(u_ref.ravel())


# ---------------- datasets + evaluation ----------------
def build(N, schemes, n_ic, noise=0.0, seed_offset=0):
    X, y, g = [], [], []
    for ic in range(n_ic):
        u0 = make_ic(N, SEED0 + seed_offset + ic)
        uref = reference(u0, T)
        for si, sc in enumerate(schemes):
            us = solve(u0, sc)
            if noise > 0:
                rng = np.random.default_rng(10_000 + (seed_offset + ic) * 7 + si)
                us = us + noise * uref.std() * rng.standard_normal(us.shape)
            X.append(signature(us, uref)); y.append(si); g.append(seed_offset + ic)
    return np.asarray(X), np.asarray(y), np.asarray(g)


def cv_acc(X, y, g):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    k = min(5, len(np.unique(g)))
    return cross_val_score(clf, X, y, groups=g, cv=GroupKFold(k), scoring='accuracy').mean()


def perm_floor(X, y, g, n=40, seed=0):
    rng = np.random.default_rng(seed)
    a = [cv_acc(X, rng.permutation(y), g) for _ in range(n)]
    return float(np.mean(a)), float(np.quantile(a, 0.95))


def detection(N, n_ic, noise):
    X, y, g = build(N, ['upwind', 'central'], n_ic, noise=noise)
    return cv_acc(X, y, g), perm_floor(X, y, g)


def nc1(N, n_ic, noise):
    # same scheme, two independent IC/noise draws labelled 0/1 -> should be chance
    Xa, _, ga = build(N, ['upwind'], n_ic, noise=noise, seed_offset=0)
    Xb, _, gb = build(N, ['upwind'], n_ic, noise=noise, seed_offset=10_000)
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int)
    g = np.r_[ga, gb]
    return cv_acc(X, y, g)


def nc2(N1, N2, n_ic, noise):
    # same scheme + same ICs, two grids labelled 0/1 -> high => grid confound
    Xa, _, ga = build(N1, ['upwind'], n_ic, noise=noise, seed_offset=0)
    Xb, _, gb = build(N2, ['upwind'], n_ic, noise=noise, seed_offset=0)
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))].astype(int)
    g = np.r_[ga, gb]                      # same IC groups -> grouped CV honest
    return cv_acc(X, y, g)


def convergence_audit(Ns, n_ic, noise=0.0):
    """Convergence-rate feature: slope of log||r|| vs log N, per (IC, scheme)."""
    logN = np.log(np.asarray(Ns, float))
    X, y, g = [], [], []
    mean_rate = {'upwind': [], 'central': []}
    for ic in range(n_ic):
        for si, sc in enumerate(['upwind', 'central']):
            rs = []
            for N in Ns:
                u0 = make_ic(N, SEED0 + ic)
                us = solve(u0, sc)
                rn = residual_norm(us, reference(u0, T))
                if noise > 0:
                    rng = np.random.default_rng(20_000 + ic * 7 + si + N)
                    rn = rn * (1 + 0.02 * rng.standard_normal())
                rs.append(rn)
            slope = np.polyfit(logN, np.log(rs), 1)[0]      # ||r|| ~ N^slope, slope<0
            X.append([-slope]); y.append(si); g.append(ic)
            mean_rate[sc].append(-slope)
    X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
    acc = cv_acc(X, y, g)
    floor = perm_floor(X, y, g)
    rates = {k: (float(np.mean(v)), float(np.std(v))) for k, v in mean_rate.items()}
    return acc, floor, rates


# ---------------- optional figure (house style); pass --plot ----------------
def _emit_figure(det_acc, det_floor, nc1_v, nc2_v, rate_acc, rates, Ns, n_ic_fig=20):
    import os
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context='paper', style='white', font='DejaVu Sans')
    except Exception:
        pass
    plt.rcParams.update({'mathtext.fontset': 'cm', 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    BLUE, RED, GREEN, GREY = '#4C72B0', '#C44E52', '#55A868', '#8a8a8a'
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'figures')
    os.makedirs(out, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.9)); fig.subplots_adjust(wspace=0.34)
    # panel A: detection + controls
    labels = ['det σ0', 'det σ1%', 'det σ5%', 'NC1', 'NC2']
    vals = [det_acc[0.0], det_acc[0.01], det_acc[0.05], nc1_v, nc2_v]
    cols = [GREEN, GREEN, GREEN, GREY, RED]
    axA.bar(range(5), vals, color=cols, width=0.66)
    fl = float(np.mean(list(det_floor.values())))
    axA.axhline(fl, color=GREY, ls=(0, (2, 2)), lw=1)
    axA.text(4.45, fl + 0.02, 'perm. floor', ha='right', fontsize=7.5, color=GREY)
    for i, v in enumerate(vals): axA.text(i, v + 0.015, f'{v:.2f}', ha='center', fontsize=8)
    axA.set_xticks(range(5)); axA.set_xticklabels(labels, fontsize=8.5); axA.set_ylim(0, 1.08)
    axA.set_ylabel('accuracy'); axA.set_title('Detection and controls', fontsize=10)
    axA.grid(axis='y', color='#e3e3e3', lw=0.8); axA.set_axisbelow(True)
    axA.text(-0.05, 1.06, 'A', transform=axA.transAxes, fontsize=13, fontweight='bold')
    # panel B: convergence-rate log-log; the legend shows the CANONICAL per-IC mean rate
    # (rates[...] from convergence_audit), so the figure and the §2.10 text report the SAME p.
    for scheme, c in (('upwind', BLUE), ('central', RED)):
        med = []
        for N in Ns:
            rn = [residual_norm(solve(make_ic(N, SEED0 + ic), scheme), reference(make_ic(N, SEED0 + ic), T))
                  for ic in range(n_ic_fig)]
            med.append(float(np.median(rn)))
        pm, ps = rates[scheme]
        axB.plot(Ns, med, 'o-', color=c, lw=2, ms=6, label=fr'{scheme}   $p={pm:.2f}\pm{ps:.2f}$')
    axB.set_xscale('log'); axB.set_yscale('log'); axB.set_xticks(Ns); axB.set_xticklabels(Ns); axB.minorticks_off()
    axB.set_xlabel('grid resolution $N$'); axB.set_ylabel(r'relative residual $\|r\|/\|u_{\mathrm{exact}}\|$')
    axB.set_title('Grid-invariant convergence rate', fontsize=10)
    axB.legend(frameon=True, framealpha=0.92, edgecolor='#ddd'); axB.grid(True, which='major', color='#e3e3e3', lw=0.8); axB.set_axisbelow(True)
    axB.text(-0.05, 1.06, 'B', transform=axB.transAxes, fontsize=13, fontweight='bold')
    fig.savefig(os.path.join(out, 'fig_audit2d.png'))
    plt.close(fig)


# ---------------- run ----------------
if __name__ == '__main__':
    import sys
    np.seterr(all='ignore')
    N = 64
    N_IC = 40
    print(f"2D advection-diffusion audit  (a=({AX},{AY}), nu={NU}, T={T}, grid {N}x{N}, {N_IC} ICs)\n")

    print("DETECTION  (upwind vs centered advection, same everything else)")
    det_acc, det_floor = {}, {}
    for noise in (0.0, 0.01, 0.05):
        acc, (fm, f95) = detection(N, N_IC, noise)
        det_acc[noise], det_floor[noise] = acc, fm
        print(f"  sigma={noise:<5}  accuracy={acc:.3f}   perm-floor mean={fm:.3f}  95%={f95:.3f}")

    print("\nCONTROLS  (scheme held fixed)")
    nc1_v = nc1(N, N_IC, 0.01)
    print(f"  NC1 initial-condition + noise (sigma=1%):  accuracy={nc1_v:.3f}   (chance ~0.50; low = good)")
    nc2_v = nc2(64, 96, N_IC, 0.0)
    print(f"  NC2 grid change 64 vs 96 (snapshot):       accuracy={nc2_v:.3f}   (high = single-snapshot grid confound)")

    print("\nMULTI-RESOLUTION  (grid-invariant convergence-rate feature)")
    Ns = [32, 48, 64, 96]
    acc, (fm, f95), rates = convergence_audit(Ns, n_ic=30, noise=0.0)
    print(f"  resolutions {Ns}")
    print(f"  recovered rate p (||r||~N^-p):  upwind {rates['upwind'][0]:.2f}+/-{rates['upwind'][1]:.2f}   "
          f"central {rates['central'][0]:.2f}+/-{rates['central'][1]:.2f}")
    print(f"  detection on rate:  accuracy={acc:.3f}   perm-floor mean={fm:.3f}  95%={f95:.3f}")
    print("\ndone.")

    if '--plot' in sys.argv:
        _emit_figure(det_acc, det_floor, nc1_v, nc2_v, acc, rates, Ns)
        print("figure -> figures/fig_audit2d.png")
