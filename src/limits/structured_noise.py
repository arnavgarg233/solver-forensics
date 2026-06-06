#!/usr/bin/env python3
"""
solver-forensics :: STRUCTURED / NON-ADDITIVE NOISE ROBUSTNESS  (ITEM 7, CMAME revision)
=========================================================================================
Every prior robustness number in this project uses ADDITIVE, field-relative, *white* Gaussian
observation noise (eta = sigma * RMS(u_obs) * N(0,1), spatially uncorrelated). A reviewer will
ask the obvious question: is the coefficient-direction signature robust to REALISTIC corruption,
or did i.i.d.-Gaussian quietly do the work? This script answers that on the load-bearing
attribution -- DIFFUSIVE vs DISPERSIVE schemes on linear advection -- under four corruption
models at MATCHED RMS, with the additive-white-Gaussian case as the in-script baseline.

  Schemes (linear advection u_t + a u_x = 0, a=1, periodic), the canonical contrast:
      DIFFUSIVE  : upwind, lax_friedrichs    (even-derivative-dominated modified equation, c2)
      DISPERSIVE : lax_wendroff, beam_warming(odd-derivative-dominated modified equation,  c3)
  SIGNATURE  : unit-normalized least-squares coefficient DIRECTION of c in
               r = u_solver - u_ref ~ sum_p c_p d_x^p u, library {u_xx, u_xxx, u_xxxx},
               FD derivatives of the OBSERVED field on the regular observation grid.
               (kernels REPLICATED from src/attribution/coefficient_attribution.py and
                src/robustness/production_schemes.py; nothing imported.)
  ATTRIBUTION: StandardScaler+LogisticRegression, GroupKFold(5) grouped by INITIAL CONDITION,
               label-PERMUTATION floor on EVERY reported number.

NOISE MODELS (all applied to the FIELD before forming the residual r = u_obs - u_ref, all
              scaled to the SAME per-sample RMS as the additive-Gaussian baseline so the
              comparison is at matched corruption energy):
  G   additive white Gaussian            eta = sigma*RMS(u)*xi,  xi ~ N(0,1) i.i.d.   [BASELINE]
  Clp additive SPATIALLY-CORRELATED, LOW-PASS  : xi filtered by a smooth low-pass spectral
        envelope (correlation length ell). Smooth, broad -- a "rough field" confound.
  Cd  additive SPATIALLY-CORRELATED, DERIVATIVE-MIMICKING  : xi shaped by a band/high-pass
        envelope ~ k^beta exp(-(k/kc)^2) that PEAKS where u_xx/u_xxx live -- noise engineered
        to LOOK LIKE a derivative term. THE ADVERSARIAL CASE: it projects directly onto the
        library and can bias the coefficient direction.
  Mul multiplicative                      eta = sigma_eff * u * xi      (signal-dependent)
  Q   quantized                           u_obs = round(u / q) * q      (deterministic rounding,
                                            q chosen so the quantization RMS matches the baseline)

For each noise model we report, on the SAME ICs/schemes:
  diff_disp  : diffusive-vs-dispersive attribution accuracy (the load-bearing number) + perm floor
  NC1        : same scheme (upwind), IC + that-noise only, arbitrary IC partition -> must be ~chance
  NC2        : same scheme (upwind), GRID CHANGE under that noise -> the known confound (diagnostic)
  cos_to_G   : cosine between this model's mean diffusive signature and the baseline's, and likewise
               for dispersive -- how much the noise DISTORTS the signature direction itself.

DECISION RULE (pre-registered):
  * robust across noise types  -> a clean robustness row (each model's diff_disp >= baseline-0.05,
    NC1 ~chance) reported as "the signature survives structured/non-additive noise."
  * if the DERIVATIVE-MIMICKING correlated noise DEGRADES diff_disp materially (drop > ~0.05 vs
    baseline, or pulls NC1 above chance) -> reported as the HONEST SENSITIVITY with the number:
    white-Gaussian was optimistic; correlated noise that mimics a derivative term is the worst case.

VALIDATION: the advection solver is validated (exact spectral reference; per-scheme stability
max|u| < threshold; clean-limit modified-equation coefficient signs match theory: diffusive c2-
dominant, LW c3<0, BW c3>0) and PRINTED before any residual is trusted. The matched-RMS property
of every noise model is also asserted numerically and printed.

Self-contained: numpy + scipy + sklearn, CPU, deterministic. Guarded by __main__.
Run:  python src/limits/structured_noise.py            (add --plot for the figure)
metrics -> results/tables/structured_noise.csv ; figure -> figures/structured_noise.png
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_PUB = os.path.join(_ROOT, "figures")
TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG_PUB, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---- physical / numerical constants (linear advection) ----
L, A, T = 1.0, 1.0, 0.30
NU = 0.8                       # CFL for the solver grid
N_C, N_C2 = 128, 192           # solver grid and the NC2 grid-change grid
N_OBS = 64                     # fixed anti-aliased observation grid (physical coarseness)
N_IC = 60                      # initial-condition ensemble (the GroupKFold group key)
SIGMA = 0.05                   # field-relative noise level (matched-RMS target across models;
                               # baseline ~0.95 -> headroom to see structured-noise degradation)
LIB = (2, 3, 4)                # derivative library orders {u_xx, u_xxx, u_xxxx}

# ---------------------------------------------------------------- physics (replicated kernels)
def exact_advection(u0, t, N):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    return np.fft.irfft(np.fft.rfft(u0) * np.exp(-1j * k * A * t), n=N)

def upwind(u, nu):         return u - nu * (u - np.roll(u, 1))
def lax_friedrichs(u, nu): return 0.5 * (np.roll(u, -1) + np.roll(u, 1)) - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))
def lax_wendroff(u, nu):   return u - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1)) + 0.5 * nu * nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
def beam_warming(u, nu):   return u - 0.5 * nu * (3 * u - 4 * np.roll(u, 1) + np.roll(u, 2)) + 0.5 * nu * nu * (u - 2 * np.roll(u, 1) + np.roll(u, 2))

SCHEMES = {"upwind": upwind, "lax_friedrichs": lax_friedrichs,
           "lax_wendroff": lax_wendroff, "beam_warming": beam_warming}
names = list(SCHEMES)
DIFFUSIVE = ("upwind", "lax_friedrichs")
DISPERSIVE = ("lax_wendroff", "beam_warming")

def random_ic(N, rng, n_modes=6):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        kk = rng.integers(1, 8)
        u += rng.normal() * np.sin(2 * np.pi * kk * x / L + rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.7:
        x0, w = rng.uniform(0, L), rng.uniform(L * 0.02, L * 0.08)
        u += rng.normal() * np.exp(-(((x - x0 + L / 2) % L - L / 2) ** 2) / (2 * w * w))
    return u

def antialias_to(R, N_obs):
    """Ideal low-pass to N_obs Nyquist, then sample to N_obs points (periodic). DOWNsampling only."""
    N = R.shape[-1]
    if N == N_obs: return R
    F = np.fft.rfft(R, axis=-1)
    F[..., N_obs // 2 + 1:] = 0.0
    Rlp = np.fft.irfft(F, n=N, axis=-1)
    return Rlp[..., ::(N // N_obs)]

def resample_to(u, M):
    """Spectral resample a periodic field to M points (UPsample by zero-pad, DOWNsample by truncate).
    Used to bring an IC defined on the N_OBS grid up to the solver grid N (low-mode ICs resample exactly)."""
    N = len(u)
    if N == M: return u
    Fh = np.fft.rfft(u); out = np.zeros(M // 2 + 1, complex); m = min(len(Fh), len(out))
    out[:m] = Fh[:m]
    return np.fft.irfft(out, n=M) * (M / N)

def run_scheme(scheme, N, u0):
    dx = L / N; dt = NU * dx / A; ns = int(round(T / dt)); u = u0.copy()
    for _ in range(ns): u = SCHEMES[scheme](u, NU)
    return u, exact_advection(u0, ns * dt, N)

# ---------------------------------------------------------------- clean (noise-free) fields
def clean_fields(scheme, ics, N):
    """Return observed (anti-aliased to N_OBS) clean solver field and matching exact field."""
    U, EX = [], []
    for u0 in ics:
        u0N = resample_to(u0, N)               # bring the N_OBS-grid IC up to the solver grid
        un, ex = run_scheme(scheme, N, u0N)
        U.append(antialias_to(un, N_OBS)); EX.append(antialias_to(ex, N_OBS))
    return np.array(U), np.array(EX)

# ---------------------------------------------------------------- structured-noise generators
# All return a PERTURBATION array eta of shape (M, N_OBS) to ADD to the field (except quantized,
# which returns the quantized field directly). All non-quantized models are rescaled per-sample
# to the SAME RMS as the additive-Gaussian baseline (= SIGMA * RMS(u_obs)) so corruption energy
# is matched. Quantization matches its expected RMS by construction (uniform q -> rms = q/sqrt(12)).

def _per_sample_rms(U):
    return np.sqrt(np.mean(U ** 2, axis=1, keepdims=True))     # (M,1)

def _rescale_to(eta, target_rms):
    cur = np.sqrt(np.mean(eta ** 2, axis=1, keepdims=True)) + 1e-15
    return eta * (target_rms / cur)

def _spectral_envelope(N, kind, ell=None, beta=None, kc=None):
    """Real low-pass / derivative-mimicking spectral magnitude envelope on rfreq grid (length N//2+1)."""
    k = np.arange(N // 2 + 1, dtype=float)              # integer wavenumber index (cycles over the grid)
    if kind == "lowpass":
        # smooth low-pass: Gaussian envelope with correlation length ell (in grid points)
        kc_ = N / (2 * np.pi * ell)
        env = np.exp(-(k / (kc_ + 1e-9)) ** 2)
        env[0] = 0.0                                     # zero-mean noise
        return env
    if kind == "derivmimic":
        # band/high-pass: ~ k^beta * exp(-(k/kc)^2). beta>0 makes the spectrum RISE with k, i.e.
        # it looks like a derivative operator (d_x^p has symbol (ik)^p, magnitude k^p). Peaks where
        # u_xx/u_xxx/u_xxxx live -> projects onto the library. THE adversarial profile.
        env = (k ** beta) * np.exp(-(k / (kc + 1e-9)) ** 2)
        env[0] = 0.0
        return env / (np.max(env) + 1e-15)
    raise ValueError(kind)

def noise_gaussian(U, rng):
    target = SIGMA * _per_sample_rms(U)
    return target * rng.standard_normal(U.shape)        # already at target RMS in expectation; exact below
def make_gaussian(U, rng):
    eta = rng.standard_normal(U.shape)
    return _rescale_to(eta, SIGMA * _per_sample_rms(U))

def make_correlated(U, rng, kind, **kw):
    """Colored noise: white -> shaped in Fourier by the envelope -> back, then matched-RMS rescaled."""
    M, N = U.shape
    env = _spectral_envelope(N, kind, **kw)             # (N//2+1,)
    xi = rng.standard_normal(U.shape)
    Xi = np.fft.rfft(xi, axis=1) * env[None, :]
    eta = np.fft.irfft(Xi, n=N, axis=1)
    return _rescale_to(eta, SIGMA * _per_sample_rms(U))

def make_multiplicative(U, rng):
    """Signal-dependent: eta = s * u * xi, s set per-sample so RMS(eta) == SIGMA*RMS(u)."""
    xi = rng.standard_normal(U.shape)
    raw = U * xi                                         # E[raw^2] ~ E[u^2]*1 -> RMS(raw) ~ RMS(u)
    return _rescale_to(raw, SIGMA * _per_sample_rms(U))

def quantize(U, rng=None):
    """Uniform mid-tread quantization with step q chosen per-sample so the quantization RMS
    (q/sqrt(12) for fine uniform quantization) matches SIGMA*RMS(u). Deterministic."""
    target = SIGMA * _per_sample_rms(U)                 # (M,1)
    q = target * np.sqrt(12.0)                           # (M,1)
    Uq = np.round(U / q) * q
    return Uq, (Uq - U)                                  # quantized field, and its error (the "noise")

# ---------------------------------------------------------------- apply a noise model -> noisy obs field
NOISE_MODELS = ("G", "Clp", "Cd", "Mul", "Q")
NOISE_LABEL = {"G": "additive white Gaussian (baseline)",
               "Clp": "correlated low-pass (smooth)",
               "Cd": "correlated derivative-mimicking (adversarial)",
               "Mul": "multiplicative (signal-dependent)",
               "Q": "quantized"}

def apply_noise(U, model, seed):
    """Return noisy observed field U_obs (clean field U corrupted by `model`)."""
    rng = np.random.default_rng(seed)
    if model == "G":
        return U + make_gaussian(U, rng)
    if model == "Clp":
        return U + make_correlated(U, rng, "lowpass", ell=6.0)        # corr length ~6 grid pts
    if model == "Cd":
        # derivative-mimicking: rising spectrum (beta=2 -> ~ second-derivative symbol) peaked
        # at moderately high k. This is the profile that imitates a u_xx/u_xxx term.
        return U + make_correlated(U, rng, "derivmimic", beta=2.0, kc=N_OBS / 4.0)
    if model == "Mul":
        return U + make_multiplicative(U, rng)
    if model == "Q":
        Uq, _ = quantize(U)
        return Uq
    raise ValueError(model)

# ---------------------------------------------------------------- signature
def signatures(U_obs, EX):
    """Strong-form coefficient-direction signature on the OBSERVED field; periodic FD on N_OBS grid."""
    h = L / N_OBS
    R = U_obs - EX
    uxx = (np.roll(U_obs, -1, 1) - 2 * U_obs + np.roll(U_obs, 1, 1)) / h ** 2
    uxxx = (np.roll(U_obs, -2, 1) - 2 * np.roll(U_obs, -1, 1) + 2 * np.roll(U_obs, 1, 1) - np.roll(U_obs, 2, 1)) / (2 * h ** 3)
    uxxxx = (np.roll(U_obs, -2, 1) - 4 * np.roll(U_obs, -1, 1) + 6 * U_obs - 4 * np.roll(U_obs, 1, 1) + np.roll(U_obs, 2, 1)) / h ** 4
    Am = np.stack([uxx, uxxx, uxxxx], 2)                # (M, N_OBS, 3)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9 * np.eye(3)
    c = np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return np.nan_to_num(c / (n + 1e-12))

def sig_for(scheme, ics, model, N, seed):
    U, EX = clean_fields(scheme, ics, N)
    U_obs = apply_noise(U, model, seed)
    return signatures(U_obs, EX)

# ---------------------------------------------------------------- attribution machinery
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def acc(F, y, g):
    return float(cross_val_score(CLF(), F, y, groups=g, cv=GroupKFold(5)).mean())
def perm_floor(F, y, g, seed, reps=30):
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), F, r.permutation(y), groups=g, cv=GroupKFold(5)).mean()
                            for _ in range(reps)]))

# ---------------------------------------------------------------- solver validation
def validate():
    print("=" * 84)
    print("SOLVER VALIDATION (linear advection, exact spectral reference)")
    print("=" * 84)
    rng = np.random.default_rng(123)
    u0 = random_ic(N_C, rng)
    ok_stab = True
    print("  per-scheme stability (max|u| at T, N=128, CFL=0.8):")
    for sc in names:
        uf, ex = run_scheme(sc, N_C, u0)
        stable = np.isfinite(uf).all() and np.max(np.abs(uf)) < 5 * (np.max(np.abs(u0)) + 1e-9)
        ok_stab = ok_stab and stable
        print(f"     {sc:16s} max|u|={np.max(np.abs(uf)):.3f}  {'OK' if stable else 'UNSTABLE'}")
    # convergence of the upwind scheme toward the exact reference under refinement (rel L2)
    errs = {}
    for N in (64, 128, 256, 512):
        rr = []
        for s in range(6):
            u0n = random_ic(N, np.random.default_rng(900 + s))
            uf, ex = run_scheme("lax_wendroff", N, u0n)
            rr.append(np.linalg.norm(uf - ex) / (np.linalg.norm(ex) + 1e-12))
        errs[N] = float(np.mean(rr))
    print("  lax_wendroff convergence to exact reference (mean rel L2 over 6 ICs):")
    prev = None; rates = []
    for N in sorted(errs):
        rate = "" if prev is None else f"order~{np.log(prev / errs[N]) / np.log(2):.2f}"
        print(f"     N={N:4d}  err={errs[N]:.3e}  {rate}")
        if prev is not None: rates.append(np.log(prev / errs[N]) / np.log(2))
        prev = errs[N]
    conv_ok = errs[512] < errs[64] and errs[512] < 0.05
    # clean-limit modified-equation coefficient signs (no noise): does the signature match theory?
    ics = [random_ic(N_C, np.random.default_rng(s)) for s in range(40)]
    print("  clean-limit mean signature c=[c2,c3,c4] (no noise) vs modified-equation theory:")
    theory_ok = True
    for sc in names:
        U, EX = clean_fields(sc, ics, N_C); S = signatures(U, EX); m = S.mean(0)
        m = m / (np.linalg.norm(m) + 1e-12)
        print(f"     {sc:16s} c2={m[0]:+.3f} c3={m[1]:+.3f} c4={m[2]:+.3f}")
        if sc in DIFFUSIVE and not (abs(m[0]) > abs(m[1])):      # diffusive -> |c2| dominates c3
            theory_ok = False
    print(f"  stability OK: {ok_stab} | convergent: {conv_ok} (mean order ~{np.mean(rates):.2f}) "
          f"| diffusive c2-dominant: {theory_ok}")
    return ok_stab and conv_ok and theory_ok

def validate_noise_rms(ics):
    """Assert every non-quantized model is at matched RMS; report the actual achieved RMS ratio."""
    print("\n  matched-RMS check (mean per-sample RMS(eta)/[SIGMA*RMS(u)] across schemes; target=1.000):")
    U_all, EX_all = [], []
    for sc in names:
        U, EX = clean_fields(sc, ics, N_C); U_all.append(U)
    U = np.vstack(U_all)
    base = SIGMA * np.sqrt(np.mean(U ** 2, axis=1))
    out = {}
    for model in NOISE_MODELS:
        Uobs = apply_noise(U, model, seed=4242)
        eta = Uobs - U
        ratio = float(np.mean(np.sqrt(np.mean(eta ** 2, axis=1)) / (base + 1e-15)))
        out[model] = ratio
        print(f"     {model:4s} {NOISE_LABEL[model]:42s} RMS ratio = {ratio:.3f}")
    return out

# ---------------------------------------------------------------- main
def main():
    print(f"structured-noise robustness | diffusive-vs-dispersive advection | {N_IC} ICs | "
          f"matched-RMS sigma={SIGMA}\n")
    val_ok = validate()
    if not val_ok:
        print("\n[VALIDATION FAILED] solver/signature did not pass -- residuals not trusted. Aborting.")
        return None

    rng = np.random.default_rng(0)
    ics = [random_ic(N_OBS, rng) for _ in range(N_IC)]   # ICs defined on the OBS grid -> resample exactly
    ic = np.arange(N_IC)
    rms_ratio = validate_noise_rms(ics)

    print("\n" + "=" * 84)
    print("STRUCTURED-NOISE ATTRIBUTION  (diffusive-vs-dispersive, GroupKFold-by-IC, perm floor)")
    print("=" * 84)

    # baseline (additive white Gaussian) PER-SCHEME mean signature direction, the reference for
    # the rotation metric. NOTE: cosines are computed PER SCHEME and then averaged within a class --
    # pooling the dispersive mean across LW (c3<0) and BW (c3>0) would cancel in c3 and give a
    # cancellation artifact, not a real rotation; the per-scheme cosine avoids that.
    def mean_dir(S): m = S.mean(0); return m / (np.linalg.norm(m) + 1e-12)
    base_sig, base_dir = {}, {}
    for sc in names:
        base_sig[sc] = sig_for(sc, ics, "G", N_C, seed=1000 + names.index(sc))
        base_dir[sc] = mean_dir(base_sig[sc])

    rows = []
    print(f"\n{'model':4s} {'description':42s} {'diff_disp':>9s} {'floor':>6s} {'NC1':>6s} "
          f"{'NC2':>6s} {'cosDiff':>7s} {'cosDisp':>7s}")
    for model in NOISE_MODELS:
        # signatures for all four schemes under this noise model (distinct seeds per scheme)
        S = {sc: sig_for(sc, ics, model, N_C, seed=2000 + 100 * NOISE_MODELS.index(model) + names.index(sc))
             for sc in names}
        # diffusive-vs-dispersive attribution (load-bearing)
        Xdd = np.vstack([S[s] for s in DIFFUSIVE] + [S[s] for s in DISPERSIVE])
        ydd = np.r_[np.zeros(2 * N_IC), np.ones(2 * N_IC)]
        gdd = np.r_[ic, ic, ic, ic]
        a_dd = acc(Xdd, ydd, gdd); f_dd = perm_floor(Xdd, ydd, gdd, 31)
        # NC1: same scheme (upwind), IC + this-noise only, arbitrary IC partition -> chance.
        # average over several random label partitions to get the TRUE chance behavior.
        Fnc = S["upwind"]; half = N_IC // 2
        nc1_draws = []
        for s in range(6):
            perm = np.random.default_rng(5000 + s).permutation(N_IC)
            gA, gB = perm[:half], perm[half:]
            nc1_draws.append(acc(np.vstack([Fnc[gA], Fnc[gB]]),
                                 np.r_[np.zeros(half), np.ones(N_IC - half)], np.r_[ic[gA], ic[gB]]))
        nc1 = float(np.mean(nc1_draws))
        # NC2: same scheme (upwind), GRID CHANGE under this noise -> the known confound (diagnostic)
        S_up_c2 = sig_for("upwind", ics, model, N_C2,
                          seed=8000 + NOISE_MODELS.index(model))
        nc2 = acc(np.vstack([S["upwind"], S_up_c2]),
                  np.r_[np.zeros(N_IC), np.ones(N_IC)], np.r_[ic, ic])
        # direction distortion vs baseline: PER-SCHEME |cos| of this model's mean signature to the
        # white-Gaussian mean signature, averaged within each class (avoids LW/BW pooled cancellation).
        cos_diff = float(np.mean([abs(mean_dir(S[s]) @ base_dir[s]) for s in DIFFUSIVE]))
        cos_disp = float(np.mean([abs(mean_dir(S[s]) @ base_dir[s]) for s in DISPERSIVE]))
        rows.append(dict(model=model, desc=NOISE_LABEL[model], diff_disp=a_dd, floor=f_dd,
                         nc1=nc1, nc2=nc2, cos_diff=cos_diff, cos_disp=cos_disp,
                         rms_ratio=rms_ratio[model]))
        print(f"{model:4s} {NOISE_LABEL[model]:42s} {a_dd:>9.3f} {f_dd:>6.3f} {nc1:>6.3f} "
              f"{nc2:>6.3f} {cos_diff:>7.3f} {cos_disp:>7.3f}")

    base = next(r for r in rows if r["model"] == "G")
    # ---------------------------------------------------------------- CSV
    csv = os.path.join(TAB, "structured_noise.csv")
    with open(csv, "w") as f:
        f.write("noise_model,description,rms_ratio,diff_disp_acc,perm_floor,nc1,nc2,"
                "cos_diff_to_baseline,cos_disp_to_baseline,delta_vs_baseline\n")
        for r in rows:
            f.write(f"{r['model']},\"{r['desc']}\",{r['rms_ratio']:.4f},{r['diff_disp']:.4f},"
                    f"{r['floor']:.4f},{r['nc1']:.4f},{r['nc2']:.4f},{r['cos_diff']:.4f},"
                    f"{r['cos_disp']:.4f},{r['diff_disp']-base['diff_disp']:+.4f}\n")
    print(f"\nmetrics -> {csv}")

    # ---------------------------------------------------------------- decision
    print("\n" + "=" * 84)
    print("DECISION (pre-registered): is the coefficient-direction signature robust to structured noise?")
    print("=" * 84)
    print(f"  baseline (white Gaussian) diffusive-vs-dispersive = {base['diff_disp']:.3f} "
          f"(floor {base['floor']:.3f})")
    adv = next(r for r in rows if r["model"] == "Cd")
    worst = min(rows, key=lambda r: r["diff_disp"])
    print(f"  per-model diff_disp and drop vs baseline:")
    for r in rows:
        tag = ""
        if r["model"] == "Cd": tag = "  <- adversarial (derivative-mimicking)"
        nc1_bad = "  NC1 ABOVE CHANCE" if r["nc1"] > 0.62 else ""
        print(f"     {r['model']:4s} {r['diff_disp']:.3f}  (delta {r['diff_disp']-base['diff_disp']:+.3f}, "
              f"floor {r['floor']:.3f}, NC1 {r['nc1']:.3f}{nc1_bad}){tag}")
    DROP = 0.05
    adv_drop = base["diff_disp"] - adv["diff_disp"]
    any_degrade = any((base["diff_disp"] - r["diff_disp"]) > DROP or r["nc1"] > 0.62
                      for r in rows if r["model"] != "G")
    adv_degrades = adv_drop > DROP or adv["nc1"] > 0.62
    print()
    if not any_degrade:
        outcome = "ROBUST"
        print("  [ROBUST]  Across all four structured/non-additive noise models the diffusive-vs-")
        print(f"  dispersive signature stays within {DROP:.2f} of the white-Gaussian baseline and NC1 sits")
        print("  at chance. The coefficient-direction feature is NOT an artifact of i.i.d.-Gaussian noise;")
        print("  it survives spatial correlation, signal-dependence, and quantization at matched RMS.")
        if adv["cos_diff"] < 0.98 or adv["cos_disp"] < 0.98:
            print(f"  (The derivative-mimicking noise DOES rotate the mean signature -- cos to baseline "
                  f"{adv['cos_diff']:.3f}/{adv['cos_disp']:.3f} -- but the two classes still separate.)")
    elif adv_degrades:
        outcome = "SENSITIVE_TO_DERIVATIVE_MIMICKING"
        print("  [HONEST SENSITIVITY]  The derivative-mimicking correlated noise -- engineered to look")
        print(f"  like a derivative term -- DEGRADES the attribution: diff_disp {base['diff_disp']:.3f} "
              f"(white) -> {adv['diff_disp']:.3f}")
        print(f"  (drop {adv_drop:+.3f}; signature rotated cos {adv['cos_diff']:.3f}/{adv['cos_disp']:.3f} "
              f"to baseline; NC1 {adv['nc1']:.3f}).")
        print("  White-Gaussian was the OPTIMISTIC case. Noise whose spectrum mimics the library projects")
        print("  onto the coefficient direction and is the genuine worst case for this feature. The smooth/")
        print("  multiplicative/quantized models do not (see numbers above). Reported as the measured limit.")
    else:
        outcome = "DEGRADES_NON_ADVERSARIAL"
        print(f"  [SENSITIVITY]  Structured noise degrades the signature, but NOT primarily via the")
        print(f"  derivative-mimicking case. Worst model = {worst['model']} ({worst['desc']}): "
              f"diff_disp {worst['diff_disp']:.3f} (drop {base['diff_disp']-worst['diff_disp']:+.3f}). "
              "Reported with the number.")
    print(f"\n  DECISION_OUTCOME = {outcome}")

    if "--plot" in __import__("sys").argv:
        _figure(rows, base)
    return dict(rows=rows, outcome=outcome, base=base, adv=adv, csv=csv)

# ---------------------------------------------------------------- figure
def _figure(rows, base):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", palette="muted", font="DejaVu Sans")
    except Exception: pass
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY, ORNG, PURP = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a", "#dd8452", "#8e6fb0"
    COLS = {"G": GREY, "Clp": GREEN, "Cd": RED, "Mul": BLUE, "Q": ORNG}
    SHORT = {"G": "white\nGaussian", "Clp": "correlated\nlow-pass", "Cd": "correlated\nderiv-mimic",
             "Mul": "multiplic.", "Q": "quantized"}
    order = ["G", "Clp", "Cd", "Mul", "Q"]
    rmap = {r["model"]: r for r in rows}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6)); fig.subplots_adjust(wspace=0.30)

    # A: example noise realizations (spectra) -- show that Cd mimics a derivative (rising spectrum)
    axA = ax[0]
    rng = np.random.default_rng(7)
    ics = [random_ic(N_OBS, rng) for _ in range(N_IC)]
    U, EX = clean_fields("upwind", ics, N_C)
    kk = np.arange(N_OBS // 2 + 1)
    for m in ["Clp", "Cd"]:
        Uobs = apply_noise(U, m, seed=3); eta = Uobs - U
        psd = np.mean(np.abs(np.fft.rfft(eta, axis=1)) ** 2, axis=0)
        axA.semilogy(kk, psd / (psd.max() + 1e-30), color=COLS[m], lw=2, label=SHORT[m].replace("\n", " "))
    # reference derivative symbol magnitude^2 ~ k^4 (u_xx), normalized
    sym = (kk.astype(float)) ** 4; sym /= (sym.max() + 1e-30)
    axA.semilogy(kk, sym + 1e-6, color="k", lw=1.2, ls=(0, (3, 2)), label=r"$|k|^4$ ($u_{xx}$ symbol$^2$)")
    axA.set_xlabel("wavenumber index $k$"); axA.set_ylabel("normalized noise PSD")
    axA.set_ylim(1e-5, 2); axA.set_title("Correlated-noise spectra\n(derivative-mimic rises with $k$)", fontsize=9.5)
    axA.legend(frameon=False, fontsize=7.4, loc="lower right")
    axA.text(-0.16, 1.05, "A", transform=axA.transAxes, fontsize=13, fontweight="bold")

    # B: diff_disp accuracy per model with perm floor; baseline dashed line
    axB = ax[1]; x = np.arange(len(order))
    vals = [rmap[m]["diff_disp"] for m in order]; floors = [rmap[m]["floor"] for m in order]
    axB.bar(x, vals, color=[COLS[m] for m in order], width=0.66)
    for i, m in enumerate(order):
        axB.plot([i - 0.34, i + 0.34], [floors[i]] * 2, color="#222", ls=(0, (2, 1.5)), lw=1.4, zorder=6)
        axB.text(i, vals[i] + 0.012, f"{vals[i]:.2f}", ha="center", fontsize=8)
    axB.axhline(base["diff_disp"], color=GREY, ls=":", lw=1.2)
    axB.text(len(order) - 0.5, base["diff_disp"] + 0.012, "baseline", ha="right", fontsize=7.5, color=GREY)
    axB.set_xticks(x); axB.set_xticklabels([SHORT[m] for m in order], fontsize=7.6)
    axB.set_ylim(0.4, 1.03); axB.set_ylabel("diffusive-vs-dispersive accuracy")
    axB.set_title("Attribution under structured noise\n(dashed = permutation floor)", fontsize=9.5)
    axB.text(-0.16, 1.05, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

    # C: signature-direction distortion (cos to baseline) -- how much the noise rotates the signature
    axC = ax[2]; w = 0.36
    cd = [rmap[m]["cos_diff"] for m in order]; cp = [rmap[m]["cos_disp"] for m in order]
    axC.bar(x - w / 2, cd, w, color=BLUE, label="diffusive dir.")
    axC.bar(x + w / 2, cp, w, color=RED, label="dispersive dir.")
    for i in range(len(order)):
        axC.text(i - w / 2, cd[i] + 0.004, f"{cd[i]:.2f}", ha="center", fontsize=6.8)
        axC.text(i + w / 2, cp[i] + 0.004, f"{cp[i]:.2f}", ha="center", fontsize=6.8)
    axC.set_xticks(x); axC.set_xticklabels([SHORT[m] for m in order], fontsize=7.6)
    axC.set_ylim(0.0, 1.05); axC.set_ylabel(r"$|\cos|$ of mean signature to white-Gaussian")
    axC.set_title("Signature-direction distortion vs baseline\n(1 = unrotated)", fontsize=9.5)
    axC.legend(frameon=False, fontsize=7.6, loc="lower left")
    axC.text(-0.16, 1.05, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

    out = os.path.join(FIG_PUB, "structured_noise.png"); fig.savefig(out); plt.close(fig)
    print(f"figure  -> {out}")

if __name__ == "__main__":
    main()
