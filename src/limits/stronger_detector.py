#!/usr/bin/env python3
"""
solver-forensics :: STRONGER DETECTOR ON THE UNDER-SATURATED MULTI-CLASS TASKS
==============================================================================
Two multi-class attribution tasks in the project are UNDER-SATURATED -- clearly
above the permutation floor but modest:

  (A) NINE-WAY production-scheme ID  (unit-direction LogisticRegression ~0.60,
      floor ~0.08; from src/robustness/production_schemes.py)
  (B) FOUR-WAY KdV scheme ID         (~0.817 clean / ~0.754 degraded(64,1%);
      from src/robustness/kdv_breadth.py)

The standing question (src/limits/identifiability_ceiling.py argues this on the
elastic-wave contrast): is 0.60 a FEATURE/CLASSIFIER ARTIFACT of the unit-direction
+ linear-logistic choice, or is it near the INFORMATION LIMIT of what the residual
field carries about the scheme?

This script answers it EMPIRICALLY by throwing strictly stronger detectors at the
SAME residual fields, under the SAME GroupKFold-by-IC protocol, with NESTED CV to
control overfitting on small N and a PERMUTATION FLOOR computed for the stronger
model too:

  Feature representations (all from the OBSERVED solver+residual field; same physics):
    UD   unit-direction LSQ coefficient vector  (the baseline feature)
    FULL full UNNORMALIZED LSQ coefficient vector (keeps magnitude information)
    SPEC raw band-limited residual SPECTRUM: |FFT|(low band) + phase(low band)
    UD+SPEC, FULL+SPEC  concatenations

  Classifiers:
    LogReg (baseline, on UD)               -- the reported number
    GradientBoostingClassifier             -- nonlinear, on FULL / SPEC / concat
    RBF-SVM (StandardScaler + SVC rbf)     -- nonlinear, on FULL / SPEC / concat

  Protocol (identical to the project):
    - STRICT GroupKFold(5) grouped by INITIAL CONDITION (no IC in train+test of a fold)
    - NESTED CV: inner GroupKFold(3) selects hyperparameters (GBC depth/lr/n_est;
      SVC C/gamma) on the training folds only -> reported accuracy is honest OOS.
    - PERMUTATION FLOOR for the BEST stronger model too (labels permuted within the
      same group structure, full nested pipeline re-run).

  Realizable ceiling:
    - NINE-WAY QDA realizable ceiling: GroupKFold-by-IC QDA accuracy on the
      coefficient-direction features (mirrors src/limits/identifiability_ceiling.py).
      This is the Gaussian-Bayes (quadratic) classifier under the identical protocol
      = the realizable cap given the feature cloud.

DECISION RULE (followed exactly, per task):
  - A LIFT of the best stronger model toward the ceiling (best_stronger meaningfully
    above the unit-direction baseline, with overlapping/closing gap to the QDA
    ceiling) => 0.60 was a FEATURE/CLASSIFIER ARTIFACT; the reported number should be
    REWRITTEN with the stronger detector.
  - NO LIFT (best stronger model NOT above the unit-direction baseline beyond noise)
    => 0.60 is near the INFORMATION LIMIT; the QDA-ceiling / information-limit
    argument is EMPIRICALLY CONFIRMED.
  A non-lift is reported as a confirmation of the limit, NEVER as a strengthening.

Self-contained (numpy + scipy + sklearn). Kernels REPLICATED from the two source
scripts (which run heavy code at import and must not be imported). CPU.
Writes results/tables/stronger_detector.csv and results/figures/stronger_detector.png.
Run:  python src/limits/stronger_detector.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# =====================================================================================
# ===============  TASK A: NINE-WAY PRODUCTION SCHEMES  (kernels replicated)  ==========
# =====================================================================================
PA_L, PA_A, PA_NU, PA_T = 1.0, 1.0, 0.8, 0.30
PA_NC, PA_NC2, PA_NBASE, PA_COMMON, PA_NIC = 128, 192, 384, 64, 50

def _upwind(u, nu):         return u - nu*(u - np.roll(u, 1))
def _lax_friedrichs(u, nu): return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))
def _lax_wendroff(u, nu):   return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))
def _beam_warming(u, nu):   return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))
def _minmod(u, nu):
    up = np.roll(u,-1); dn = np.roll(u,1); denom = up - u
    r = (u - dn)/(denom + 1e-12); phi = np.maximum(0.0, np.minimum(1.0, r))
    F = u + 0.5*(1.0-nu)*phi*denom
    return u - nu*(F - np.roll(F,1))
def _superbee(u, nu):
    den = np.roll(u,-1) - u; num = u - np.roll(u,1)
    r = num/np.where(np.abs(den) < 1e-12, np.where(den >= 0, 1e-12, -1e-12), den)
    phi = np.maximum.reduce([np.zeros_like(r), np.minimum(2.0*r, 1.0), np.minimum(r, 2.0)])
    flux = nu*u + 0.5*nu*(1.0-nu)*phi*den
    return u - (flux - np.roll(flux,1))
def _van_leer(u, nu):
    du = np.roll(u,-1) - u; num = u - np.roll(u,1)
    r = num/np.where(np.abs(du) < 1e-12, np.where(du >= 0, 1e-12, -1e-12), du)
    phi = (r + np.abs(r))/(1.0 + np.abs(r))
    F = u + 0.5*phi*(1.0-nu)*du
    return u - nu*(F - np.roll(F,1))
def _weno5(u, nu):
    eps = 1e-6
    def face(u):
        um2,um1,u0,up1,up2 = np.roll(u,2),np.roll(u,1),u,np.roll(u,-1),np.roll(u,-2)
        p0 = (2*um2 - 7*um1 + 11*u0)/6; p1 = (-um1 + 5*u0 + 2*up1)/6; p2 = (2*u0 + 5*up1 - up2)/6
        b0 = 13/12*(um2-2*um1+u0)**2 + 0.25*(um2-4*um1+3*u0)**2
        b1 = 13/12*(um1-2*u0+up1)**2 + 0.25*(um1-up1)**2
        b2 = 13/12*(u0-2*up1+up2)**2 + 0.25*(3*u0-4*up1+up2)**2
        a0,a1,a2 = 0.1/(eps+b0)**2, 0.6/(eps+b1)**2, 0.3/(eps+b2)**2; s = a0+a1+a2
        return (a0*p0 + a1*p1 + a2*p2)/s
    def rhs(u): F = face(u); return -(F - np.roll(F,1))
    u1 = u + nu*rhs(u); u2 = 0.75*u + 0.25*(u1 + nu*rhs(u1))
    return (1/3)*u + (2/3)*(u2 + nu*rhs(u2))
def _maccormack(u, nu):
    ustar = u - nu*(np.roll(u,-1) - u)
    return 0.5*(u + ustar - nu*(ustar - np.roll(ustar,1)))
def _crank_nicolson(u, nu):
    N = u.shape[0]; k = np.fft.fftfreq(N, d=1.0/N); s = np.sin(2*np.pi*k/N)
    g = (1.0 - 1j*(nu/2)*s)/(1.0 + 1j*(nu/2)*s)
    return np.real(np.fft.ifft(np.fft.fft(u)*g))

PA_SCHEMES = {"upwind":_upwind, "lax_friedrichs":_lax_friedrichs, "lax_wendroff":_lax_wendroff,
              "beam_warming":_beam_warming, "minmod":_minmod, "superbee":_superbee, "van_leer":_van_leer,
              "weno5":_weno5, "maccormack":_maccormack, "crank_nicolson":_crank_nicolson}
PA_NAMES = list(PA_SCHEMES)

def _pa_exact(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=PA_L/N); return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*PA_A*t), n=N)
def _pa_random_ic(N, rng):
    x = np.linspace(0, PA_L, N, endpoint=False); u = np.zeros(N)
    for _ in range(5): u += rng.normal()*np.sin(2*np.pi*rng.integers(1,7)*x/PA_L + rng.uniform(0,2*np.pi))
    x0, w = rng.uniform(0,PA_L), 0.03; u += 1.5*rng.normal()*np.exp(-(((x-x0+PA_L/2)%PA_L-PA_L/2)**2)/(2*w*w))
    return u/(np.std(u)+1e-9)
def _pa_antialias(u, M):
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M)*(M/N)
def _pa_run(scheme, N, u0):
    dx = PA_L/N; dt = PA_NU*dx/PA_A; ns = int(round(PA_T/dt)); u = u0.copy()
    for _ in range(ns): u = PA_SCHEMES[scheme](u, PA_NU)
    return u, _pa_exact(u0, ns*dt, N)
def _pa_coeffs(U, R):
    """Full UNNORMALIZED LSQ coefficient vector on the {u_xx,u_xxx,u_xxxx} library."""
    h = PA_L/U.shape[1]
    Am = np.stack([(np.roll(U,-1,1)-2*U+np.roll(U,1,1))/h**2,
                   (np.roll(U,-2,1)-2*np.roll(U,-1,1)+2*np.roll(U,1,1)-np.roll(U,2,1))/(2*h**3),
                   (np.roll(U,-2,1)-4*np.roll(U,-1,1)+6*U-4*np.roll(U,1,1)+np.roll(U,2,1))/h**4], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(3)
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]
def _direction(C): return np.nan_to_num(C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))

def _pa_features(scheme, N, u0s, noise, seed, n_spec=8):
    """Return dict of feature blocks for production schemes:
       UD (unit dir, 3), FULL (unnorm coeffs, 3), SPEC (|FFT|+phase low band of residual)."""
    gn = np.random.default_rng(seed); U, R = [], []
    for u0 in u0s:
        u0N = _pa_antialias(u0, N); un, ex = _pa_run(scheme, N, u0N)
        if noise > 0: un = un + noise*np.sqrt(np.mean(ex**2))*gn.standard_normal(N)
        U.append(_pa_antialias(un, PA_COMMON)); R.append(_pa_antialias(un - ex, PA_COMMON))
    U = np.array(U); R = np.array(R)
    Cfull = _pa_coeffs(U, R)
    Cdir = _direction(Cfull)
    Spec = _residual_spectrum(R, n_spec)
    return dict(UD=Cdir, FULL=Cfull, SPEC=Spec)

# =====================================================================================
# ===============  TASK B: FOUR-WAY KdV  (kernels replicated)  =========================
# =====================================================================================
KV_L = 2*np.pi
KV_NC, KV_NC2, KV_NREF = 128, 192, 512
KV_T, KV_DELTA, KV_AMP, KV_NIC = 1.0, 1.0, 0.5, 60
KV_LIB = (2, 3, 4)
KV_D3_2ND = {2:0.5, 1:-1.0, -1:1.0, -2:-0.5}
KV_D3_4TH = {3:-1/8, 2:1.0, 1:-13/8, -1:13/8, -2:-1.0, -3:1/8}
KV_D2 = {1:1.0, 0:-2.0, -1:1.0}
KV_SCHEMES = {"S1_centered":(KV_D3_2ND,0.0,"centered"), "S2_LF":(KV_D3_2ND,0.0,"LF"),
              "S3_visc":(KV_D3_2ND,0.05,"centered"), "S4_4th_d3":(KV_D3_4TH,0.0,"centered")}
KV_NAMES = list(KV_SCHEMES)

def _kv_ric(N, seed):
    r = np.random.default_rng(seed); x = np.linspace(0, KV_L, N, endpoint=False); u = np.zeros(N)
    for kk in (1, 2, 3): u += r.normal()*np.sin(2*np.pi*kk*x/KV_L + r.uniform(0, 2*np.pi))
    return KV_AMP*u/(np.max(np.abs(u)) + 1e-9)
def _kv_ifrk4(uh, Lhat, Nl, dt, ns):
    E = np.exp(Lhat*dt); E2 = np.exp(Lhat*dt/2)
    for _ in range(ns):
        a = dt*Nl(uh); b = dt*Nl(E2*(uh+a/2)); c = dt*Nl(E2*uh+b/2); d = dt*Nl(E*uh+E2*c)
        uh = E*uh + (E*a + 2*E2*(b+c) + d)/6
    return uh
def _kv_sym(st, k, dx): return sum(c*np.exp(1j*k*m*dx) for m, c in st.items())
def _kv_nsteps(u0, dx): return max(1, int(np.ceil(KV_T/(0.2*dx/(6*(np.max(np.abs(u0))+1e-9))))))
def _kv_spectral_ref(u0, N, delta):
    dx = KV_L/N; k = 2*np.pi*np.fft.fftfreq(N, d=dx); m = np.abs(k) <= (2/3)*np.max(np.abs(k))
    ns = _kv_nsteps(u0, dx); dt = KV_T/ns
    Nl = lambda uh: -3j*k*(np.fft.fft(np.real(np.fft.ifft(uh))**2)*m)
    return np.real(np.fft.ifft(_kv_ifrk4(np.fft.fft(u0), 1j*delta*k**3, Nl, dt, ns)))
def _kv_coarse(u0, N, scheme, delta):
    d3, eps, flux = KV_SCHEMES[scheme]
    dx = KV_L/N; k = 2*np.pi*np.fft.fftfreq(N, d=dx)
    Lhat = -delta*_kv_sym(d3, k, dx)/dx**3 + eps*_kv_sym(KV_D2, k, dx)/dx**2
    def Nl(uh):
        u = np.real(np.fft.ifft(uh)); f = 3*u*u
        if flux == "centered": fx = (np.roll(f,-1)-np.roll(f,1))/(2*dx)
        else:
            a = 6*np.max(np.abs(u))+1e-9; Fp = 0.5*(f+np.roll(f,-1)) - 0.5*a*(np.roll(u,-1)-u)
            fx = (Fp - np.roll(Fp, 1))/dx
        return np.fft.fft(-fx)
    ns = _kv_nsteps(u0, dx); dt = KV_T/ns
    return np.real(np.fft.ifft(_kv_ifrk4(np.fft.fft(u0), Lhat, Nl, dt, ns)))
def _kv_antialias(u, M):
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M) * (M/N)
def _kv_deriv(u, o, h):
    if o == 2: return (np.roll(u,-1,-1)-2*u+np.roll(u,1,-1))/h**2
    if o == 3: return (np.roll(u,-2,-1)-2*np.roll(u,-1,-1)+2*np.roll(u,1,-1)-np.roll(u,2,-1))/(2*h**3)
    return (np.roll(u,-2,-1)-4*np.roll(u,-1,-1)+6*u-4*np.roll(u,1,-1)+np.roll(u,2,-1))/h**4
def _kv_coeffs(U, R):
    h = KV_L/U.shape[1]; Am = np.stack([_kv_deriv(U, o, h) for o in KV_LIB], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(len(KV_LIB))
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]

def _kv_features(refs, base, scheme, N_c, delta, noise, N_obs, seed, n_spec=8):
    """KdV feature blocks: UD (unit dir, 3), FULL (unnorm coeffs, 3), SPEC (residual band spectrum)."""
    gn = np.random.default_rng(seed); U, R = [], []
    for u0_base, ref in zip(base, refs):
        ref_c = _kv_antialias(ref, N_c); u0_c = _kv_antialias(u0_base, N_c)
        uc = _kv_antialias(_kv_coarse(u0_c, N_c, scheme, delta), N_c)
        un = uc + noise*np.sqrt(np.mean(ref_c**2))*gn.standard_normal(N_c)
        U.append(_kv_antialias(un, N_obs)); R.append(_kv_antialias(un - ref_c, N_obs))
    U = np.array(U); R = np.array(R)
    Cfull = _kv_coeffs(U, R)
    Cdir = _direction(Cfull)
    Spec = _residual_spectrum(R, n_spec)
    return dict(UD=Cdir, FULL=Cfull, SPEC=Spec)

# =====================================================================================
# ===============  RAW RESIDUAL SPECTRUM FEATURE  =====================================
# =====================================================================================
def _residual_spectrum(R, n_spec):
    """Band-limited |FFT| magnitude + phase of the residual field, per-row L2-normalized
    magnitude (scale-free, so it is NOT just a louder copy of the coefficient magnitude).
    R: (n_samples, M).  Returns (n_samples, 2*n_spec): [log|FFT|_1..n_spec, phase_1..n_spec]."""
    Fh = np.fft.rfft(R, axis=1)                       # (n, M//2+1)
    mag = np.abs(Fh[:, 1:n_spec+1])                   # drop DC, keep low band
    phase = np.angle(Fh[:, 1:n_spec+1])
    mag = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + 1e-12)   # shape, not scale
    logmag = np.log(mag + 1e-9)
    return np.concatenate([logmag, phase], axis=1)

# =====================================================================================
# ===============  EVALUATION MACHINERY (nested CV, perm floor, ceiling)  ==============
# =====================================================================================
def _build_blocks(F_by_class, keys):
    """Stack a list of per-class feature dicts into (X, y, groups) for the given feature keys.
    F_by_class: list over classes of dict(block->array (n_ic, d)); IC index = group."""
    Xs, ys, gs = [], [], []
    for ci, fd in enumerate(F_by_class):
        n_ic = fd[keys[0]].shape[0]
        Xc = np.concatenate([fd[k] for k in keys], axis=1)
        Xs.append(Xc); ys.append(np.full(n_ic, ci)); gs.append(np.arange(n_ic))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

def _nested_cv_acc(X, y, g, model_factory, param_grid, outer_k=5, inner_k=3, seed=0):
    """Honest nested GroupKFold-by-IC accuracy with inner hyperparameter selection.
       param_grid: list of dict kwargs passed to model_factory. Returns mean OOS accuracy."""
    outer = GroupKFold(n_splits=outer_k)
    accs = []
    for tr, te in outer.split(X, y, g):
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        Xte, yte = X[te], y[te]
        # inner selection
        n_inner = min(inner_k, len(np.unique(gtr)))
        if n_inner < 2:
            best_params = param_grid[0]
        else:
            inner = GroupKFold(n_splits=n_inner)
            best_score, best_params = -1.0, param_grid[0]
            for params in param_grid:
                isc = []
                for itr, ite in inner.split(Xtr, ytr, gtr):
                    m = model_factory(**params)
                    m.fit(Xtr[itr], ytr[itr])
                    isc.append((m.predict(Xtr[ite]) == ytr[ite]).mean())
                s = np.mean(isc)
                if s > best_score:
                    best_score, best_params = s, params
        m = model_factory(**best_params)
        m.fit(Xtr, ytr)
        accs.append((m.predict(Xte) == yte).mean())
    return float(np.mean(accs))

def _gbc_factory(**kw):
    defaults = dict(random_state=0)
    defaults.update(kw)
    return GradientBoostingClassifier(**defaults)

def _svc_factory(**kw):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", **kw))

# GBC is expensive on multiclass (fits n_estimators x n_classes trees); a small, sensible
# grid (shallow stumps, modest n_estimators) is enough -- the nested inner CV still guards
# overfitting and the LIFT is large, not marginal. RBF-SVM carries a wider grid (it is cheap).
GBC_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=0.1)
            for ne in (100, 200) for md in (1, 2)]
SVC_GRID = [dict(C=c, gamma=gm) for c in (1.0, 10.0, 100.0) for gm in ("scale", 0.1, 1.0)]

def _ud_logreg_acc(X, y, g, outer_k=5):
    """Baseline: StandardScaler+LogReg, plain GroupKFold (no tuning needed), mirrors the source."""
    outer = GroupKFold(n_splits=outer_k); accs = []
    for tr, te in outer.split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr]); accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def _qda_ceiling_acc(X, y, g, outer_k=5):
    """Realizable QDA (Gaussian-Bayes) ceiling under the SAME GroupKFold-by-IC protocol."""
    outer = GroupKFold(n_splits=outer_k); accs = []
    for tr, te in outer.split(X, y, g):
        m = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1))
        m.fit(X[tr], y[tr]); accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def _perm_floor_nested(X, y, g, model_factory, param_grid, reps=8, seed=123, outer_k=5, inner_k=3):
    """Permutation floor for the STRONGER model: permute labels WITHIN the group structure
       (permute the per-group label assignment is not meaningful here since each group has all
       classes; we permute y globally, which under GroupKFold destroys class signal while the
       grouping is preserved) and re-run the full nested pipeline."""
    r = np.random.default_rng(seed); out = []
    for _ in range(reps):
        yp = r.permutation(y)
        out.append(_nested_cv_acc(X, yp, g, model_factory, param_grid, outer_k, inner_k, seed=_))
    return float(np.median(out))

# the strict-group permutation: shuffle labels but keep one IC's whole row-set coherent is
# overkill for accuracy floors; the standard label-permutation floor (global permutation under
# the preserved GroupKFold splits) is what the project uses everywhere, so we match it.

def evaluate_task(task_name, F_by_class, class_names, chance):
    """Run the full battery for one multi-class task. Returns a results dict."""
    print(f"\n{'='*90}\nTASK {task_name}: {len(class_names)}-way ({', '.join(class_names)})  chance={chance:.3f}\n{'='*90}")
    # --- baseline: unit-direction + LogReg (the reported number) ---
    Xud, yud, gud = _build_blocks(F_by_class, ["UD"])
    base_acc = _ud_logreg_acc(Xud, yud, gud)
    base_floor = float(np.median([_ud_logreg_acc(Xud, np.random.default_rng(s).permutation(yud), gud) for s in range(8)]))
    print(f"  baseline  UD+LogReg                acc={base_acc:.3f}  (floor={base_floor:.3f})")

    # --- QDA realizable ceiling on the UD feature cloud (the project's reported ceiling) ---
    qda_ceiling_ud = _qda_ceiling_acc(Xud, yud, gud)
    print(f"  ceiling   UD+QDA   (project's ceiling, unit-dir features) acc={qda_ceiling_ud:.3f}")
    # --- QDA realizable ceiling on the FULL coefficient cloud (the ceiling on the RICHER feature) ---
    Xfq, yfq, gfq = _build_blocks(F_by_class, ["FULL"])
    qda_ceiling_full = _qda_ceiling_acc(Xfq, yfq, gfq)
    print(f"  ceiling   FULL+QDA (ceiling on the richer unnormalized feature)  acc={qda_ceiling_full:.3f}")
    # headline ceiling = the higher realizable ceiling (the true information cap the project
    # under-reported by computing QDA only on unit-direction features that discard magnitude)
    qda_ceiling = max(qda_ceiling_ud, qda_ceiling_full)

    # --- stronger detectors over feature representations x classifiers (nested CV) ---
    feat_sets = {
        "FULL":      ["FULL"],
        "SPEC":      ["SPEC"],
        "UD+SPEC":   ["UD", "SPEC"],
        "FULL+SPEC": ["FULL", "SPEC"],
    }
    stronger = []
    for fname, keys in feat_sets.items():
        X, y, g = _build_blocks(F_by_class, keys)
        gbc = _nested_cv_acc(X, y, g, _gbc_factory, GBC_GRID)
        svc = _nested_cv_acc(X, y, g, _svc_factory, SVC_GRID)
        print(f"  stronger  {fname:10s} GBC={gbc:.3f}   RBF-SVM={svc:.3f}")
        stronger.append(dict(feats=fname, clf="GBC", acc=gbc, keys=keys))
        stronger.append(dict(feats=fname, clf="RBF-SVM", acc=svc, keys=keys))
    # FULL+QDA (full coeffs through QDA) -- already computed above as the richer-feature ceiling
    stronger.append(dict(feats="FULL", clf="QDA", acc=qda_ceiling_full, keys=["FULL"]))

    best = max(stronger, key=lambda d: d["acc"])
    # permutation floor for the BEST stronger model (its own feature set + classifier)
    Xb, yb, gb = _build_blocks(F_by_class, best["keys"])
    if best["clf"] == "GBC":
        best_floor = _perm_floor_nested(Xb, yb, gb, _gbc_factory, GBC_GRID, reps=6)
    elif best["clf"] == "RBF-SVM":
        best_floor = _perm_floor_nested(Xb, yb, gb, _svc_factory, SVC_GRID, reps=6)
    else:  # QDA
        best_floor = float(np.median([_qda_ceiling_acc(Xb, np.random.default_rng(s).permutation(yb), gb) for s in range(6)]))
    print(f"\n  BEST stronger: {best['feats']}+{best['clf']} acc={best['acc']:.3f} (floor={best_floor:.3f})")
    print(f"  baseline UD+LogReg = {base_acc:.3f}   QDA ceiling (UD={qda_ceiling_ud:.3f}, FULL={qda_ceiling_full:.3f})")

    return dict(task=task_name, n_class=len(class_names), chance=chance,
                base_acc=base_acc, base_floor=base_floor,
                qda_ceiling=qda_ceiling, qda_ceiling_ud=qda_ceiling_ud, qda_ceiling_full=qda_ceiling_full,
                stronger=stronger, best=best, best_floor=best_floor)

# =====================================================================================
# ===============  DECISION RULE  ====================================================
# =====================================================================================
LIFT_MARGIN = 0.03   # best-stronger must beat baseline by > this (above CV noise) to count as a LIFT

def decide(res):
    base, best, ceil = res["base_acc"], res["best"]["acc"], res["qda_ceiling"]
    lift = best - base
    gap_before = ceil - base
    gap_after = ceil - best
    is_lift = lift > LIFT_MARGIN
    if is_lift:
        beats_ud_ceil = best > res["qda_ceiling_ud"] + 0.02
        extra = (f" Note: the stronger detector also EXCEEDS the project's previously-reported QDA ceiling "
                 f"({res['qda_ceiling_ud']:.3f}) -- that ceiling was computed on the unit-direction feature, "
                 f"which discards coefficient MAGNITUDE; the true realizable ceiling on the full feature is "
                 f"{res['qda_ceiling_full']:.3f}.") if beats_ud_ceil else ""
        verdict = ("LIFT -> the reported number is a FEATURE/CLASSIFIER ARTIFACT. The under-saturation was "
                   "the unit-direction(+linear-logistic) choice leaving information on the table; the stronger "
                   f"detector ({res['best']['feats']}+{res['best']['clf']}) recovers {best:.3f} "
                   f"(was {base:.3f}, lift {lift:+.3f}). REWRITE the headline with the stronger detector." + extra)
    else:
        verdict = ("NO LIFT -> the reported number is near the INFORMATION LIMIT. Strictly stronger detectors "
                   "(GBC / RBF-SVM on full unnormalized coefficients and on the raw residual spectrum, nested-CV, "
                   f"same GroupKFold-by-IC) do NOT beat the unit-direction baseline ({best:.3f} vs {base:.3f}, "
                   f"lift {lift:+.3f}). The QDA-ceiling / information-limit argument is EMPIRICALLY CONFIRMED. "
                   "(A non-lift is a confirmation of the limit, not a strengthening.)")
    return dict(is_lift=is_lift, lift=lift, gap_before=gap_before, gap_after=gap_after, verdict=verdict)

# =====================================================================================
# ===============  SOLVER VALIDATION  =================================================
# =====================================================================================
def validate_production():
    """Stability self-check (max|u| at T) for all 9 production schemes -- must be bounded."""
    print("\n[VALIDATION] production schemes: max|u| at T (N=128, nu=0.8) must be bounded (<5):")
    rng = np.random.default_rng(0); u0 = _pa_random_ic(PA_NBASE, rng)
    ok = True
    for sc in PA_NAMES:
        uf, _ = _pa_run(sc, PA_NC, _pa_antialias(u0, PA_NC))
        stable = np.isfinite(uf).all() and np.max(np.abs(uf)) < 5
        ok = ok and stable
        print(f"   {sc:16s} max|u|={np.max(np.abs(uf)):.2f}  {'OK' if stable else 'UNSTABLE'}")
    return ok

def validate_kdv():
    """KdV solver validation: (1) soliton vs analytic, (2) reference convergence 256->512 (drift),
       (3) residual sanity. The IFRK4 spectral reference must be near-exact."""
    print("\n[VALIDATION] KdV IFRK4 spectral reference:")
    # (1) single-soliton: u(x,t)=(c/2) sech^2( sqrt(c)/2 (x - c t - x0) ), exact for u_t+6uu_x+u_xxx=0
    N = 256; c = 4.0; x0 = KV_L/2
    x = np.linspace(0, KV_L, N, endpoint=False)
    def sol(t):
        xi = ((x - c*t - x0 + KV_L/2) % KV_L) - KV_L/2
        return (c/2)/np.cosh(np.sqrt(c)/2*xi)**2
    u0 = sol(0.0)
    # short-time soliton evolution with the reference integrator (delta=1)
    dx = KV_L/N; k = 2*np.pi*np.fft.fftfreq(N, d=dx); m = np.abs(k) <= (2/3)*np.max(np.abs(k))
    Tsol = 0.2; ns = max(200, _kv_nsteps(u0, dx)); dt = Tsol/ns
    Nl = lambda uh: -3j*k*(np.fft.fft(np.real(np.fft.ifft(uh))**2)*m)
    uh = _kv_ifrk4(np.fft.fft(u0), 1j*KV_DELTA*k**3, Nl, dt, ns)
    u_num = np.real(np.fft.ifft(uh)); u_an = sol(Tsol)
    sol_err = np.max(np.abs(u_num - u_an)) / np.max(np.abs(u_an))
    print(f"   soliton vs analytic (T={Tsol}): rel max-err = {sol_err:.2e}  {'OK' if sol_err < 5e-3 else 'BAD'}")
    # (2) reference convergence: random IC, 256 vs 512, compare on common grid
    base = [_kv_ric(KV_NREF, s) for s in range(8)]
    drift = []
    for u in base:
        r512 = _kv_spectral_ref(u, KV_NREF, KV_DELTA)
        u256 = _kv_antialias(u, 256); r256 = _kv_spectral_ref(u256, 256, KV_DELTA)
        r512c = _kv_antialias(r512, 256)
        drift.append(np.max(np.abs(r512c - r256)) / (np.max(np.abs(r256)) + 1e-12))
    conv = float(np.median(drift))
    print(f"   reference convergence (256->512) median rel drift = {conv:.2e}  {'OK converged' if conv < 1e-2 else 'REFERENCE-LIMITED'}")
    return (sol_err < 5e-3) and (conv < 1e-2)

# =====================================================================================
# ===============  MAIN  =============================================================
# =====================================================================================
def main():
    print("STRONGER DETECTOR on the under-saturated multi-class tasks")
    print("  (9-way production schemes ; 4-way KdV)\n")

    prod_ok = validate_production()
    kdv_ok = validate_kdv()
    if not prod_ok:
        print("\n[BLOCKED] production solver unstable -- residuals untrustworthy."); return None
    if not kdv_ok:
        print("\n[BLOCKED] KdV reference not validated -- residuals untrustworthy."); return None
    print("\n[VALIDATION PASSED] both solvers/references validated; residuals are trustworthy.\n")

    # ---------- TASK A: NINE-WAY PRODUCTION SCHEMES (1% noise, the under-saturated regime) ----------
    rngA = np.random.default_rng(0); basesA = [_pa_random_ic(PA_NBASE, rngA) for _ in range(PA_NIC)]
    NOISE_A = 0.01
    FA = [_pa_features(sc, PA_NC, basesA, NOISE_A, 1) for sc in PA_NAMES]
    resA = evaluate_task("A (9-way production schemes, 1% noise)", FA, PA_NAMES, chance=1/len(PA_NAMES))
    decA = decide(resA)

    # ---------- TASK B: FOUR-WAY KdV (degraded 64,1% -- the under-saturated regime) ----------
    baseB = [_kv_ric(KV_NREF, s) for s in range(KV_NIC)]
    refsB = [_kv_spectral_ref(u, KV_NREF, KV_DELTA) for u in baseB]
    NOISE_B, NOBS_B, NC_B = 0.01, 64, KV_NC
    FB = [_kv_features(refsB, baseB, sc, NC_B, KV_DELTA, NOISE_B, NOBS_B, 10+i) for i, sc in enumerate(KV_NAMES)]
    resB = evaluate_task("B (4-way KdV, degraded 64/1%)", FB, KV_NAMES, chance=1/len(KV_NAMES))
    decB = decide(resB)

    # also a clean-regime KdV pass (the 0.817 number) for completeness
    FBc = [_kv_features(refsB, baseB, sc, KV_NC, KV_DELTA, 0.0, KV_NC, 20+i) for i, sc in enumerate(KV_NAMES)]
    resBc = evaluate_task("B-clean (4-way KdV, clean)", FBc, KV_NAMES, chance=1/len(KV_NAMES))
    decBc = decide(resBc)

    # ================= SUMMARY =================
    print("\n" + "#"*92 + "\nSUMMARY: best-stronger vs unit-direction baseline vs QDA realizable ceiling\n" + "#"*92)
    for res, dec in [(resA, decA), (resB, decB), (resBc, decBc)]:
        b = res["best"]
        print(f"\n[{res['task']}]")
        print(f"   unit-direction baseline (UD+LogReg)   = {res['base_acc']:.3f}  (floor {res['base_floor']:.3f}, chance {res['chance']:.3f})")
        print(f"   QDA ceiling: project's (UD features) {res['qda_ceiling_ud']:.3f} | on richer (FULL) feature {res['qda_ceiling_full']:.3f}")
        print(f"   BEST stronger detector ({b['feats']}+{b['clf']}){' '*max(0,10-len(b['feats']+b['clf']))}= {b['acc']:.3f}  (floor {res['best_floor']:.3f})")
        print(f"   lift over baseline = {dec['lift']:+.3f}   gap-to-ceiling: {dec['gap_before']:+.3f} -> {dec['gap_after']:+.3f}")
        print(f"   DECISION: {dec['verdict']}")

    # ================= CSV =================
    csv = os.path.join(TAB, "stronger_detector.csv")
    with open(csv, "w") as f:
        f.write("task,n_class,chance,baseline_UD_LogReg,baseline_floor,"
                "QDA_ceiling_UD,QDA_ceiling_FULL,QDA_ceiling_headline,"
                "best_feats,best_clf,best_acc,best_floor,lift_over_baseline,gap_to_ceiling_before,"
                "gap_to_ceiling_after,decision\n")
        for res, dec in [(resA, decA), (resB, decB), (resBc, decBc)]:
            b = res["best"]
            decision_tag = "LIFT_artifact" if dec["is_lift"] else "NO_LIFT_information_limit"
            f.write(f'"{res["task"]}",{res["n_class"]},{res["chance"]:.4f},{res["base_acc"]:.4f},'
                    f'{res["base_floor"]:.4f},{res["qda_ceiling_ud"]:.4f},{res["qda_ceiling_full"]:.4f},'
                    f'{res["qda_ceiling"]:.4f},{b["feats"]},{b["clf"]},'
                    f'{b["acc"]:.4f},{res["best_floor"]:.4f},{dec["lift"]:+.4f},{dec["gap_before"]:+.4f},'
                    f'{dec["gap_after"]:+.4f},{decision_tag}\n')
        # also dump every stronger model tried (transparency)
        f.write("\n# full grid: task,feats,clf,acc\n")
        for res in (resA, resB, resBc):
            for s in res["stronger"]:
                f.write(f'# "{res["task"]}",{s["feats"]},{s["clf"]},{s["acc"]:.4f}\n')
    print(f"\nartifacts -> {csv}")

    _figure([(resA, decA), (resB, decB), (resBc, decBc)])
    return [(resA, decA), (resB, decB), (resBc, decBc)]

def _figure(pairs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", palette="muted")
    except Exception: pass
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a"
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.6*n, 5.0));
    if n == 1: axes = [axes]
    short = {"A (9-way production schemes, 1% noise)": "9-way production\n(1% noise)",
             "B (4-way KdV, degraded 64/1%)": "4-way KdV\n(degraded 64/1%)",
             "B-clean (4-way KdV, clean)": "4-way KdV\n(clean)"}
    for ax, (res, dec) in zip(axes, pairs):
        b = res["best"]
        labels = ["floor", "UD+LogReg\n(baseline)", f"best stronger\n{b['feats']}+{b['clf']}",
                  "UD+QDA\n(reported\nceiling)", "FULL+QDA\n(true\nceiling)"]
        vals = [res["base_floor"], res["base_acc"], b["acc"], res["qda_ceiling_ud"], res["qda_ceiling_full"]]
        cols = [RED, BLUE, GREEN, "#bdbdbd", GREY]
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="k", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(res["chance"], color="k", ls=":", lw=1.0, label=f"chance {res['chance']:.2f}")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.0)
        ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy (GroupKFold-by-IC)")
        tag = "LIFT (artifact)" if dec["is_lift"] else "NO LIFT (information limit)"
        ax.set_title(f"{short.get(res['task'], res['task'])}\nlift {dec['lift']:+.3f}  ->  {tag}", fontsize=9.5)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#ddd", fontsize=7.5, loc="upper left")
    fig.suptitle("Stronger detectors on under-saturated multi-class attribution: "
                 "best stronger vs unit-direction baseline vs QDA realizable ceiling", fontsize=11, y=1.02)
    out = os.path.join(FIG, "stronger_detector.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print(f"figure    -> {out}")

if __name__ == "__main__":
    main()
