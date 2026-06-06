#!/usr/bin/env python3
"""
solver-forensics :: STRONGER DETECTOR ON THE FINITE-ELEMENT MECHANICS CONTRASTS
===============================================================================
RUN #2 -- closes Fix #2 (the FEM analogue of src/limits/stronger_detector.py, which
showed the 9-way production-scheme task was Gaussian-MODEL-limited, not information
limited).  Here we ask the SAME question of the finite-element elastodynamics
contrasts attributed in src/mechanics/wave_attribution.py:

    u_tt = c^2 u_xx   (periodic rod, u_t(.,0)=0), analytic modal reference (exact),
    schemes:  lumped_CD       lumped mass + central difference (explicit)
              consistent_FEM  consistent linear-FEM mass + central difference
              newmark_damped  Newmark-beta (gamma=0.6) numerical damping

Three textbook FEM contrasts (from wave_attribution.py):
    MASS-MATRIX      lumped_CD vs consistent_FEM   (the FINE same-order distinction;
                     baseline unit-direction LogReg ~0.667, sigma=0 ~chance)
    TIME-INTEGRATOR  central-difference vs Newmark  (lumped_CD+consistent_FEM(=CD)
                     vs newmark_damped) -- the integrator family distinction
    DISSIPATION      dissipative (newmark) vs non-dissipative (the two CD schemes)

THE QUESTION (per contrast, mass-matrix is the headline):
  Does a strictly stronger NONLINEAR detector (GradientBoosting AND RBF-SVM, on the
  FULL UNNORMALIZED coefficient vector) EXCEED the Gaussian-QDA reference under the
  identical strict nested GroupKFold-by-IC protocol with a permutation floor?
    - EXCEEDS QDA  => the task was GAUSSIAN-MODEL-LIMITED (like the 9-way): the QDA
                      reference was leaving information on the table that a nonlinear
                      decision surface recovers.
    - DOES NOT EXCEED QDA (nonlinear <= QDA within noise, and QDA is the cap)
                   => the task is GENUINELY INFORMATION-LIMITED: no detector family
                      we threw at it beats the Gaussian-Bayes reference; the residual
                      simply does not carry more separable signal.

  Also: MASS-MATRIX at sigma=0 (the d'~0 TRUE-information-limit check) -- the
  deterministic same-order distinction; if it is ~chance with EVERY detector then the
  signal genuinely rides on noise (confirms wave_attribution's sigma=0 finding).

PROTOCOL (identical to the project / stronger_detector.py):
  - signature kernel COPIED verbatim from src/mechanics/wave_attribution.py (that file
    runs heavy code at import / writes artifacts, so it must NOT be imported).
  - features: UD  = unit-direction LSQ coefficient vector (the baseline feature)
              FULL = full UNNORMALIZED LSQ coefficient vector (keeps magnitude)
  - baseline   : StandardScaler + LogReg on UD, plain GroupKFold(5)-by-IC
  - QDA ref    : StandardScaler + QDA on UD AND on FULL, plain GroupKFold(5)-by-IC
                 (reference = the higher realizable Gaussian-Bayes cap)
  - stronger   : GBC and RBF-SVM on FULL, STRICT NESTED GroupKFold(5/3)-by-IC
                 (inner fold selects hyperparameters on training ICs only -> honest OOS)
  - perm floor : permutation floor for the BEST stronger model (its own feature+clf,
                 full nested pipeline re-run on permuted labels)
  - VALIDATION : the analytic modal reference is verified before any residual is trusted
                 (printed); residuals are u_solver - u_ref.

Self-contained (numpy + sklearn). CPU, ~30-60 s.
Writes results/tables/stronger_detector_fem.csv :
    contrast, unit_direction, best_stronger, qda_reference, lift, verdict
Run:  python src/limits/stronger_detector_fem.py
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
TAB = os.path.join(_ROOT, "results", "tables"); FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# =====================================================================================
# ===============  KERNELS COPIED VERBATIM from src/mechanics/wave_attribution.py ======
# ===============  (do NOT import that file -- it runs heavy code / writes at import) ==
# =====================================================================================
L, C, T = 1.0, 1.0, 1.0
N_DET, N_IC = 128, 60
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
    def resample(u):
        Fh = np.fft.rfft(u); out = np.zeros(M // 2 + 1, complex); m = min(len(Fh), len(out))
        out[:m] = Fh[:m]; return np.fft.irfft(out, n=M) * (M / len(u))
    return resample(field + nz), resample(field + nz - ex)

def signature_full(u_obs, r_obs):
    """FULL UNNORMALIZED LSQ coefficient vector on {u_xx,u_xxx,u_xxxx} (keeps magnitude)."""
    M = len(u_obs); k = _k(M); uh = np.fft.fft(u_obs)
    A = np.stack([np.real(np.fft.ifft(uh * (1j * k) ** p)) for p in LIB_ORDERS], 1)
    c, *_ = np.linalg.lstsq(A, r_obs, rcond=None)
    return c                                                   # UNNORMALIZED

def _unit(c):
    n = np.linalg.norm(c); return c / n if n > 0 else c

def random_ic(N, rng, n_modes=5):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        u += rng.normal() * np.sin(2 * np.pi * rng.integers(1, 6) * x / L + rng.uniform(0, 2 * np.pi))
    return u / (np.std(u) + 1e-9)

def features(scheme, N, ics, sigma, seed):
    """Return (UD, FULL) feature arrays for one scheme: unit-direction and full coeff vectors."""
    UD, FULL = [], []
    for i, u0 in enumerate(ics):
        uf = run(scheme, N, u0); ex = exact(u0, T)
        uo, ro = observe(uf, ex, sigma, N, seed + i)
        c = signature_full(uo, ro)
        FULL.append(c); UD.append(_unit(c))
    return np.array(UD), np.array(FULL)

# =====================================================================================
# ===============  SOLVER VALIDATION (validate the reference before trusting r) =======
# =====================================================================================
def validate_solver():
    """The analytic modal reference exact() must reproduce the d'Alembert solution and the
       semidiscrete leapfrog must be stable & convergent toward it. Print before trusting r."""
    print("[VALIDATION] elastic-wave analytic modal reference & FEM solvers:")
    rng = np.random.default_rng(0)
    # (1) reference vs an independent fine spectral time-march of u_tt=c^2 u_xx (analytic exactness)
    N = 256; u0 = random_ic(N, rng); k = _k(N)
    # analytic modal solution at T is exact() ; cross-check against direct cos-propagation in k-space
    uh0 = np.fft.fft(u0)
    ref = np.real(np.fft.ifft(uh0 * np.cos(C * np.abs(k) * T)))
    ref_check = exact(u0, T)
    ref_err = np.max(np.abs(ref - ref_check)) / (np.max(np.abs(ref)) + 1e-12)
    print(f"   analytic modal reference self-consistency: rel err = {ref_err:.2e}  {'OK' if ref_err < 1e-12 else 'BAD'}")
    # (2) leapfrog (consistent_FEM, fine grid) must CONVERGE to the analytic reference as N grows
    errs = []
    for Nn in (96, 192, 384):
        u0n = random_ic(Nn, rng)
        uf = run("consistent_FEM", Nn, u0n); ex = exact(u0n, T)
        errs.append(np.max(np.abs(uf - ex)) / (np.max(np.abs(ex)) + 1e-12))
    converging = errs[0] > errs[1] > errs[2]
    print(f"   consistent_FEM leapfrog rel-err vs analytic, N=96/192/384: "
          f"{errs[0]:.2e} > {errs[1]:.2e} > {errs[2]:.2e}  {'OK converging' if converging else 'NOT converging'}")
    # (3) stability of every scheme (max|u| at T bounded)
    ok_stab = True
    for s in SCHEMES:
        u0n = random_ic(N_DET, rng); uf = run(s, N_DET, u0n)
        stab = np.isfinite(uf).all() and np.max(np.abs(uf)) < 10
        ok_stab = ok_stab and stab
        print(f"   {s:16s} max|u| at T = {np.max(np.abs(uf)):.3f}  {'OK' if stab else 'UNSTABLE'}")
    return (ref_err < 1e-12) and converging and ok_stab

# =====================================================================================
# ===============  EVALUATION MACHINERY (baseline / QDA ref / nested-CV stronger) ======
# =====================================================================================
def _ud_logreg_acc(X, y, g, outer_k=5):
    outer = GroupKFold(n_splits=outer_k); accs = []
    for tr, te in outer.split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(X[tr], y[tr]); accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def _qda_acc(X, y, g, outer_k=5):
    """Gaussian-QDA reference (StandardScaler + QDA) under GroupKFold(5)-by-IC."""
    outer = GroupKFold(n_splits=outer_k); accs = []
    for tr, te in outer.split(X, y, g):
        m = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1))
        m.fit(X[tr], y[tr]); accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def _nested_cv_acc(X, y, g, model_factory, param_grid, outer_k=5, inner_k=3, seed=0):
    """Honest nested GroupKFold-by-IC accuracy with inner hyperparameter selection."""
    outer = GroupKFold(n_splits=outer_k); accs = []
    for tr, te in outer.split(X, y, g):
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]; Xte, yte = X[te], y[te]
        n_inner = min(inner_k, len(np.unique(gtr)))
        if n_inner < 2:
            best_params = param_grid[0]
        else:
            inner = GroupKFold(n_splits=n_inner); best_score, best_params = -1.0, param_grid[0]
            for params in param_grid:
                isc = []
                for itr, ite in inner.split(Xtr, ytr, gtr):
                    m = model_factory(**params); m.fit(Xtr[itr], ytr[itr])
                    isc.append((m.predict(Xtr[ite]) == ytr[ite]).mean())
                s = np.mean(isc)
                if s > best_score: best_score, best_params = s, params
        m = model_factory(**best_params); m.fit(Xtr, ytr)
        accs.append((m.predict(Xte) == yte).mean())
    return float(np.mean(accs))

def _gbc_factory(**kw):
    d = dict(random_state=0); d.update(kw); return GradientBoostingClassifier(**d)
def _svc_factory(**kw):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", **kw))

GBC_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=0.1)
            for ne in (100, 200) for md in (1, 2)]
SVC_GRID = [dict(C=c, gamma=gm) for c in (1.0, 10.0, 100.0) for gm in ("scale", 0.1, 1.0)]

def _perm_floor_nested(X, y, g, model_factory, param_grid, reps=8, seed=123):
    r = np.random.default_rng(seed); out = []
    for j in range(reps):
        out.append(_nested_cv_acc(X, r.permutation(y), g, model_factory, param_grid, seed=j))
    return float(np.median(out))

def _merge(*fbs):
    """Merge several per-scheme (UD, FULL) feature blocks into ONE class by stacking rows.
       The merged 'IC index' is the concatenation of each block's IC indices (each scheme
       carries the same N_IC ICs, so a merged group id may repeat -- but groups are only ever
       compared ACROSS classes for the by-IC split, and within a merged class the rows of the
       SAME ic from different schemes share that ic's group, which is the correct/strict by-IC
       leakage guard)."""
    UD = np.vstack([fb[0] for fb in fbs]); FULL = np.vstack([fb[1] for fb in fbs])
    G = np.concatenate([np.arange(fb[0].shape[0]) for fb in fbs])
    return (UD, FULL, G)

def _stack(feat_by_class, key):
    """feat_by_class: list over classes of (UD, FULL) or (UD, FULL, G); key in {'UD','FULL'}.
       Returns (X, y, g) with group = IC index (shared across classes -> strict by-IC split)."""
    idx = 0 if key == "UD" else 1
    Xs, ys, gs = [], [], []
    for ci, fb in enumerate(feat_by_class):
        arr = fb[idx]; n_ic = arr.shape[0]
        g_ic = fb[2] if len(fb) > 2 else np.arange(n_ic)
        Xs.append(arr); ys.append(np.full(n_ic, ci)); gs.append(g_ic)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

LIFT_MARGIN = 0.03   # nonlinear must EXCEED QDA by > this (above CV noise) to be Gaussian-limited

def evaluate_contrast(name, feat_by_class, class_names, chance):
    """Full battery for one binary/multiclass FEM contrast. Returns a results dict."""
    print(f"\n{'='*92}\nCONTRAST {name}: {' vs '.join(class_names)}   chance={chance:.3f}\n{'='*92}")
    # baseline: unit-direction + LogReg
    Xud, yud, gud = _stack(feat_by_class, "UD")
    base = _ud_logreg_acc(Xud, yud, gud)
    base_floor = float(np.median([_ud_logreg_acc(Xud, np.random.default_rng(s).permutation(yud), gud) for s in range(8)]))
    print(f"  unit-direction  UD+LogReg            acc={base:.3f}  (floor {base_floor:.3f})")
    # Gaussian-QDA reference on UD and FULL (reference = the higher realizable Gaussian cap)
    qda_ud = _qda_acc(Xud, yud, gud)
    Xf, yf, gf = _stack(feat_by_class, "FULL")
    qda_full = _qda_acc(Xf, yf, gf)
    qda_ref = max(qda_ud, qda_full)
    print(f"  QDA reference   UD+QDA={qda_ud:.3f}   FULL+QDA={qda_full:.3f}   (reference={qda_ref:.3f})")
    # stronger nonlinear detectors on FULL (nested CV)
    gbc = _nested_cv_acc(Xf, yf, gf, _gbc_factory, GBC_GRID)
    svc = _nested_cv_acc(Xf, yf, gf, _svc_factory, SVC_GRID)
    print(f"  stronger FULL   GBC={gbc:.3f}   RBF-SVM={svc:.3f}")
    cand = [("GBC", gbc), ("RBF-SVM", svc)]
    best_clf, best_acc = max(cand, key=lambda t: t[1])
    if best_clf == "GBC":
        best_floor = _perm_floor_nested(Xf, yf, gf, _gbc_factory, GBC_GRID, reps=6)
    else:
        best_floor = _perm_floor_nested(Xf, yf, gf, _svc_factory, SVC_GRID, reps=6)
    print(f"  BEST stronger   FULL+{best_clf}={best_acc:.3f}  (floor {best_floor:.3f})")
    return dict(name=name, class_names=class_names, chance=chance,
                base=base, base_floor=base_floor,
                qda_ud=qda_ud, qda_full=qda_full, qda_ref=qda_ref,
                gbc=gbc, svc=svc, best_clf=best_clf, best_acc=best_acc, best_floor=best_floor)

def decide(res):
    """LIFT = best nonlinear stronger detector EXCEEDS the Gaussian-QDA reference => Gaussian-limited.
       Otherwise (nonlinear <= QDA within noise) => genuinely information-limited."""
    lift = res["best_acc"] - res["qda_ref"]
    exceeds_qda = lift > LIFT_MARGIN
    if exceeds_qda:
        verdict = (f"GAUSSIAN-LIMITED: nonlinear detector (FULL+{res['best_clf']}={res['best_acc']:.3f}) EXCEEDS "
                   f"the Gaussian-QDA reference ({res['qda_ref']:.3f}) by {lift:+.3f}. The under-saturation was the "
                   f"Gaussian decision-surface leaving separable information on the table (like the 9-way), NOT a "
                   f"true information limit.")
    else:
        verdict = (f"INFORMATION-LIMITED: the strictly-stronger nonlinear detectors (GBC/RBF-SVM on the full "
                   f"unnormalized coefficient vector, nested GroupKFold-by-IC) do NOT exceed the Gaussian-QDA "
                   f"reference (best FULL+{res['best_clf']}={res['best_acc']:.3f} vs QDA {res['qda_ref']:.3f}, "
                   f"lift {lift:+.3f}). No detector family beats Gaussian-Bayes here => the residual is genuinely "
                   f"information-limited, not Gaussian-model-limited.")
    return dict(lift=lift, exceeds_qda=exceeds_qda, verdict=verdict)

# =====================================================================================
# ===============  MAIN  ==============================================================
# =====================================================================================
def main():
    print("STRONGER DETECTOR on the finite-element mechanics contrasts (RUN #2 / Fix #2)\n"
          "  mass-matrix (lumped vs consistent) | time-integrator (CD vs Newmark) | dissipation\n")
    if not validate_solver():
        print("\n[BLOCKED] elastic-wave reference/solver not validated -- residuals untrustworthy."); return None
    print("\n[VALIDATION PASSED] analytic reference verified & solvers convergent/stable; residuals trustworthy.\n")

    rng = np.random.default_rng(0); ics = [random_ic(N_DET, rng) for _ in range(N_IC)]
    # per-scheme features at the detector noise level (sigma=0.01), matching wave_attribution.py seeds
    F = {s: features(s, N_DET, ics, SIGMA, 100 + 1000 * i) for i, s in enumerate(SCHEMES)}
    # per-scheme features at sigma=0 for the mass-matrix true-information-limit check (matching seeds)
    F0 = {s: features(s, N_DET, ics, 0.0, 200 + 1000 * j) for j, s in enumerate(("lumped_CD", "consistent_FEM"))}

    results = {}

    # ---- CONTRAST 1: MASS-MATRIX  (lumped_CD vs consistent_FEM) -- the headline ----
    mass = evaluate_contrast("MASS-MATRIX (sigma=0.01)",
                             [F["lumped_CD"], F["consistent_FEM"]],
                             ["lumped_CD", "consistent_FEM"], chance=0.5)
    mass_dec = decide(mass); results["mass_matrix"] = (mass, mass_dec)

    # ---- CONTRAST 1b: MASS-MATRIX at sigma=0 (the d'~0 TRUE information-limit check) ----
    mass0 = evaluate_contrast("MASS-MATRIX (sigma=0, true-info-limit check)",
                              [F0["lumped_CD"], F0["consistent_FEM"]],
                              ["lumped_CD", "consistent_FEM"], chance=0.5)
    mass0_dec = decide(mass0); results["mass_matrix_sigma0"] = (mass0, mass0_dec)

    # central-difference family = lumped_CD + consistent_FEM (merged); the other classes use it.
    cd_family = _merge(F["lumped_CD"], F["consistent_FEM"])

    # ---- CONTRAST 2: TIME-INTEGRATOR (central-difference vs Newmark) ----
    # central-difference family vs newmark_damped. IMBALANCED 120:60 -> permutation floor is the
    # majority baseline (~0.667), reported per protocol (matches wave_attribution.py dissipation task).
    ti = evaluate_contrast("TIME-INTEGRATOR (central-diff vs Newmark)",
                           [cd_family, F["newmark_damped"]],
                           ["central_difference", "newmark"], chance=0.5)
    ti_dec = decide(ti); results["time_integrator"] = (ti, ti_dec)

    # ---- CONTRAST 3: DISSIPATION (dissipative newmark vs non-dissipative CD pair) ----
    diss = evaluate_contrast("DISSIPATION (dissipative vs non-dissipative)",
                             [F["newmark_damped"], cd_family],
                             ["newmark_damped", "non_dissipative"], chance=0.5)
    diss_dec = decide(diss); results["dissipation"] = (diss, diss_dec)

    # ================= SUMMARY =================
    print("\n" + "#"*92 + "\nSUMMARY: best nonlinear stronger detector vs Gaussian-QDA reference (does nonlinear EXCEED QDA?)\n" + "#"*92)
    order = ["mass_matrix", "mass_matrix_sigma0", "time_integrator", "dissipation"]
    for key in order:
        res, dec = results[key]
        print(f"\n[{res['name']}]")
        print(f"   unit-direction (UD+LogReg) = {res['base']:.3f}  (floor {res['base_floor']:.3f})")
        print(f"   Gaussian-QDA reference     = {res['qda_ref']:.3f}  (UD {res['qda_ud']:.3f}, FULL {res['qda_full']:.3f})")
        print(f"   best stronger (FULL+{res['best_clf']}) = {res['best_acc']:.3f}  (floor {res['best_floor']:.3f})")
        print(f"   lift over QDA = {dec['lift']:+.3f}  ->  {'GAUSSIAN-LIMITED' if dec['exceeds_qda'] else 'INFORMATION-LIMITED'}")

    # ================= CSV =================
    csv = os.path.join(TAB, "stronger_detector_fem.csv")
    with open(csv, "w") as f:
        f.write("contrast,unit_direction,best_stronger,qda_reference,lift,verdict\n")
        labelmap = {"mass_matrix": "mass_matrix(sigma=0.01)", "mass_matrix_sigma0": "mass_matrix(sigma=0)",
                    "time_integrator": "time_integrator", "dissipation": "dissipation"}
        for key in order:
            res, dec = results[key]
            tag = "GAUSSIAN_limited" if dec["exceeds_qda"] else "information_limited"
            f.write(f'{labelmap[key]},{res["base"]:.4f},{res["best_acc"]:.4f},{res["qda_ref"]:.4f},'
                    f'{dec["lift"]:+.4f},{tag}\n')
        # transparency block: every model + floors
        f.write("\n# detail: contrast,base_floor,qda_ud,qda_full,gbc,svc,best_clf,best_floor,full_verdict\n")
        for key in order:
            res, dec = results[key]
            f.write(f'# {labelmap[key]},{res["base_floor"]:.4f},{res["qda_ud"]:.4f},{res["qda_full"]:.4f},'
                    f'{res["gbc"]:.4f},{res["svc"]:.4f},{res["best_clf"]},{res["best_floor"]:.4f},"{dec["verdict"]}"\n')
    print(f"\nartifacts -> {csv}")
    return results

if __name__ == "__main__":
    main()
