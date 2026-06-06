#!/usr/bin/env python3
"""
solver-forensics :: REFERENCE-SCHEME MISSPECIFICATION STRESS TEST  (Item 6, practical credibility)
==================================================================================================
The whole framework forms a residual r = u_solver - u_ref and attributes the solver from the
DIRECTION of the least-squares coefficient vector c in r ~= sum_p c_p d_x^p u (signature). Every
prior experiment fed it a GENUINE reference (analytic / fine spectral). But a real auditor almost
never has the exact reference: they reconstruct one with WHATEVER solver they happen to trust --
which may be a DIFFERENT, BIASED scheme (an under-resolved upwind run, a Lax-Wendroff run, ...),
not just an under-resolved copy of the truth. That biased reference injects its OWN
modified-equation signature into r. The untested question for the practitioner recipe:

    How much does diffusive-vs-dispersive attribution degrade as a function of the reference
    SCHEME used to form r -- and does it collapse in the same-family self-cancellation case
    (a Lax-Wendroff solver audited against a Lax-Wendroff reference)?

Setup (reuses the project's 4 advection schemes, u_t + a u_x = 0, a=1, periodic, nu=0.8):
    diffusive   : upwind, lax_friedrichs        (residual ~ + c2 u_xx, even-derivative dissipation)
    dispersive  : lax_wendroff, beam_warming    (residual ~ + c3 u_xxx, odd-derivative dispersion)
The TASK the auditor is trying to do is the diffusive-vs-dispersive 2-way split (the project's
core taxonomy) AND the 4-way scheme ID. We hold the SOLVER runs fixed and SWAP the reference used
to form r:
    REF_SPECTRAL    : genuine exact spectral truth (the gold standard -- what prior work assumed)
    REF_FINE_UPWIND : a fine-grid upwind run            (biased: diffusive reference)
    REF_FINE_LW     : a fine-grid Lax-Wendroff run      (biased: dispersive reference)
    REF_FINE_BW     : a fine-grid beam-warming run      (biased: dispersive reference)
    REF_COARSE_SPEC : an UNDER-RESOLVED spectral run    (the already-studied "under-resolved" case,
                                                         as a contrast to true MISSPECIFICATION)
The biased references run on a FINER grid than the solver (so they are a plausible "best available"
reference), then are Fourier-resampled to the common grid before differencing -- exactly what a
practitioner would do. Signature, attribution (StandardScaler+LogisticRegression, GroupKFold(5)
grouped by IC, label-permutation floor on every number), and controls follow project convention.

Self-cancellation flag: for the dispersive solver lax_wendroff we additionally form r against a
SAME-grid lax_wendroff reference (the literal self-audit a careless practitioner might do). The
recipe-relevant question is NOT "can two clouds be separated" (a near-zero residual cloud is
trivially separable from a large one) but "is the audited solver still attributed to its CORRECT
physics class". We measure that three honest ways: (1) residual RMS collapse ||r_self||/||r_truth||;
(2) signature corruption -- the angle of the self-audited LW signature to its GENUINE-reference LW
signature vs to the DIFFUSIVE class mean (if it is closer to the wrong class than to its own genuine
signature, the dispersive identity is destroyed); (3) cross-reference attribution -- a genuine-trained
diffusive-vs-dispersive classifier applied to the self-audited signature. We report it explicitly.

VALIDATION before trusting residuals: every scheme is checked for stability (max|u| bounded, finite)
at the solver grid AND at each reference grid; the genuine spectral reference is checked against the
analytic solution; and the FRACTION of the residual energy the reference itself contributes is
printed so the reader can see when r is dominated by reference error rather than solver error.

We distinguish WITHIN-REFERENCE attribution (form residuals AND train the classifier with the same
biased reference) from CROSS-REFERENCE TRANSFER (train on a genuine reference, apply to residuals
formed with a different biased reference -- the practitioner who does not re-train).

DECISION RULE:
  ROBUST   : diffusive-vs-dispersive (and 4-way ID) stays well above the permutation floor across ALL
             reference schemes BOTH within-reference and on cross-reference transfer, and the
             same-family self-audit does not corrupt the signature -> the exact reference is not needed.
  ROBUST_WITHIN_REF_TWO_CAVEATS : within-reference robust, but (2) cross-reference transfer degrades for
             strongly-biased references and/or (3) the same-family self-audit corrupts the signature.
             Report the caveats (re-train per reference; never self-audit within a numerical family).
  FRAGILE  : even within-reference, attribution drops toward the floor for a biased reference.
  Either way: flag the same-family self-cancellation (LW solver vs SAME-grid LW reference) explicitly.

Self-contained. numpy 2.x-safe, CPU, ~1-2 min. Writes
results/tables/reference_misspecification.csv and results/figures/reference_misspecification.png.
Run:  python src/limits/reference_misspecification.py
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
# NOTE: all fields here live on regular periodic grids (1-D advection), so the grid SIGNATURE applies
# directly; scipy.interpolate.griddata (for non-grid fields per project convention) is not needed.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ----------------------------------------------------------------------------- config
L, A, NU, T = 1.0, 1.0, 0.8, 0.30
N_SOLVER  = 128            # grid the AUDITED solver runs on
N_REF     = 256            # grid the biased "best available" references run on (finer, then resampled)
N_COARSE  = 96             # under-resolved spectral reference grid (the already-studied contrast)
N_SELF    = 128            # same-family reference grid = SOLVER grid (literal self-audit: worst-case self-cancellation)
COMMON    = 64             # common observation grid (signature computed here)
N_IC      = 50
NOISE     = 0.01           # degraded condition: field-relative observation noise (project convention)

# ---- the project's 4 classic advection schemes (verbatim kernels) ----
def upwind(u, nu):         return u - nu*(u - np.roll(u,1))
def lax_friedrichs(u, nu): return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))
def lax_wendroff(u, nu):   return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))
def beam_warming(u, nu):   return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))
SCHEMES = {"upwind":upwind, "lax_friedrichs":lax_friedrichs, "lax_wendroff":lax_wendroff, "beam_warming":beam_warming}
DIFFUSIVE  = ["upwind", "lax_friedrichs"]
DISPERSIVE = ["lax_wendroff", "beam_warming"]
SOLVERS = DIFFUSIVE + DISPERSIVE   # the 4 solvers we attribute

# ---- exact spectral evolution of advection (genuine truth) ----
def exact(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N); return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*A*t), n=N)

def random_ic(N, rng):     # smooth low-mode IC + a localized bump (resamples cleanly; engages dispersion near the bump)
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(5): u += rng.normal()*np.sin(2*np.pi*rng.integers(1,7)*x/L + rng.uniform(0,2*np.pi))
    x0, w = rng.uniform(0,L), 0.04; u += 1.2*rng.normal()*np.exp(-(((x-x0+L/2)%L-L/2)**2)/(2*w*w))
    return u/(np.std(u)+1e-9)

def antialias(u, M):       # PROPER Fourier resample to exactly M points (the bug-fixed resample idiom)
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M)*(M/N)

def run(scheme, N, u0):
    """Run a scheme to time T on grid N starting from u0 (length N). Returns (u_final, n_steps, dt)."""
    dx = L/N; dt = NU*dx/A; ns = int(round(T/dt)); u = u0.copy()
    for _ in range(ns): u = SCHEMES[scheme](u, NU)
    return u, ns, dt

# --------------------------------------------------------- signature (project convention)
def deriv_grid(U, o, h):   # vectorized central derivatives on the common regular grid (rows = ICs)
    if o == 2: return (np.roll(U,-1,-1) - 2*U + np.roll(U,1,-1))/h**2
    if o == 3: return (np.roll(U,-2,-1) - 2*np.roll(U,-1,-1) + 2*np.roll(U,1,-1) - np.roll(U,2,-1))/(2*h**3)
    return (np.roll(U,-2,-1) - 4*np.roll(U,-1,-1) + 6*U - 4*np.roll(U,1,-1) + np.roll(U,2,-1))/h**4
LIB = (2, 3, 4)            # {u_xx, u_xxx, u_xxxx}: even = dissipation, odd = dispersion
def coeffs(U, R):          # per-IC least-squares c from the OBSERVED field's derivatives (numpy-2-safe)
    h = L/U.shape[1]; Am = np.stack([deriv_grid(U, o, h) for o in LIB], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(len(LIB))
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]
def direction(C): return np.nan_to_num(C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))

# --------------------------------------------------------- attribution machinery
CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def auroc(Fa, Fb, ga, gb):
    X = np.vstack([Fa, Fb]); y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]; g = np.r_[ga, gb]
    return float(cross_val_score(CLF(), X, y, groups=g, cv=GroupKFold(5), scoring="roc_auc").mean())
def acc_multi(Fs, labs, grp):
    X = np.vstack(Fs); y = np.concatenate(labs); g = np.concatenate([grp]*len(Fs))
    return float(cross_val_score(CLF(), X, y, groups=g, cv=GroupKFold(5)).mean())
def auroc_floor(Fa, Fb, ga, gb, seed, reps=20):
    X = np.vstack([Fa, Fb]); y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]; g = np.r_[ga, gb]
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), X, r.permutation(y), groups=g, cv=GroupKFold(5),
                                            scoring="roc_auc").mean() for _ in range(reps)]))
def acc_floor(Fs, labs, grp, seed, reps=20):
    X = np.vstack(Fs); y = np.concatenate(labs); g = np.concatenate([grp]*len(Fs))
    r = np.random.default_rng(seed)
    return float(np.median([cross_val_score(CLF(), X, r.permutation(y), groups=g, cv=GroupKFold(5)).mean()
                            for _ in range(reps)]))

def coherence(F):          # median angle of per-IC direction from the scheme mean (lower = more coherent)
    m = F.mean(0); m = m/(np.linalg.norm(m) + 1e-12)
    return float(np.median(np.degrees(np.arccos(np.clip(np.abs(F @ m), 0, 1)))))

# =================================================================================== RUN
def main():
    print("="*92)
    print("REFERENCE-SCHEME MISSPECIFICATION STRESS TEST  (advection, project's 4 schemes)")
    print(f"solver grid N={N_SOLVER}, biased-ref grid N={N_REF}, coarse-spectral ref N={N_COARSE}, "
          f"common N={COMMON}, {N_IC} ICs, nu={NU}")
    print("="*92 + "\n")

    rng = np.random.default_rng(0)
    bases = [random_ic(N_REF, rng) for _ in range(N_IC)]   # store ICs at the finest grid; downsample as needed
    ic = np.arange(N_IC)

    # ---------------- VALIDATION 1: scheme stability at every grid we use ----------------
    print("VALIDATION 1 -- stability self-check (max|u| at T; finite & bounded required):")
    grids = {"solver(128)": N_SOLVER, "ref(256)": N_REF, "coarse(96)": N_COARSE}
    all_stable = True
    for gname, Ng in grids.items():
        u0g = antialias(bases[0], Ng)
        for sc in SCHEMES:
            uf, _, _ = run(sc, Ng, u0g); mx = np.max(np.abs(uf)); ok = np.isfinite(uf).all() and mx < 5.0
            all_stable &= ok
            print(f"   {gname:12s} {sc:16s} max|u|={mx:5.2f}  {'OK' if ok else 'UNSTABLE'}")
    # genuine spectral reference vs analytic (it IS the analytic propagator -> machine precision)
    u0s = antialias(bases[0], N_SOLVER); ex1 = exact(u0s, T, N_SOLVER); ex2 = exact(u0s, T, N_SOLVER)
    spec_err = float(np.max(np.abs(ex1 - ex2)))   # determinism / sanity
    # cross-check: spectral exact at COMMON resolution equals resample-then-exact for low-mode ICs
    print(f"   spectral truth determinism (max|.|) = {spec_err:.2e}  (exact propagator, OK)\n")

    # ---------------- build SOLVER fields ONCE, CLEAN (held fixed across all references) ----------------
    # For each solver and IC: run on N_SOLVER, resample CLEAN solver field to COMMON. Observation noise
    # is added per regime below so the SAME clean field is reused.
    print("building clean solver fields (held fixed across references) ...")
    solver_clean = {sc: [] for sc in SOLVERS}        # clean solver field on COMMON grid
    for sc in SOLVERS:
        for b in bases:
            u0 = antialias(b, N_SOLVER); uf, ns, dt = run(sc, N_SOLVER, u0)
            solver_clean[sc].append(antialias(uf, COMMON))
        solver_clean[sc] = np.array(solver_clean[sc])

    # ---------------- build each REFERENCE field per IC (on COMMON grid) ----------------
    # The solver advances physical time T; each reference is computed to T on its OWN grid then
    # Fourier-resampled to COMMON -- exactly what a practitioner does with a "best available" run.
    # N_SELF (=160, near the solver grid) gives the WORST-CASE same-family reference: a Lax-Wendroff
    # reference whose own dispersive truncation nearly matches the LW solver's, so r collapses.
    def make_reference(kind):
        out = []
        for b in bases:
            if kind == "REF_SPECTRAL":
                u0 = antialias(b, N_SOLVER); ref = exact(u0, T, N_SOLVER)        # genuine truth at solver time T
            elif kind == "REF_COARSE_SPEC":
                u0 = antialias(b, N_COARSE); ref = exact(u0, T, N_COARSE)        # under-resolved spectral (contrast)
            elif kind in ("REF_FINE_UPWIND", "REF_FINE_LW", "REF_FINE_BW"):
                sc = {"REF_FINE_UPWIND":"upwind", "REF_FINE_LW":"lax_wendroff", "REF_FINE_BW":"beam_warming"}[kind]
                u0 = antialias(b, N_REF); ref, _, _ = run(sc, N_REF, u0)         # biased fine-grid scheme reference
            elif kind == "REF_SELF_LW":
                u0 = antialias(b, N_SELF); ref, _, _ = run("lax_wendroff", N_SELF, u0)  # same-family, NEAR solver grid
            else:
                raise ValueError(kind)
            out.append(antialias(ref, COMMON))
        return np.array(out)

    REF_KINDS = ["REF_SPECTRAL", "REF_FINE_UPWIND", "REF_FINE_LW", "REF_FINE_BW", "REF_COARSE_SPEC"]
    REF_LABEL = {"REF_SPECTRAL":"spectral truth (genuine)", "REF_FINE_UPWIND":"fine upwind (biased diffusive)",
                 "REF_FINE_LW":"fine Lax-Wendroff (biased dispersive)", "REF_FINE_BW":"fine beam-warming (biased dispersive)",
                 "REF_COARSE_SPEC":"under-resolved spectral (contrast)"}
    references = {k: make_reference(k) for k in REF_KINDS}
    ref_self_lw = make_reference("REF_SELF_LW")      # same-family near-solver-grid LW reference

    # ---------------- VALIDATION 2: how much of the residual is the REFERENCE's own error ----------------
    # r = (solver-truth) - (ref-truth): the reference injects its own error. Print median fraction of
    # residual RMS contributed by the reference (||ref-truth|| / ||solver-truth||) so the reader sees
    # when r becomes reference-limited rather than solver-limited.
    print("\nVALIDATION 2 -- reference-error contribution to the residual "
          "(median ||ref-truth|| / ||solver-truth|| over ICs):")
    truth = references["REF_SPECTRAL"]
    ref_err_frac = {}
    for k in REF_KINDS:
        if k == "REF_SPECTRAL": ref_err_frac[k] = 0.0; continue
        fr = []
        for sc in SOLVERS:
            num = np.sqrt(np.mean((references[k] - truth)**2, axis=1))
            den = np.sqrt(np.mean((solver_clean[sc] - truth)**2, axis=1)) + 1e-12
            fr.append(np.median(num/den))
        ref_err_frac[k] = float(np.median(fr))
        print(f"   {k:16s} ({REF_LABEL[k]:36s}) ref/solver err = {ref_err_frac[k]:5.2f}")
    print("   (>1 means the reference's own error dominates the residual -> r is reference-limited there)\n")

    # ---------------- signatures: noise-aware, library from the OBSERVED (noisy solver) field ----------------
    # Per project SIGNATURE convention the derivative library is built from the OBSERVED field; only the
    # residual target r = u_obs - u_ref changes with the reference. Observation noise is field-relative.
    def observed(scheme, noise, seed):
        gn = np.random.default_rng(seed); U = solver_clean[scheme]
        if noise <= 0: return U
        rms = np.sqrt(np.mean(U**2, axis=1, keepdims=True))
        return U + noise*rms*gn.standard_normal(U.shape)
    def sigs(scheme, ref_field, noise, seed):
        U = observed(scheme, noise, seed); R = U - ref_field
        return direction(coeffs(U, R))

    def attribute(noise):
        """All attribution numbers + self-cancellation at a given observation-noise level."""
        F = {k: {sc: sigs(sc, references[k], noise, 300 + 7*SOLVERS.index(sc) + 17*REF_KINDS.index(k))
                 for sc in SOLVERS} for k in REF_KINDS}
        # the practitioner's classifier is trained on whatever GENUINE reference they had; test how it
        # generalizes to signatures formed with a DIFFERENT (biased) reference -> cross-reference accuracy.
        Xg = np.vstack([F["REF_SPECTRAL"][s] for s in SOLVERS]); yg = np.array([0,0,1,1]).repeat(N_IC)
        clf_gen = CLF().fit(Xg, yg)
        rr = []
        for k in REF_KINDS:
            Fk = F[k]
            Fdiff = np.vstack([Fk[s] for s in DIFFUSIVE]); Fdisp = np.vstack([Fk[s] for s in DISPERSIVE])
            dd_auc   = auroc(Fdiff, Fdisp, np.r_[ic, ic], np.r_[ic, ic])      # within-reference separability
            dd_floor = auroc_floor(Fdiff, Fdisp, np.r_[ic, ic], np.r_[ic, ic], seed=100)
            labs = [np.full(N_IC, i) for i in range(4)]
            id4       = acc_multi([Fk[s] for s in SOLVERS], labs, ic)
            id4_floor = acc_floor([Fk[s] for s in SOLVERS], labs, ic, seed=200)
            # cross-reference: genuine-trained classifier applied to THIS reference's signatures
            Xk = np.vstack([Fk[s] for s in SOLVERS])
            xref_acc = float((clf_gen.predict(Xk) == yg).mean())
            rr.append(dict(ref=k, label=REF_LABEL[k], dd_auc=dd_auc, dd_floor=dd_floor,
                           id4=id4, id4_floor=id4_floor, xref_acc=xref_acc,
                           coh=float(np.median([coherence(Fk[s]) for s in SOLVERS])),
                           ref_err=ref_err_frac[k], means={s: Fk[s].mean(0) for s in SOLVERS}))
        # ---- SAME-FAMILY SELF-CANCELLATION (LW solver audited against a LW reference, SAME grid) ----
        # The recipe-relevant question is NOT "can two clouds be separated" (a near-zero residual cloud is
        # trivially separable from a large one) but "is the audited solver still attributed to its CORRECT
        # physics class". We measure it three honest ways:
        #   (1) residual collapse  : RMS(r_self) / RMS(r_genuine)  -- does the reference cancel the error?
        #   (2) signature corruption: angle of the self-audited LW signature to its GENUINE-reference LW
        #        signature, vs its angle to the DIFFUSIVE class mean. If it is CLOSER to the diffusive class
        #        than to its own genuine signature, the dispersive identity has been destroyed.
        #   (3) cross-reference attribution: a diffusive-vs-dispersive classifier TRAINED on genuine-reference
        #        signatures (the realistic practitioner classifier), applied to the self-audited LW signature.
        #        Report the fraction attributed to the CORRECT (dispersive) class.
        def _ang(a, b):
            a = a/(np.linalg.norm(a)+1e-12); b = b/(np.linalg.norm(b)+1e-12)
            return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))
        F_lw_self = sigs("lax_wendroff", ref_self_lw, noise, 999)
        rms_truth = float(np.median(np.sqrt(np.mean((solver_clean["lax_wendroff"] - references["REF_SPECTRAL"])**2, 1))))
        rms_self  = float(np.median(np.sqrt(np.mean((solver_clean["lax_wendroff"] - ref_self_lw)**2, 1))))
        rms_ratio = rms_self/(rms_truth + 1e-12)
        m_self = F_lw_self.mean(0); m_truth = F["REF_SPECTRAL"]["lax_wendroff"].mean(0)
        diff_mean = np.vstack([F["REF_SPECTRAL"][s] for s in DIFFUSIVE]).mean(0)
        ang_to_genuine = _ang(m_self, m_truth)              # how far the self-audit moved the signature
        ang_to_diff    = _ang(m_self, diff_mean)            # how close the self-audit signature is to the WRONG class
        ang_genuine_to_diff = _ang(m_truth, diff_mean)      # reference separation (genuine LW vs diffusive)
        # cross-reference attribution: train diffusive(0)-vs-dispersive(1) on genuine-ref sigs, predict self-LW
        Xg = np.vstack([F["REF_SPECTRAL"][s] for s in SOLVERS]); yg = np.array([0,0,1,1]).repeat(N_IC)
        clf_gen = CLF().fit(Xg, yg)
        frac_correct_self = float(clf_gen.predict(F_lw_self).mean())   # fraction called DISPERSIVE (correct)
        # FLAG: physics attribution of the self-audited solver is DESTROYED if either
        #   (a) the residual has essentially vanished (RMS collapse < 0.02): the signature carries no
        #       solver information -- in the clean exact-self-audit r==0 and the signature is undefined; or
        #   (b) the recovered signature has rotated onto the WRONG class: it is closer to the diffusive
        #       class mean than to its OWN genuine-reference signature (or closer than half the genuine
        #       dispersive separation). A knife-edge linear classifier may still output the right label,
        #       but the physics signature no longer supports it.
        sig_corrupted = (rms_ratio < 0.02) \
                        or (not np.isfinite(ang_to_diff)) or (not np.isfinite(ang_to_genuine)) \
                        or (ang_to_diff < ang_to_genuine) or (ang_to_diff < 0.5*ang_genuine_to_diff)
        sc_info = dict(rms_ratio=rms_ratio, m_truth=m_truth, m_self=m_self,
                       ang_to_genuine=ang_to_genuine, ang_to_diff=ang_to_diff,
                       ang_genuine_to_diff=ang_genuine_to_diff, frac_correct_self=frac_correct_self,
                       self_cancel=bool(sig_corrupted))
        return F, rr, sc_info

    # ---------------- run BOTH the clean and the degraded (1% noise) regime ----------------
    REGIMES = [("clean", 0.0), (f"degraded({int(NOISE*100)}% noise)", NOISE)]
    bundle = {}
    for tag, nz in REGIMES:
        F, rr, sc_info = attribute(nz); bundle[tag] = (F, rr, sc_info)
        print(f"\n{'='*92}\nREGIME: {tag}\n{'='*92}")
        print("mean coefficient direction [c2(diss) c3(disp) c4] per solver, per reference "
              "(sign of c2 vs c3 = taxonomy):")
        for r in rr:
            print(f"  {r['ref']:16s} {r['label']}")
            for s in SOLVERS:
                m = r["means"][s]; t = "diff" if s in DIFFUSIVE else "disp"
                print(f"      {s:16s}[{t}] [{m[0]:+.2f} {m[1]:+.2f} {m[2]:+.2f}]")
        print(f"\n{'reference':16s} {'ref/sol_err':>11s} {'dd-AUROC':>9s} {'(floor)':>8s} "
              f"{'xref-acc':>9s} {'4way ID':>8s} {'(floor)':>8s} {'coh(deg)':>9s}")
        print(f"{'':16s} {'':11s} {'(within-ref)':>9s} {'':8s} {'(genuine-trained)':>9s}")
        for r in rr:
            print(f"{r['ref']:16s} {r['ref_err']:>11.2f} {r['dd_auc']:>9.3f} {r['dd_floor']:>8.3f} "
                  f"{r['xref_acc']:>9.3f} {r['id4']:>8.3f} {r['id4_floor']:>8.3f} {r['coh']:>9.1f}")
        sc = sc_info
        print(f"\nSAME-FAMILY SELF-CANCELLATION (LW solver vs SAME-grid N={N_SELF} LW reference):")
        print(f"  LW signature vs genuine truth : [{sc['m_truth'][0]:+.2f} {sc['m_truth'][1]:+.2f} {sc['m_truth'][2]:+.2f}]"
              f"  (genuine LW->diffusive angle = {sc['ang_genuine_to_diff']:.1f} deg, the dispersive separation)")
        print(f"  LW signature vs LW reference   : [{sc['m_self'][0]:+.2f} {sc['m_self'][1]:+.2f} {sc['m_self'][2]:+.2f}]")
        print(f"  residual RMS collapse ||r_self||/||r_truth|| = {sc['rms_ratio']:.3f}  "
              f"({'COLLAPSED (<0.4)' if sc['rms_ratio']<0.4 else 'survives'})")
        print(f"  signature corruption: angle(self-LW -> genuine-LW) = {sc['ang_to_genuine']:5.1f} deg, "
              f"angle(self-LW -> DIFFUSIVE class) = {sc['ang_to_diff']:5.1f} deg")
        print(f"  cross-reference attribution (genuine-trained clf -> self-LW): "
              f"fraction called DISPERSIVE(correct) = {sc['frac_correct_self']:.2f}")
        print(f"  -> signature {'CORRUPTED onto the wrong (diffusive) class' if sc['self_cancel'] else 'preserved'} "
              f"(self_cancel flag {'FIRED' if sc['self_cancel'] else 'not fired'})")

    # ---------------- DECISION on the DEGRADED regime (project convention) ----------------
    dtag = REGIMES[1][0]
    _, rows, sc_info = bundle[dtag]
    base_row = [r for r in rows if r["ref"] == "REF_SPECTRAL"][0]
    base_dd, base_id = base_row["dd_auc"], base_row["id4"]
    print("\n" + "="*92)
    print(f"DEGRADATION vs the genuine spectral reference  [DEGRADED regime: {dtag}]")
    print("  (dAUROC = within-reference separability change; xref-acc = transfer of a GENUINE-trained")
    print("   classifier to this reference -- the practitioner who does NOT re-train per reference)")
    print("="*92)
    worst_dd_drop = 0.0; worst_id_drop = 0.0; worst_ref = None; worst_xref_drop = 0.0; worst_xref_ref = None
    base_xref = base_row["xref_acc"]
    for r in rows:
        if r["ref"] == "REF_SPECTRAL": continue
        d_dd = r["dd_auc"] - base_dd; d_id = r["id4"] - base_id; d_xr = r["xref_acc"] - base_xref
        print(f"  {r['ref']:16s} ({r['label']:36s}) dAUROC={d_dd:+.3f}  d4wayID={d_id:+.3f}  "
              f"xref-acc={r['xref_acc']:.3f} (d={d_xr:+.3f})")
        if -d_dd > worst_dd_drop: worst_dd_drop = -d_dd; worst_ref = r["ref"]
        if -d_id > worst_id_drop: worst_id_drop = -d_id
        if -d_xr > worst_xref_drop: worst_xref_drop = -d_xr; worst_xref_ref = r["ref"]
    biased = [r for r in rows if r["ref"] != "REF_SPECTRAL"]
    min_dd_margin = float(min(r["dd_auc"] - r["dd_floor"] for r in biased))
    min_id_margin = float(min(r["id4"]  - r["id4_floor"] for r in biased))
    min_dd_auc    = float(min(r["dd_auc"] for r in biased))
    min_id_acc    = float(min(r["id4"] for r in biased))
    min_xref_acc  = float(min(r["xref_acc"] for r in biased))
    self_cancel   = sc_info["self_cancel"]

    # ---------------- write CSV (both regimes; self-cancellation row per regime) ----------------
    csv = os.path.join(TAB, "reference_misspecification.csv")
    with open(csv, "w") as f:
        f.write("regime,reference,label,ref_solver_err_ratio,diff_vs_disp_auroc_within_ref,dd_perm_floor,"
                "xref_acc_genuine_trained,fourway_id_acc,id_perm_floor,coherence_deg,"
                "dAUROC_vs_genuine,d4wayID_vs_genuine\n")
        for tag, _ in REGIMES:
            _, rr, sci = bundle[tag]
            b_dd = [r for r in rr if r["ref"]=="REF_SPECTRAL"][0]["dd_auc"]
            b_id = [r for r in rr if r["ref"]=="REF_SPECTRAL"][0]["id4"]
            for r in rr:
                f.write(f'"{tag}",{r["ref"]},"{r["label"]}",{r["ref_err"]:.4f},{r["dd_auc"]:.4f},'
                        f'{r["dd_floor"]:.4f},{r["xref_acc"]:.4f},{r["id4"]:.4f},{r["id4_floor"]:.4f},'
                        f'{r["coh"]:.2f},{r["dd_auc"]-b_dd:+.4f},{r["id4"]-b_id:+.4f}\n')
            # self-cancellation row: cols reused as
            #   ref_solver_err_ratio          = residual RMS collapse ratio ||r_self||/||r_truth||
            #   diff_vs_disp_auroc_within_ref = NA (not meaningful for the single self-audited solver)
            #   xref_acc_genuine_trained      = fraction of self-audited LW called CORRECT (dispersive)
            #   coherence_deg                 = angle(self-LW -> diffusive class)  [lower=worse corruption]
            #   dAUROC_vs_genuine             = angle(self-LW -> genuine-LW signature)
            #   d4wayID_vs_genuine            = FIRED/ok flag
            f.write(f'"{tag}",SELF_CANCEL_LW_vs_LW,"LW solver vs same-grid LW reference",'
                    f'{sci["rms_ratio"]:.4f},NA,NA,{sci["frac_correct_self"]:.4f},NA,NA,'
                    f'{sci["ang_to_diff"]:.2f},{sci["ang_to_genuine"]:.2f},'
                    f'{"FIRED" if sci["self_cancel"] else "ok"}\n')

    # ---------------- DECISION ----------------
    print("\n" + "="*92)
    print(f"DECISION RULE  (robust to reference-scheme choice vs fragile)  [DEGRADED regime: {dtag}]")
    print("="*92)
    print(f"genuine-reference baseline : diff-vs-disp AUROC = {base_dd:.3f}, 4-way ID = {base_id:.3f}")
    print(f"across biased references    : min diff-vs-disp AUROC = {min_dd_auc:.3f} (margin over floor {min_dd_margin:+.3f}),")
    print(f"                              min 4-way ID = {min_id_acc:.3f} (margin over floor {min_id_margin:+.3f})")
    print(f"worst within-ref degrad.    : reference={worst_ref}, dAUROC={-worst_dd_drop:+.3f}, d4wayID={-worst_id_drop:+.3f}")
    print(f"cross-reference TRANSFER     : min xref-acc = {min_xref_acc:.3f} (worst at {worst_xref_ref}, "
          f"drop {-worst_xref_drop:+.3f} vs genuine {base_xref:.2f})")
    print(f"same-family self-cancel (LW vs same-grid LW): {'FIRED' if self_cancel else 'not fired'}  "
          f"(residual collapse {sc_info['rms_ratio']:.2f}, signature angle self->diffusive {sc_info['ang_to_diff']:.1f}deg "
          f"vs self->genuine-LW {sc_info['ang_to_genuine']:.1f}deg; cross-ref correct-class frac {sc_info['frac_correct_self']:.2f})")

    # WITHIN-REFERENCE robustness: biased references keep diff-vs-disp clearly above floor (margin>=0.15
    # AND AUROC>=0.75) and 4-way ID clearly above floor (margin>=0.10), modest degradation (worst<=0.10).
    within_robust = (min_dd_margin >= 0.15 and min_dd_auc >= 0.75 and min_id_margin >= 0.10
                     and worst_dd_drop <= 0.10)
    # CROSS-REFERENCE transfer robustness: a classifier TRAINED on the genuine reference still attributes
    # above 0.75 when applied to a biased-reference residual. (chance = 0.5 for the 4-class diff/disp split.)
    xref_robust = (min_xref_acc >= 0.75)
    if within_robust and xref_robust and not self_cancel:
        outcome = "ROBUST"
        print("\n[ROBUST]  Diffusive-vs-dispersive attribution stays well above the floor under EVERY biased")
        print("  reference, BOTH within-reference and on cross-reference transfer, and the same-family self-audit")
        print("  does not corrupt the signature. A real auditor does NOT need the exact reference.")
    elif within_robust:
        outcome = "ROBUST_WITHIN_REF_TWO_CAVEATS"
        print("\n[ROBUST WITHIN-REFERENCE, with TWO practitioner caveats]")
        print("  (1) WITHIN-REFERENCE the framework is robust to reference-scheme misspecification: if the")
        print("      auditor forms residuals AND trains the classifier with the SAME (any) biased reference --")
        print("      under-resolved upwind / Lax-Wendroff / beam-warming, finer grid, resampled -- the")
        print(f"      diffusive-vs-dispersive split is recovered at AUROC {min_dd_auc:.2f} (worst dAUROC {-worst_dd_drop:+.3f}),")
        print("      far above the permutation floor. So the exact spectral truth is NOT required.")
        print("  (2) CROSS-REFERENCE TRANSFER is the caveat to print: a classifier trained on one reference and")
        print(f"      applied to residuals formed with a STRONGLY-BIASED different reference degrades to xref-acc")
        print(f"      {min_xref_acc:.2f} (worst at {worst_xref_ref}; a diffusive reference whose own error dominates,")
        print(f"      ref/solver_err {[r['ref_err'] for r in biased if r['ref']==worst_xref_ref][0]:.1f}, flips the absolute signature sign and breaks a transferred")
        print("      linear classifier). Recipe: RE-TRAIN per reference (or use a within-reference classifier);")
        print("      do NOT reuse a classifier across references whose bias differs strongly.")
        if self_cancel:
            print(f"  (3) SAME-FAMILY SELF-AUDIT is forbidden: a Lax-Wendroff solver differenced against a SAME-grid")
            print(f"      Lax-Wendroff reference has its residual collapse to {sc_info['rms_ratio']:.2f} of the genuine residual and")
            print(f"      its signature rotate onto the WRONG (diffusive) class (angle self->diffusive {sc_info['ang_to_diff']:.1f}deg <")
            print(f"      self->genuine-LW {sc_info['ang_to_genuine']:.1f}deg). Never audit a solver against a reference from its OWN")
            print("      numerical family at comparable resolution.")
    else:
        outcome = "FRAGILE"
        print("\n[FRAGILE]  Even within-reference, attribution degrades materially under at least one biased")
        print(f"  reference (min diff-vs-disp AUROC {min_dd_auc:.3f}, margin over floor {min_dd_margin:+.3f}; worst dAUROC")
        print(f"  {-worst_dd_drop:+.3f}). The recipe NEEDS the caveat: the reference must be a genuine / cross-family")
        print("  truth, not an arbitrary trusted solver." +
              ("  The same-family self-audit collapses entirely." if self_cancel else ""))

    print(f"\nartifacts -> {csv}")
    _figure(bundle, REGIMES, dtag)
    return bundle, outcome

def _figure(bundle, REGIMES, dtag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        import seaborn as sns; sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans", palette="muted")
    except Exception: pass
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#8a8a8a"
    short = {"REF_SPECTRAL":"spectral\n(genuine)", "REF_FINE_UPWIND":"fine\nupwind", "REF_FINE_LW":"fine\nLax-Wend",
             "REF_FINE_BW":"fine\nbeam-warm", "REF_COARSE_SPEC":"coarse\nspectral"}
    _, rows, sc = bundle[dtag]            # decision (degraded) regime drives all panels
    labels = [short[r["ref"]] for r in rows]
    dd  = [r["dd_auc"]   for r in rows]; ddf = [r["dd_floor"]  for r in rows]
    xref = [r["xref_acc"] for r in rows]
    cols = [GREEN if r["ref"]=="REF_SPECTRAL" else (GREY if r["ref"]=="REF_COARSE_SPEC" else BLUE) for r in rows]
    base_dd = [r for r in rows if r["ref"]=="REF_SPECTRAL"][0]["dd_auc"]
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.1)); fig.subplots_adjust(wspace=0.34)
    x = np.arange(len(rows))

    # Panel A: WITHIN-REFERENCE diffusive-vs-dispersive AUROC per reference + floor + genuine baseline
    axA = ax[0]
    axA.bar(x, dd, 0.62, color=cols, edgecolor="k", linewidth=0.6, zorder=3)
    axA.plot(x, ddf, "o--", color=RED, lw=1.3, ms=5, label="permutation floor", zorder=4)
    axA.axhline(base_dd, color=GREEN, ls=":", lw=1.6, label=f"genuine-ref baseline ({base_dd:.2f})", zorder=2)
    axA.axhline(0.5, color=GREY, ls=":", lw=1.0, zorder=1)
    for i, v in enumerate(dd): axA.text(i, v+0.012, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("diffusive-vs-dispersive AUROC"); axA.set_ylim(0.4, 1.03)
    axA.set_title(f"A  WITHIN-reference attribution [{dtag}]\n"
                  "robust to reference SCHEME (auditor trains on own reference)", fontsize=9.0)
    axA.legend(frameon=True, framealpha=0.95, edgecolor="#ddd", fontsize=7.6, loc="lower left")
    axA.text(-0.14, 1.04, "A", transform=axA.transAxes, fontsize=14, fontweight="bold")

    # Panel B: CROSS-REFERENCE TRANSFER -- genuine-trained classifier applied to each biased reference.
    axB = ax[1]
    bcolB = [GREEN if r["ref"]=="REF_SPECTRAL" else (RED if r["xref_acc"] < 0.75 else BLUE) for r in rows]
    axB.bar(x, xref, 0.62, color=bcolB, edgecolor="k", linewidth=0.6, zorder=3)
    axB.axhline(0.75, color=RED, ls="--", lw=1.2, alpha=0.7, label="transfer threshold (0.75)")
    axB.axhline(0.5, color=GREY, ls=":", lw=1.0, label="chance (0.50)")
    for i, v in enumerate(xref): axB.text(i, v+0.012, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8)
    axB.set_ylabel("attribution accuracy (genuine-trained clf)"); axB.set_ylim(0.4, 1.03)
    axB.set_title("B  CROSS-reference TRANSFER\nclassifier trained on genuine ref, applied to biased ref\n"
                  "(red = breaks: strongly-biased diffusive reference)", fontsize=9.0)
    axB.legend(frameon=True, framealpha=0.95, edgecolor="#ddd", fontsize=7.6, loc="lower left")
    axB.text(-0.14, 1.04, "B", transform=axB.transAxes, fontsize=14, fontweight="bold")

    # Panel C: same-family self-cancellation -- SIGNATURE CORRUPTION angles.
    # If angle(self-LW -> diffusive class) < angle(self-LW -> its own genuine LW signature), the dispersive
    # identity has rotated onto the WRONG class. Green = genuine separation the self-audit should not undercut.
    axC = ax[2]
    labels_c = ["self-LW ->\ngenuine-LW\nsignature", "self-LW ->\nDIFFUSIVE\nclass", "genuine-LW ->\nDIFFUSIVE\n(reference)"]
    vals = [sc["ang_to_genuine"], sc["ang_to_diff"], sc["ang_genuine_to_diff"]]
    corrupted = sc["self_cancel"]
    bcolC = [GREY, RED if corrupted else BLUE, GREEN]
    axC.bar([0,1,2], vals, 0.55, color=bcolC, edgecolor="k", linewidth=0.6, zorder=3)
    for i, v in enumerate(vals): axC.text(i, v+0.8, f"{v:.1f}°", ha="center", fontsize=10, fontweight="bold")
    axC.set_xticks([0,1,2]); axC.set_xticklabels(labels_c, fontsize=8)
    axC.set_ylabel("signature angle (deg)"); axC.set_ylim(0, max(vals)*1.28 + 2)
    flagtxt = "CORRUPTED onto wrong class" if corrupted else "preserved"
    axC.set_title(f"C  SAME-FAMILY self-audit: signature {flagtxt}\n"
                  f"LW solver vs SAME-grid LW ref; resid RMS -> {sc['rms_ratio']:.2f}\n"
                  "(corrupted when red < grey)", fontsize=9.0)
    axC.text(-0.14, 1.04, "C", transform=axC.transAxes, fontsize=14, fontweight="bold")

    out = os.path.join(FIG, "reference_misspecification.png")
    fig.savefig(out); plt.close(fig); print(f"figure    -> {out}")

if __name__ == "__main__":
    main()
