"""
solver-forensics :: KdV BREADTH GATE
==========================================================
Breadth test of the audit framework on an intrinsically DISPERSIVE PDE.
Pre-registration (approved, inline in chat 2026-06-05) + amendments:
  - NC2 (grid) SCOPES, never KILLs; KILL reserved for NC-phys or reference non-convergence.
  - S4 = a *different* u_xxx discretization; its signature is VALIDATED here, not asserted.
  - field-relative noise = sigma * RMS(u_reference).
  - reference-convergence (N_ref 512->1024) must be small or the run is reference-limited.
  - KdV-aware RESTRICTED library {u_xx, u_xxx, u_xxxx} (u_xxxxx tested in de-risk, c5~0, excluded).
  - "Build, measure honestly": de-risk showed KdV's nonlinear-dispersive dynamics SCRAMBLE the
    clean diffusive-vs-dispersive taxonomy (residuals are c2-dominated; schemes separate by
    c2-sign + c3 content, not into a clean taxonomy). So self-validation REPORTS the actual
    (degraded) signatures; the decision rides on the audit contrast (scheme-change AUROC) +
    NC-phys + controls, NOT on a pre-asserted taxonomy.

KdV: u_t + 6 u u_x + delta u_xxx = 0, periodic. Reference: pseudo-spectral IFRK4 (exact,
de-risked: soliton vs analytic 7e-9, 512->1024 convergence 1.2e-4). Coarse schemes: IFRK4
with the scheme's FD u_xxx symbol (integrating factor removes the dt~dx^3 stiffness AND the
1st-order temporal error that swamped the spatial signature in the IMEX-Euler de-risk).
Observation uses a PROPER Fourier resample (not integer-stride decimation).

Decision: GO if scheme-change (A=S1 centered vs B=S2 LF-flux) AUROC >= 0.85 above controls,
NC-phys low enough that physical dispersion is not masquerading (margin_phys >= 0.20,
NC-phys <= 0.65), reference converges. MIXED/SCOPE if clean/high-res only or NC2 fires.
KILL/SCOPE if NC-phys fires (margin_phys < 0.10) or reference does not converge.

Pure numpy + sklearn, CPU, numpy-2-safe.
"""
import os
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

L = 2*np.pi
N_C, N_C2, N_REF, N_REF_HI = 128, 192, 512, 1024
T, DELTA, DELTA2, AMP, N_IC = 1.0, 1.0, 2.0, 0.5, 60
LIB = (2, 3, 4)                                  # KdV-aware restricted library: u_xx,u_xxx,u_xxxx
D3_2ND = {2:0.5, 1:-1.0, -1:1.0, -2:-0.5}        # 2nd-order centered u_xxx stencil (/dx^3)
D3_4TH = {3:-1/8, 2:1.0, 1:-13/8, -1:13/8, -2:-1.0, -3:1/8}   # 4th-order centered u_xxx
D2 = {1:1.0, 0:-2.0, -1:1.0}                     # u_xx stencil (/dx^2)
SCHEMES = {"S1_centered":(D3_2ND,0.0,"centered"), "S2_LF":(D3_2ND,0.0,"LF"),
           "S3_visc":(D3_2ND,0.05,"centered"), "S4_4th_d3":(D3_4TH,0.0,"centered")}
names = list(SCHEMES); A_SCH, B_SCH = "S1_centered", "S2_LF"   # audit contrast (stabilization change)

def ric(N, seed):                                # smooth random periodic IC (low modes -> resamples exactly)
    r = np.random.default_rng(seed); x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for kk in (1, 2, 3): u += r.normal()*np.sin(2*np.pi*kk*x/L + r.uniform(0, 2*np.pi))
    return AMP*u/(np.max(np.abs(u)) + 1e-9)
def ifrk4(uh, Lhat, Nl, dt, ns):
    E = np.exp(Lhat*dt); E2 = np.exp(Lhat*dt/2)
    for _ in range(ns):
        a = dt*Nl(uh); b = dt*Nl(E2*(uh+a/2)); c = dt*Nl(E2*uh+b/2); d = dt*Nl(E*uh+E2*c)
        uh = E*uh + (E*a + 2*E2*(b+c) + d)/6
    return uh
def sym(st, k, dx): return sum(c*np.exp(1j*k*m*dx) for m, c in st.items())
def n_steps(u0, dx): return max(1, int(np.ceil(T/(0.2*dx/(6*(np.max(np.abs(u0))+1e-9))))))
def spectral_ref(u0, N, delta):                  # exact dispersion symbol i*delta*k^3
    dx = L/N; k = 2*np.pi*np.fft.fftfreq(N, d=dx); m = np.abs(k) <= (2/3)*np.max(np.abs(k))
    ns = n_steps(u0, dx); dt = T/ns
    Nl = lambda uh: -3j*k*(np.fft.fft(np.real(np.fft.ifft(uh))**2)*m)
    return np.real(np.fft.ifft(ifrk4(np.fft.fft(u0), 1j*delta*k**3, Nl, dt, ns)))
def coarse(u0, N, scheme, delta):                # IFRK4 with the scheme's FD u_xxx symbol + flux/diffusion
    d3, eps, flux = SCHEMES[scheme]
    dx = L/N; k = 2*np.pi*np.fft.fftfreq(N, d=dx)
    Lhat = -delta*sym(d3, k, dx)/dx**3 + eps*sym(D2, k, dx)/dx**2
    def Nl(uh):
        u = np.real(np.fft.ifft(uh)); f = 3*u*u
        if flux == "centered": fx = (np.roll(f,-1)-np.roll(f,1))/(2*dx)
        else:                                    # Lax-Friedrichs flux (dissipative)
            a = 6*np.max(np.abs(u))+1e-9; Fp = 0.5*(f+np.roll(f,-1)) - 0.5*a*(np.roll(u,-1)-u)
            fx = (Fp - np.roll(Fp, 1))/dx
        return np.fft.fft(-fx)
    ns = n_steps(u0, dx); dt = T/ns
    return np.real(np.fft.ifft(ifrk4(np.fft.fft(u0), Lhat, Nl, dt, ns)))
def antialias(u, M):                             # PROPER Fourier resample to exactly M (the bug-fixed version)
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M) * (M/N)
def deriv(u, o, h):
    if o == 2: return (np.roll(u,-1,-1)-2*u+np.roll(u,1,-1))/h**2
    if o == 3: return (np.roll(u,-2,-1)-2*np.roll(u,-1,-1)+2*np.roll(u,1,-1)-np.roll(u,2,-1))/(2*h**3)
    return (np.roll(u,-2,-1)-4*np.roll(u,-1,-1)+6*u-4*np.roll(u,1,-1)+np.roll(u,2,-1))/h**4
def coeffs(U, R):                                # dense LSQ on the restricted library; numpy-2-safe
    h = L/U.shape[1]; Am = np.stack([deriv(U, o, h) for o in LIB], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(len(LIB))
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]
def direction(C): return np.nan_to_num(C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))

def sigs(refs, base, scheme, N_c, delta, noise, N_obs, seed):
    """Coefficient-direction features for (scheme, N_c, delta) over the given ICs."""
    gn = np.random.default_rng(seed); U, R = [], []
    for u0_base, ref in zip(base, refs):
        ref_c = antialias(ref, N_c); u0_c = antialias(u0_base, N_c)
        uc = antialias(coarse(u0_c, N_c, scheme, delta), N_c)
        un = uc + noise*np.sqrt(np.mean(ref_c**2))*gn.standard_normal(N_c)
        U.append(antialias(un, N_obs)); R.append(antialias(un - ref_c, N_obs))
    return direction(coeffs(np.array(U), np.array(R)))

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def auroc(Fa, Fb, ga, gb):
    X = np.vstack([Fa, Fb]); y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]; g = np.r_[ga, gb]
    return cross_val_score(CLF(), X, y, groups=g, cv=GroupKFold(5), scoring="roc_auc").mean()
def acc_multi(Fs, labs, grp):
    X = np.vstack(Fs); y = np.concatenate(labs); g = np.concatenate([grp]*len(Fs))
    return cross_val_score(CLF(), X, y, groups=g, cv=GroupKFold(5)).mean()

# ================================================================ RUN
print(f"KdV gate: u_t+6uu_x+{DELTA}u_xxx=0, periodic L=2pi, coarse N={N_C}, ref spectral N={N_REF}, {N_IC} ICs\n")
base = [ric(N_REF, s) for s in range(N_IC)]; ic = np.arange(N_IC)
refs   = [spectral_ref(u, N_REF, DELTA) for u in base]          # delta=1 references
refs_d2= [spectral_ref(u, N_REF, DELTA2) for u in base]         # delta=2 (NC-phys)

# --- reference-convergence control (amendment 4): per-scheme direction drift 512 vs 1024 ---
refs_hi = [spectral_ref(antialias(u, N_REF_HI), N_REF_HI, DELTA) for u in base[:25]]
refs_hi_d2 = [spectral_ref(antialias(u, N_REF_HI), N_REF_HI, DELTA2) for u in base[:25]]
drift = []
for sc in names:
    c512 = sigs(refs[:25], base[:25], sc, N_C, DELTA, 0.0, N_C, 1)
    c1024 = sigs(refs_hi, base[:25], sc, N_C, DELTA, 0.0, N_C, 1)
    drift.append(np.median([np.degrees(np.arccos(np.clip(abs(c512[i]@c1024[i]),0,1))) for i in range(25)]))
ref_drift_d1 = float(np.median(drift))
# delta=2 reference (used by NC-phys) must converge too - check explicitly, not out-of-band
c512_d2 = sigs(refs_d2[:25], base[:25], A_SCH, N_C, DELTA2, 0.0, N_C, 1)
c1024_d2 = sigs(refs_hi_d2, base[:25], A_SCH, N_C, DELTA2, 0.0, N_C, 1)
ref_drift_d2 = float(np.median([np.degrees(np.arccos(np.clip(abs(c512_d2[i]@c1024_d2[i]),0,1))) for i in range(25)]))
ref_drift = max(ref_drift_d1, ref_drift_d2)                     # guard on the worse of the two
print(f"reference-convergence (512->1024): delta=1 per-scheme drift={ref_drift_d1:.1f} deg, "
      f"delta=2 (NC-phys) drift={ref_drift_d2:.1f} deg  "
      f"({'OK converged' if ref_drift < 10 else 'REFERENCE-LIMITED'})\n")

# --- self-validation: REPORT the actual signatures (taxonomy is degraded; do not assert) ---
print("self-validation - mean clean coefficient direction per scheme [c2,c3,c4] (taxonomy REPORTED, not asserted):")
clean_feats = {}
for sc in names:
    F = sigs(refs, base, sc, N_C, DELTA, 0.0, N_C, 1); clean_feats[sc] = F; m = F.mean(0)
    print(f"   {sc:14s} [{m[0]:+.2f} {m[1]:+.2f} {m[2]:+.2f}]")
print("   (de-risk finding: KdV residuals are c2-dominated; schemes separate by c2-sign + c3 content,\n"
      "    NOT a clean diffusive-vs-dispersive split - reported as a measured taxonomy degradation)\n")

# --- attribution + controls across regimes ---
REGIMES = [("clean", 0.0, N_C), ("1% noise", 0.01, N_C), ("degraded(64,1%)", 0.01, 64), ("degraded(64,5%)", 0.05, 64)]
rows = {}
for tag, nz, nobs in REGIMES:
    FA  = sigs(refs, base, A_SCH, N_C,  DELTA,  nz, nobs, 10)
    FB  = sigs(refs, base, B_SCH, N_C,  DELTA,  nz, nobs, 11)
    F4  = sigs(refs, base, "S4_4th_d3", N_C, DELTA, nz, nobs, 12)
    FAp = sigs([antialias(r,N_REF) for r in refs], base, A_SCH, N_C2, DELTA, nz, nobs, 13)  # NC2: S1@192
    FAd = sigs(refs_d2, base, A_SCH, N_C, DELTA2, nz, nobs, 14)                              # NC-phys: S1@delta2
    h = N_IC//2
    rows[tag] = dict(
        scheme = auroc(FA, FB, ic, ic),                          # A=S1 vs B=S2 (the audit contrast)
        s1_s4  = auroc(FA, F4, ic, ic),                          # dispersion-stencil change (fragile)
        nc1    = auroc(FA[:h], FA[h:], ic[:h], ic[h:]),          # IC/noise only
        nc2    = auroc(FA, FAp, ic, ic),                         # grid change (SCOPES)
        ncphys = auroc(FA, FAd, ic, ic),                         # physical-dispersion change (KILL test)
        id4    = acc_multi([FA, FB, sigs(refs,base,"S3_visc",N_C,DELTA,nz,nobs,15), F4],
                           [np.full(N_IC,i) for i in range(4)], ic))
    print(f"  evaluated: {tag}")

# ================================================================ table + csv + decision
print(f"\n{'regime':17s} {'A-vs-B↑':>8s} {'4way-acc':>9s} {'NC1':>6s} {'NC2(grid)':>10s} {'NCphys':>8s} {'S1vS4':>7s}")
for tag,_,_ in REGIMES:
    r = rows[tag]
    print(f"{tag:17s} {r['scheme']:>8.3f} {r['id4']:>9.3f} {r['nc1']:>6.3f} {r['nc2']:>10.3f} {r['ncphys']:>8.3f} {r['s1_s4']:>7.3f}")

with open(os.path.join(TAB, "kdv_breadth_results.csv"), "w") as f:
    f.write("regime,A_vs_B,id4way,NC1,NC2_grid,NC_phys,S1_vs_S4,margin_phys,margin_grid\n")
    for tag,_,_ in REGIMES:
        r = rows[tag]; f.write(f'"{tag}",{r["scheme"]:.4f},{r["id4"]:.4f},{r["nc1"]:.4f},{r["nc2"]:.4f},'
                               f'{r["ncphys"]:.4f},{r["s1_s4"]:.4f},{r["scheme"]-r["ncphys"]:.4f},{r["scheme"]-r["nc2"]:.4f}\n')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = np.arange(len(REGIMES)); w = 0.2; plt.figure(figsize=(11,5))
for i,(key,lab,c) in enumerate([("scheme","A-vs-B scheme↑","C0"),("nc2","NC2 grid","C1"),
                                ("ncphys","NC-phys (KILL test)","C3"),("nc1","NC1 IC/noise","C7")]):
    plt.bar(xs+(i-1.5)*w, [rows[t]["{}".format(key)] for t,_,_ in REGIMES], w, label=lab, color=c)
plt.axhline(0.85,color="C0",ls="--",alpha=.5); plt.axhline(0.65,color="C3",ls="--",alpha=.5); plt.axhline(0.5,color="grey",ls=":")
plt.xticks(xs,[t for t,_,_ in REGIMES],fontsize=8,rotation=12); plt.ylim(0.3,1.03); plt.ylabel("AUROC")
plt.title("KdV gate: scheme attribution vs grid (NC2) and physical-dispersion (NC-phys) confounds")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG,"kdv_breadth_result.png"),dpi=130)

print("\n" + "="*74 + "\nPRE-REGISTERED DECISION (degraded = coarse 64 + 1% noise)\n" + "="*74)
d = rows["degraded(64,1%)"]; scheme, nc1, nc2, ncphys = d["scheme"], d["nc1"], d["nc2"], d["ncphys"]
mphys, mgrid = scheme-ncphys, scheme-nc2
print(f"A-vs-B(scheme)={scheme:.3f}  NC1={nc1:.3f}  NC2(grid)={nc2:.3f}  NC-phys={ncphys:.3f}  "
      f"margin_phys={mphys:+.3f}  margin_grid={mgrid:+.3f}  ref_drift={ref_drift:.0f}deg")
ref_ok = ref_drift < 10
if not ref_ok:
    print("\n[KILL/REFERENCE-LIMITED]  reference does not converge -> measuring reference error, not the solver.")
elif mphys < 0.10:
    print("\n[KILL/SCOPE]  NC-phys fires as hard as a scheme change (margin_phys<0.10): physical KdV dispersion")
    print("  MASQUERADES as a solver signature -> measured LIMIT of solver forensics in dispersive PDEs.")
elif scheme >= 0.85 and nc1 <= 0.60 and ncphys <= 0.65 and mphys >= 0.20:
    scoped = mgrid < 0.20 or nc2 > 0.65
    print(f"\n[GO{' (controlled-resolution-only)' if scoped else ''}]  KdV scheme attribution survives, and the")
    print(f"  solver signature is NOT masquerading physical dispersion (NC-phys={ncphys:.2f}, margin_phys={mphys:+.2f}).")
    print("  The framework generalizes to a dispersive PDE." + (" NC2 grid-confound fires -> scope to controlled"
          " resolution (consistent with R5)." if scoped else ""))
    print("  NOTE (measure-honestly): the diffusive-vs-dispersive TAXONOMY degraded on KdV (self-validation");
    print("  above) - schemes attribute but the clean physical-interpretability is a measured KdV limit.")
else:
    print(f"\n[MIXED/SCOPE]  attribution works but not above all controls at degraded (scheme={scheme:.2f}, "
          f"NC1={nc1:.2f}, NC-phys={ncphys:.2f}).")
    print("  Report as conditional - works in clean/high-res only, or controlled-resolution-only. R5/spine unchanged.")
print(f"\nartifacts -> {os.path.join(TAB,'kdv_breadth_results.csv')}")
