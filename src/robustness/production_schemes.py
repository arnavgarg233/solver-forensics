"""
Production-scheme robustness test.
==================================
Does the attribution framework survive production-grade numerical schemes whose
truncation error is solution-dependent, not a clean fixed modified-equation term?

Scheme zoo (linear advection u_t + a u_x = 0, a=1, periodic), classic + production:
  classic (fixed stencil):  upwind, lax_friedrichs, lax_wendroff, beam_warming
  TVD flux limiters:        minmod, superbee, van_leer        (NONLINEAR: stencil adapts to local smoothness)
  high order / implicit:    weno5 (5th-order WENO + SSP-RK3), maccormack, crank_nicolson (implicit)
Each production scheme was independently implemented and validated (stable, TVD where
applicable, observed convergence order) by a separate agent.

Pre-registered decision (degraded = 1% field noise):
  GO    : production schemes attribute above controls (9-way scheme-ID >> chance, the three
          subtly-different LIMITERS discriminate above chance, NC1 at chance) -> framework
          generalizes to production schemes.
  KILL/SCOPE : the limiters' solution-dependent truncation makes their signatures IC-scattered
          (low per-scheme coherence, limiter discrimination near chance) -> measured limit:
          works on fixed-stencil schemes, degrades on adaptive/nonlinear ones.
Bug-fixed Fourier resample throughout; numpy-2-safe.
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
L, A, NU, T = 1.0, 1.0, 0.8, 0.30
N_C, N_C2, N_BASE, COMMON, N_IC = 128, 192, 384, 64, 50

# ---- classic fixed-stencil schemes ----
def upwind(u, nu):         return u - nu*(u - np.roll(u,1))
def lax_friedrichs(u, nu): return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))
def lax_wendroff(u, nu):   return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))
def beam_warming(u, nu):   return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))

# ---- production schemes (agent-validated; verbatim) ----
def minmod(u, nu):
    up = np.roll(u,-1); dn = np.roll(u,1); denom = up - u
    r = (u - dn)/(denom + 1e-12); phi = np.maximum(0.0, np.minimum(1.0, r))
    F = u + 0.5*(1.0-nu)*phi*denom
    return u - nu*(F - np.roll(F,1))
def superbee(u, nu):
    den = np.roll(u,-1) - u; num = u - np.roll(u,1)
    r = num/np.where(np.abs(den) < 1e-12, np.where(den >= 0, 1e-12, -1e-12), den)
    phi = np.maximum.reduce([np.zeros_like(r), np.minimum(2.0*r, 1.0), np.minimum(r, 2.0)])
    flux = nu*u + 0.5*nu*(1.0-nu)*phi*den
    return u - (flux - np.roll(flux,1))
def van_leer(u, nu):
    du = np.roll(u,-1) - u; num = u - np.roll(u,1)
    r = num/np.where(np.abs(du) < 1e-12, np.where(du >= 0, 1e-12, -1e-12), du)
    phi = (r + np.abs(r))/(1.0 + np.abs(r))
    F = u + 0.5*phi*(1.0-nu)*du
    return u - nu*(F - np.roll(F,1))
def weno5(u, nu):
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
def maccormack(u, nu):
    ustar = u - nu*(np.roll(u,-1) - u)
    return 0.5*(u + ustar - nu*(ustar - np.roll(ustar,1)))
def crank_nicolson(u, nu):
    N = u.shape[0]; k = np.fft.fftfreq(N, d=1.0/N); s = np.sin(2*np.pi*k/N)
    g = (1.0 - 1j*(nu/2)*s)/(1.0 + 1j*(nu/2)*s)
    return np.real(np.fft.ifft(np.fft.fft(u)*g))

SCHEMES = {"upwind":upwind, "lax_friedrichs":lax_friedrichs, "lax_wendroff":lax_wendroff,
           "beam_warming":beam_warming, "minmod":minmod, "superbee":superbee, "van_leer":van_leer,
           "weno5":weno5, "maccormack":maccormack, "crank_nicolson":crank_nicolson}
names = list(SCHEMES); LIMITERS = ["minmod", "superbee", "van_leer"]; CLASSIC = ["upwind","lax_friedrichs","lax_wendroff","beam_warming"]

def exact(u0, t, N):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N); return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*A*t), n=N)
def random_ic(N, rng):                               # modes + a bump so the limiters engage near steep features
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(5): u += rng.normal()*np.sin(2*np.pi*rng.integers(1,7)*x/L + rng.uniform(0,2*np.pi))
    x0, w = rng.uniform(0,L), 0.03; u += 1.5*rng.normal()*np.exp(-(((x-x0+L/2)%L-L/2)**2)/(2*w*w))
    return u/(np.std(u)+1e-9)
def antialias(u, M):
    N = len(u)
    if N == M: return u
    return np.fft.irfft(np.fft.rfft(u)[:M//2+1], n=M)*(M/N)
def run(scheme, N, u0):
    dx = L/N; dt = NU*dx/A; ns = int(round(T/dt)); u = u0.copy()
    for _ in range(ns): u = SCHEMES[scheme](u, NU)
    return u, exact(u0, ns*dt, N)
def coeffs(U, R):
    h = L/U.shape[1]
    Am = np.stack([(np.roll(U,-1,1)-2*U+np.roll(U,1,1))/h**2,
                   (np.roll(U,-2,1)-2*np.roll(U,-1,1)+2*np.roll(U,1,1)-np.roll(U,2,1))/(2*h**3),
                   (np.roll(U,-2,1)-4*np.roll(U,-1,1)+6*U-4*np.roll(U,1,1)+np.roll(U,2,1))/h**4], 2)
    AtA = np.einsum('mni,mnk->mik', Am, Am) + 1e-9*np.eye(3)
    return np.linalg.solve(AtA, np.einsum('mni,mn->mi', Am, R)[..., None])[..., 0]
def direction(C): return np.nan_to_num(C/(np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))

def feats(scheme, N, u0s, noise, seed):
    gn = np.random.default_rng(seed); U, R = [], []
    for u0 in u0s:
        u0N = antialias(u0, N); un, ex = run(scheme, N, u0N)
        if noise > 0: un = un + noise*np.sqrt(np.mean(ex**2))*gn.standard_normal(N)
        U.append(antialias(un, COMMON)); R.append(antialias(un - ex, COMMON))
    return direction(coeffs(np.array(U), np.array(R)))
def coherence(F):
    m = F.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(np.median(np.degrees(np.arccos(np.clip(np.abs(F @ m), 0, 1)))))

CLF = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def acc(Fs, labs, g): return cross_val_score(CLF(), np.vstack(Fs), np.concatenate(labs), groups=np.concatenate(g), cv=GroupKFold(5)).mean()
def auroc(Fa, Fb, ga, gb):
    return cross_val_score(CLF(), np.vstack([Fa,Fb]), np.r_[np.zeros(len(Fa)),np.ones(len(Fb))], groups=np.r_[ga,gb], cv=GroupKFold(5), scoring="roc_auc").mean()

# ================================================================ RUN
print(f"production-scheme robustness | {len(names)} schemes ({len(LIMITERS)} nonlinear limiters), {N_IC} ICs, advection\n")
rng = np.random.default_rng(0); bases = [random_ic(N_BASE, rng) for _ in range(N_IC)]; ic = np.arange(N_IC); h = N_IC//2

# stability self-check at the solver grid
print("stability self-check (max|u| at T, N=128, nu=0.8):")
for sc in names:
    uf, _ = run(sc, N_C, antialias(bases[0], N_C)); print(f"   {sc:16s} {np.max(np.abs(uf)):.2f}  {'OK' if np.isfinite(uf).all() and np.max(np.abs(uf))<5 else 'UNSTABLE'}")

results = {}
for noise in (0.0, 0.01):
    F = {sc: feats(sc, N_C, bases, noise, 1) for sc in names}
    lab = {sc: i for i, sc in enumerate(names)}
    id9 = acc([F[sc] for sc in names], [np.full(N_IC, lab[sc]) for sc in names], [ic]*len(names))
    lim3 = acc([F[sc] for sc in LIMITERS], [np.full(N_IC, i) for i in range(3)], [ic]*3)
    nc1 = auroc(F["minmod"][:h], F["minmod"][h:], ic[:h], ic[h:])
    F2_mm = feats("minmod", N_C2, bases, noise, 2); nc2 = auroc(F["minmod"], F2_mm, ic, ic)
    # permutation floor for the 9-way
    rngp = np.random.default_rng(9); Xall = np.vstack([F[sc] for sc in names]); yall = np.concatenate([np.full(N_IC, lab[sc]) for sc in names]); gall = np.concatenate([ic]*len(names))
    floor = np.median([cross_val_score(CLF(), Xall, rngp.permutation(yall), groups=gall, cv=GroupKFold(5)).mean() for _ in range(15)])
    results[noise] = dict(id9=id9, lim3=lim3, nc1=nc1, nc2=nc2, floor=floor,
                          coh_lim=np.median([coherence(F[sc]) for sc in LIMITERS]),
                          coh_cls=np.median([coherence(F[sc]) for sc in CLASSIC]))

print("\nper-scheme signature coherence (median angle of per-IC direction from scheme mean; lower=coherent), clean:")
Fc = {sc: feats(sc, N_C, bases, 0.0, 1) for sc in names}
for sc in names: print(f"   {sc:16s} {coherence(Fc[sc]):5.1f} deg" + ("   [limiter]" if sc in LIMITERS else ""))

print(f"\n{'noise':>6} {'9way-acc':>9s} {'(floor)':>8s} {'limiter-3way':>13s} {'NC1':>6s} {'NC2':>6s} {'coh:lim/cls':>13s}")
for noise in (0.0, 0.01):
    r = results[noise]
    print(f"{noise*100:>5.0f}% {r['id9']:>9.3f} {r['floor']:>8.3f} {r['lim3']:>13.3f} {r['nc1']:>6.3f} {r['nc2']:>6.3f} {r['coh_lim']:>6.1f}/{r['coh_cls']:<6.1f}")

with open(os.path.join(TAB, "production_schemes_results.csv"), "w") as f:
    f.write("noise,id9way,floor,limiter3way,nc1,nc2,coherence_limiter,coherence_classic\n")
    for noise in (0.0, 0.01):
        r = results[noise]; f.write(f"{noise},{r['id9']:.4f},{r['floor']:.4f},{r['lim3']:.4f},{r['nc1']:.4f},{r['nc2']:.4f},{r['coh_lim']:.2f},{r['coh_cls']:.2f}\n")

print("\n" + "="*72 + "\nPRE-REGISTERED DECISION (degraded = 1% noise)\n" + "="*72)
r = results[0.01]
print(f"9-way ID = {r['id9']:.3f} (chance {1/len(names):.2f}, floor {r['floor']:.2f})  limiter-3way = {r['lim3']:.3f} "
      f"(chance 0.33)  NC1 = {r['nc1']:.3f}  coherence limiter {r['coh_lim']:.0f} deg vs classic {r['coh_cls']:.0f} deg")
if r['id9'] > r['floor'] + 0.20 and r['lim3'] > 0.50 and r['nc1'] <= 0.65:
    print("\n[GO]  production schemes attribute well above the floor and the three subtly-different limiters")
    print("  discriminate above chance. The framework generalizes to production-grade (limiter / WENO / implicit)")
    print(f"  schemes. (Limiters do carry more IC-scatter: coherence {r['coh_lim']:.0f} deg vs classic {r['coh_cls']:.0f} deg,")
    print("   reported as the mechanism, but their mean signature is still attributable.)")
elif r['lim3'] <= 0.45 or r['id9'] <= r['floor'] + 0.10:
    print("\n[SCOPE/KILL]  the limiters' solution-dependent truncation scatters their signatures: limiter")
    print(f"  discrimination {r['lim3']:.2f} near chance and/or 9-way {r['id9']:.2f} near the floor. Measured limit:")
    print("  attribution works on fixed-stencil schemes and degrades on adaptive/nonlinear ones.")
else:
    print("\n[MIXED]  production schemes partly attribute (limiters degraded but above chance). Report as a")
    print("  scoped capability: coarse families attribute, fine limiter discrimination is partial.")
print(f"\nartifacts -> {os.path.join(TAB, 'production_schemes_results.csv')}")
