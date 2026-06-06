"""
solver-forensics :: WEEK-1 GATE  (HARDENED RE-RUN)
=======================================================================
This is the REAL kill-switch. The earlier 0.99 came from an unpaired,
unnormalized setup with a plain KFold split, so it could have been reading
per-IC shape or raw error magnitude rather than a scheme fingerprint.

Six changes make the re-run the actual gate:

 1. PAIRED ICs        : every initial condition is run through ALL four
                        schemes; each residual carries its IC index.
 2. GROUP SPLIT       : GroupKFold on the IC index is the ONLY CV split, so
                        no initial condition appears in both train and test.
 3. NORMALIZATION ABL.: the whole evaluation runs twice - raw residuals, and
                        L2-normalized residuals - to separate spectral SHAPE
                        from error MAGNITUDE.
 4. OBSERVATION MODEL : every downsample is done two ways - raw point
                        decimation, and anti-aliased (low-pass then decimate).
                        If raw beats anti-aliased, aliasing is folding
                        high-wavenumber info back into the observable band.
 5. NOISE SWEEP       : Gaussian noise at {0, 1%, 5%} of each residual's RMS
                        is added BEFORE observation.
 6. SEPARATE CLAIMS   : diffusive-vs-dispersive (load-bearing), LW-vs-BW
                        (expected to decay = Nyquist result), and 4-way ID
                        are tracked independently at every setting.

Pure CPU / numpy + sklearn. Runs on an M2 Pro in a couple of minutes. No GPU.
No numbers are hardcoded; the run produces them.
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings; warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "results", "figures"); TAB = os.path.join(_ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ----------------------------------------------------------------------
# Physics / numerics config
# ----------------------------------------------------------------------
L, a, nu, T, N = 1.0, 1.0, 0.8, 0.30, 256   # domain, speed, CFL, time, grid
ic_rng = np.random.default_rng(0)           # initial-condition randomness (fixed)

# ---- EXACT solution of linear advection u_t + a u_x = 0 (periodic) ----
def exact_advection(u0, t):
    k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
    return np.fft.irfft(np.fft.rfft(u0)*np.exp(-1j*k*a*t), n=N)

# ---- classic schemes (one step each), CFL nu = a*dt/dx ----
def upwind(u):          return u - nu*(u - np.roll(u,1))                       # 1st order, diffusive
def lax_friedrichs(u):  return 0.5*(np.roll(u,-1)+np.roll(u,1)) - 0.5*nu*(np.roll(u,-1)-np.roll(u,1))  # 1st, very diffusive
def lax_wendroff(u):    return u - 0.5*nu*(np.roll(u,-1)-np.roll(u,1)) + 0.5*nu*nu*(np.roll(u,-1)-2*u+np.roll(u,1))  # 2nd, dispersive (behind)
def beam_warming(u):    return u - 0.5*nu*(3*u-4*np.roll(u,1)+np.roll(u,2)) + 0.5*nu*nu*(u-2*np.roll(u,1)+np.roll(u,2))  # 2nd, dispersive (ahead)

SCHEMES = {"upwind":upwind, "lax_friedrichs":lax_friedrichs,
           "lax_wendroff":lax_wendroff, "beam_warming":beam_warming}
names = list(SCHEMES)
DIFFUSIVE  = {"upwind","lax_friedrichs"}      # 1st-order group
DISPERSIVE = {"lax_wendroff","beam_warming"}  # 2nd-order group

def random_ic(n_modes=6):
    x = np.linspace(0, L, N, endpoint=False); u = np.zeros(N)
    for _ in range(n_modes):
        kk = ic_rng.integers(1,8)
        u += ic_rng.normal()*np.sin(2*np.pi*kk*x/L + ic_rng.uniform(0,2*np.pi))
    if ic_rng.random() < 0.7:                   # a sharp bump strengthens the fingerprint
        x0, w = ic_rng.uniform(0,L), ic_rng.uniform(L*0.02, L*0.08)
        u += ic_rng.normal()*np.exp(-(((x-x0+L/2)%L - L/2)**2)/(2*w*w))
    return u

# ----------------------------------------------------------------------
# (1) PAIRED data generation: one IC -> residual for ALL four schemes
# ----------------------------------------------------------------------
dx = L/N; dt = nu*dx/abs(a); nsteps = int(round(T/dt))
N_IC = 250
print(f"grid N={N}, steps={nsteps}, CFL nu={nu}, T={T}")
print(f"PAIRED design: {N_IC} ICs x {len(names)} schemes = {N_IC*len(names)} residuals")
print("split: GroupKFold by IC index (no IC in both train and test)\n")

resids, labels, groups = [], [], []
for ic_idx in range(N_IC):
    u0 = random_ic()
    ex = exact_advection(u0, nsteps*dt)
    for lab, name in enumerate(names):
        u = u0.copy()
        for _ in range(nsteps): u = SCHEMES[name](u)
        resids.append(u - ex)          # discretization-error fingerprint, same IC across schemes
        labels.append(lab)
        groups.append(ic_idx)
resids = np.array(resids); labels = np.array(labels); groups = np.array(groups)

# per-residual RMS, fixed on the CLEAN residual (noise is relative to this)
rms = np.sqrt(np.mean(resids**2, axis=1, keepdims=True))

# ----------------------------------------------------------------------
# Observation models, noise, featurization
# ----------------------------------------------------------------------
def antialias_decimate(R, ds):
    """Ideal low-pass (keep modes below the decimated grid's Nyquist) then decimate."""
    if ds == 1: return R
    F = np.fft.rfft(R, axis=1)
    kcut = N // (2*ds)                  # new Nyquist index after decimating by ds
    F[:, kcut+1:] = 0.0
    Rlp = np.fft.irfft(F, n=N, axis=1)
    return Rlp[:, ::ds]

def observe(R, obs_model, ds):
    if ds == 1:            return R
    if obs_model == "raw": return R[:, ::ds]          # (4a) aliased point decimation
    return antialias_decimate(R, ds)                  # (4b) anti-aliased decimation

def add_noise(R, level, seed):
    """(5) Gaussian noise at `level` * per-residual RMS, added before observation."""
    if level == 0.0: return R
    g = np.random.default_rng(seed)
    return R + level * rms * g.standard_normal(R.shape)

def featurize(obs, normalize):
    """Shift-invariant features = magnitude spectrum. (3) optionally L2-normalize first."""
    if normalize:
        obs = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-12)
    return np.abs(np.fft.rfft(obs, axis=1))

# ----------------------------------------------------------------------
# (2) Evaluation = GroupKFold-by-IC accuracy for a given distinction
# ----------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
def make_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def evaluate(X, dist):
    if dist == "diff_disp":                                   # load-bearing: 1st vs 2nd order
        y = np.array([0 if names[l] in DIFFUSIVE else 1 for l in labels])
        Xd, yd, gd = X, y, groups
    elif dist == "lw_bw":                                     # HARD: 2nd vs 2nd (expect decay)
        sel = np.isin(labels, [names.index("lax_wendroff"), names.index("beam_warming")])
        Xd, yd, gd = X[sel], labels[sel], groups[sel]
    elif dist == "four":                                      # 4-way scheme ID
        Xd, yd, gd = X, labels, groups
    return cross_val_score(make_clf(), Xd, yd, groups=gd, cv=gkf).mean()

# ----------------------------------------------------------------------
# Sweep everything
# ----------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.01, 0.05]
DOWNSAMPLES  = [1, 2, 4, 8, 16, 32]
OBS_MODELS   = ["raw", "antialias"]
NORMS        = [False, True]
DISTS        = ["diff_disp", "lw_bw", "four"]

results = {}   # (norm, obs, noise, dist, ds) -> accuracy
total = len(NOISE_LEVELS)*len(OBS_MODELS)*len(DOWNSAMPLES)*len(NORMS)*len(DISTS)
done = 0
for li, level in enumerate(NOISE_LEVELS):
    Rn = add_noise(resids, level, seed=1000+li)
    for obs_model in OBS_MODELS:
        for ds in DOWNSAMPLES:
            obs_field = observe(Rn, obs_model, ds)
            for norm in NORMS:
                X = featurize(obs_field, norm)
                for dist in DISTS:
                    results[(norm, obs_model, level, dist, ds)] = evaluate(X, dist)
                    done += 1
        print(f"  ... noise={level:>4} obs={obs_model:<9} done {done}/{total}")

obs_pts = {ds: len(range(0, N, ds)) for ds in DOWNSAMPLES}

# ----------------------------------------------------------------------
# Tables - one per distinction, both norm modes + both obs models side by side
#   col legend: r/a = raw / anti-aliased ; U/N = unnormalized / L2-normalized ;
#               0/5 = noise 0 / noise 0.05
# ----------------------------------------------------------------------
COLS = [("raw",False,0.0,"rU0"), ("raw",False,0.05,"rU5"),
        ("raw",True, 0.0,"rN0"), ("raw",True, 0.05,"rN5"),
        ("antialias",False,0.0,"aU0"), ("antialias",False,0.05,"aU5"),
        ("antialias",True, 0.0,"aN0"), ("antialias",True, 0.05,"aN5")]
TITLES = {"diff_disp":"DIFFUSIVE vs DISPERSIVE   (load-bearing claim - must survive)",
          "lw_bw":    "LAX-WENDROFF vs BEAM-WARMING   (2nd vs 2nd - decay = Nyquist result)",
          "four":     "4-WAY SCHEME ID"}
CHANCE = {"diff_disp":0.5, "lw_bw":0.5, "four":0.25}

print("\nlegend:  r=raw-decimate  a=anti-aliased | U=unnormalized  N=L2-normalized | 0=noise0  5=noise0.05")
for dist in DISTS:
    print(f"\n=== {TITLES[dist]}   (chance={CHANCE[dist]}) ===")
    print(f"{'obs_pts':>8} | " + " ".join(f"{lab:>6}" for *_, lab in COLS))
    for ds in DOWNSAMPLES:
        row = " ".join(f"{results[(nm,ob,nz,dist,ds)]:6.3f}" for ob,nm,nz,_ in COLS)
        print(f"{obs_pts[ds]:>8} | {row}")

# ----------------------------------------------------------------------
# (4) Aliasing check: warn where raw beats anti-aliased (info folding back)
# ----------------------------------------------------------------------
print("\n--- aliasing check (raw decimation scoring ABOVE anti-aliased by >0.02) ---")
warns = []
for norm in NORMS:
    for level in [0.0, 0.05]:
        for ds in DOWNSAMPLES:
            for dist in DISTS:
                r = results[(norm,"raw",level,dist,ds)]
                a = results[(norm,"antialias",level,dist,ds)]
                if r - a > 0.02:
                    warns.append((dist, obs_pts[ds], norm, level, r, a, r-a))
if not warns:
    print("  none - anti-aliased >= raw everywhere (no aliasing leakage detected)")
else:
    for dist, pts, norm, level, r, a, d in sorted(warns, key=lambda w: -w[-1])[:12]:
        nm = "N" if norm else "U"
        print(f"  {dist:>10} pts={pts:>3} {nm} noise={level:<4}: raw={r:.3f} > aa={a:.3f}  (+{d:.3f}) "
              f"-> high-k info aliased into band")

# ----------------------------------------------------------------------
# CSV dump (all 216 numbers, nothing lost to table layout)
# ----------------------------------------------------------------------
csv_path = os.path.join(TAB, "fft_shape_probe_results.csv")
with open(csv_path, "w") as f:
    f.write("normalized,obs_model,noise,distinction,downsample,obs_pts,accuracy\n")
    for (norm, ob, lv, dist, ds), acc in results.items():
        f.write(f"{int(norm)},{ob},{lv},{dist},{ds},{obs_pts[ds]},{acc:.4f}\n")
print(f"\nfull results -> {csv_path}")

# ----------------------------------------------------------------------
# Plots - the load-bearing claim, the Nyquist decay, and the aliasing effect
# ----------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pts = [obs_pts[ds] for ds in DOWNSAMPLES]
def curve(norm, ob, lv, dist): return [results[(norm, ob, lv, dist, ds)] for ds in DOWNSAMPLES]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# Panel 1: load-bearing claim under the honest pipeline (anti-aliased)
ax[0].plot(pts, curve(False,"antialias",0.0,"diff_disp"), "o-",  label="unnorm, noise 0")
ax[0].plot(pts, curve(True, "antialias",0.0,"diff_disp"), "s-",  label="L2-norm, noise 0")
ax[0].plot(pts, curve(True, "antialias",0.05,"diff_disp"),"^-",  label="L2-norm, noise 5%")
ax[0].axhline(0.5, color="grey", ls=":")
ax[0].set_title("Load-bearing: diffusive vs dispersive\n(anti-aliased, GroupKFold-by-IC)")

# Panel 2: Nyquist decay of same-order + 4-way (honest setting: L2-norm, anti-aliased)
ax[1].plot(pts, curve(True,"antialias",0.0,"lw_bw"), "o-", label="LW vs BW, noise 0")
ax[1].plot(pts, curve(True,"antialias",0.05,"lw_bw"),"o--",label="LW vs BW, noise 5%")
ax[1].plot(pts, curve(True,"antialias",0.0,"four"),  "s-", label="4-way, noise 0")
ax[1].axhline(0.5, color="grey", ls=":"); ax[1].axhline(0.25, color="grey", ls=":")
ax[1].set_title("Nyquist limit: same-order + 4-way\n(L2-norm, anti-aliased)")

# Panel 3: aliasing - raw vs anti-aliased for the load-bearing claim
ax[2].plot(pts, curve(True,"raw",0.0,"diff_disp"),      "o-", label="raw decimation")
ax[2].plot(pts, curve(True,"antialias",0.0,"diff_disp"),"s-", label="anti-aliased")
ax[2].axhline(0.5, color="grey", ls=":")
ax[2].set_title("Aliasing effect on diff/disp\n(L2-norm, noise 0)")

for a_ in ax:
    a_.set_xscale("log", base=2); a_.invert_xaxis()
    a_.set_xlabel("observed points (coarser --->)"); a_.set_ylabel("GroupKFold CV accuracy")
    a_.set_ylim(0.2, 1.02); a_.legend(fontsize=8); a_.grid(alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(FIG, "fft_shape_probe_result.png")
plt.savefig(plot_path, dpi=130)
print(f"plot         -> {plot_path}")

# ----------------------------------------------------------------------
# Verdict - read straight off the produced numbers (no assumptions)
# ----------------------------------------------------------------------
def g(norm, ob, lv, dist, ds): return results[(norm, ob, lv, dist, ds)]
print("\n" + "="*72)
print("VERDICT  (read off this run - the hardest honest setting)")
print("="*72)

full_old_like = g(False, "raw", 0.0, "diff_disp", 1)   # closest to the original 0.99 setup
full_honest   = g(True,  "antialias", 0.0,  "diff_disp", 1)
print(f"diff/disp @ full res:  old-like (unnorm,raw,noise0) = {full_old_like:.3f} | "
      f"honest (L2-norm,AA,noise0) = {full_honest:.3f}")

for ds in (8, 16, 32):
    hard = g(True, "antialias", 0.05, "diff_disp", ds)   # L2-norm + anti-aliased + 5% noise
    print(f"diff/disp @ {obs_pts[ds]:>3} pts  (L2-norm, anti-aliased, 5% noise) = {hard:.3f}")

dd16 = g(True, "antialias", 0.05, "diff_disp", 16)       # headline kill-switch number
if   dd16 > 0.90: tag, msg = "GREEN", "fingerprint is real, IC- and magnitude-invariant. Green light for Phase 1."
elif dd16 > 0.70: tag, msg = "AMBER", "degraded but above chance - partly an IC/amplitude artifact. Rethink scope."
else:             tag, msg = "RED",   "easy signal was largely an IC or amplitude artifact. Do NOT start Phase 1."
print(f"\n[{tag}] load-bearing diff/disp @ 16 pts (hardest honest) = {dd16:.3f} -> {msg}")

lw_fine  = g(True, "antialias", 0.0, "lw_bw", 1)
lw_coarse= g(True, "antialias", 0.0, "lw_bw", 32)
print(f"[expected] LW-vs-BW slides {lw_fine:.3f} (256 pts) -> {lw_coarse:.3f} (8 pts): "
      f"{'Nyquist decay, report as a measured limit' if lw_fine-lw_coarse>0.04 else 'little decay this run'}")
