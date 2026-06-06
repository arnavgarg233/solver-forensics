# solver-forensics

**A modified-equation signature method for identifying hidden discretization schemes** - identify the numerical scheme inside a black-box solver from its output alone.

Every consistent discretization solves not the target equation but a nearby *modified
equation*, whose leading correction terms are fixed by the scheme. The residual between a
solver's output and a reference therefore carries a readable fingerprint of *which scheme
produced it*. This repository recovers that fingerprint, attributes the scheme, and - through a
set of negative controls - establishes exactly when that attribution is admissible as evidence.

It is a **controlled audit, not just a classifier**: an attribution method, the controls that
turn a residual into evidence (initial-condition/noise, grid resolution, physical-dispersion
regime), and a measured map of where solver forensics holds. The coefficient-recovery step
coincides with established sparse identification (SITE/SINDy) on the right library and is *more*
robust than it under library misspecification; the contribution is the audit built around it.

| | |
|---|---|
| **Signature** | unit-normalized direction of `c` in `r = u − u_ref ≈ Σ_p c_p ∂_xᵖ u` |
| **Validation** | `GroupKFold` by initial condition + a label-permutation floor on every score |
| **Reach** | finite differences, finite volume, finite elements - linear advection, Burgers, KdV/Kawahara, 2D advection-diffusion, elastic-wave & SUPG FEM, and a real third-party library (`py-pde`) |

## Headline results

Attribution is physical and transfers (linear advection, diffusive-vs-dispersive accuracy):

| | clean | 1% noise | 5% noise | grid/CFL transfer |
|---|---|---|---|---|
| **diffusive vs dispersive** | **0.995** | **0.996** | **0.965** | **0.89 - 1.00** |

It holds across solvers, dimensions, and paradigms (full numbers in `results/tables/`):

| setting | result |
|---|---|
| open-solver audit (`py-pde`), hidden scheme change | **1.00 / 0.98** (clean / 1% noise) |
| 2D advection-diffusion audit | **0.99 - 1.00** |
| real library, documented integrator change | **0.99**, signature matches the changelog |
| **SUPG at 2D engineering scale** (rotating flow, Pe≈13) | presence **0.975**, silent-τ detuning **0.96-1.00** |
| production schemes, 9-way identification | **0.88** (full coefficient vector) |
| FEM element order, convergence rate | detection **1.00** |
| dense vs sparse recovery (SITE/STLSQ) | matched on support; more robust under over-specification |

The method maps its own envelope, and that is the point. A single passive snapshot cannot
separate a **grid change** from a scheme change - so the method *says so*, and an active
**convergence-rate** query, grid-invariant by construction, removes the confound (shown in 1D,
2D, Burgers, KdV, Kawahara, and finite elements). Reporting that boundary, with a permutation
floor on every score, is exactly what makes the positive results trustworthy.

## Repository layout

```
solver-forensics/
├── src/
│   ├── attribution/   the signature is real, physical, and transfers across grid/CFL
│   ├── robustness/    harder regimes - nonlinear, dispersive, production schemes
│   ├── audit/         the application - 1D/2D py-pde, real-library, SUPG & stabilization audits
│   ├── limits/        where attribution holds, the active-query repairs, identifiability, cost
│   ├── mechanics/     computational mechanics - elastic wave, 2D unstructured FEM
│   └── baselines/     head-to-head against sparse (SITE-style) recovery
├── scripts/make_figures.py   regenerates the core paper figures
├── figures/                  publication figures
├── results/tables/           run-of-record metrics (CSV)
├── requirements.txt
└── LICENSE
```

Each script is self-contained: it validates its solver, prints its result table with a
label-permutation floor and a GO / KILL decision, and writes metrics to `results/tables/`.

## Installation

```bash
pip install -r requirements.txt
```

CPU-only, tested on macOS / Apple Silicon and Linux. Deterministic given the in-script fixed
seeds; NumPy-2 compatible.

## Reproducing

```bash
python src/attribution/coefficient_attribution.py   # any experiment runs standalone
python src/audit/audit_2d.py                         # 2D audit
python scripts/make_figures.py                       # regenerate the paper figures
```

## Use it on your own solver

Beyond reproducing the paper, the method ships as a small reusable package - point it at *your
own* solver's output:

```bash
pip install -e .          # or: pip install git+https://github.com/arnavgarg233/solver-forensics
```

```python
import solver_forensics as sf

# one field -> its modified-equation fingerprint (unit coefficient direction)
fp = sf.signature(u_solver, u_ref, dx=L/N)          # periodic fields on a uniform grid

# attribute a scheme across initial conditions, with a built-in permutation floor
result = sf.audit({
    "scheme_A": [(uA[i], uref[i]) for i in range(n_ic)],   # one (solver, reference) pair per IC
    "scheme_B": [(uB[i], uref[i]) for i in range(n_ic)],
}, dx=L/N)
# {'accuracy': 0.99, 'permutation_floor': 0.50, 'margin': 0.49, 'admissible': True}
```

`signature` returns the magnitude-invariant fingerprint; `audit` reports the cross-validated
(GroupKFold-by-initial-condition) attribution accuracy against its label-permutation floor - so
every call comes with its own admissibility check, not just a number.

## Citation

```bibtex
@article{garg2026modified,
  title   = {A modified-equation signature method for identifying hidden discretization schemes},
  author  = {Garg, Arnav},
  journal = {Under review},
  year    = {2026}
}
```

The exact code and results used in the paper are archived at release
[`v1.0.0`](https://github.com/arnavgarg233/solver-forensics/releases/tag/v1.0.0).

## License

MIT - see [`LICENSE`](LICENSE).
