import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))              # repo root
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import supg_2d_engineering as A                              # verified FEM + signature machinery
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

import csv
import warnings

warnings.filterwarnings("ignore")

B = 2000
SIGMAS = (0.0, A.SIGMA_MAIN)
CONFIGS = ("galerkin", "supg", "supg_half", "supg_2tau", "streamdiff_inc", "artvisc")
TAU_SCALES = {"supg": 1.0, "supg_half": 0.5, "supg_2tau": 2.0}


def sig_from_grid(Us, Ur):
    R = Us - Ur
    Dlib, sl = A._fd_library(Us)
    Amat = np.column_stack([Dlib[name].ravel() for name in A.LIB])
    b = R[sl, sl].ravel()
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    nrm = np.linalg.norm(c)
    return c / nrm if nrm > 0 else c


def _solve_clean_grid(config, pts, elems, on_bnd, ic, geom, nu_art):
    if config == "galerkin":
        u, _ = A.assemble("galerkin", pts, elems, on_bnd, ic, geom=geom)
    elif config == "artvisc":
        u, _ = A.assemble("artvisc", pts, elems, on_bnd, ic,
                          nu_art=nu_art, geom=geom)
    elif config == "streamdiff_inc":
        u, _ = A.assemble("streamdiff_inc", pts, elems, on_bnd, ic, geom=geom)
    else:
        u, _ = A.assemble("supg", pts, elems, on_bnd, ic,
                          tau_scale=TAU_SCALES[config], geom=geom)
    return A._to_grid(pts, u)


def _signatures_from_clean(config, clean_grids, ref_grids, sigma, seed):
    rng = np.random.default_rng(seed)
    signatures = []
    for Us_clean, Ur in zip(clean_grids, ref_grids):
        Us = Us_clean
        if sigma > 0:
            rms = np.sqrt(np.mean(Us**2))
            Us = Us + sigma * rms * rng.standard_normal(Us.shape)
        signatures.append(sig_from_grid(Us, Ur))
    return np.asarray(signatures)


def _bootstrap_summary(S, seed):
    full_mean = np.mean(S, axis=0)
    full_norm = np.linalg.norm(full_mean)
    full_dir = full_mean / full_norm if full_norm > 0 else full_mean

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(S), size=(B, len(S)))
    boot_means = np.mean(S[sample_idx], axis=1)
    boot_norms = np.linalg.norm(boot_means, axis=1, keepdims=True)
    boot_dirs = np.divide(boot_means, boot_norms,
                          out=np.zeros_like(boot_means), where=boot_norms > 0)

    ci_lo, ci_hi = np.percentile(boot_dirs, (2.5, 97.5), axis=0)
    cosines = np.abs(boot_dirs @ full_dir)
    angular = np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))
    return dict(mean=full_dir, ci_lo=ci_lo, ci_hi=ci_hi,
                angular_ci_deg=float(np.percentile(angular, 95.0)))


def _intervals_disjoint(a, b):
    return (a["ci_hi"] < b["ci_lo"]) | (b["ci_hi"] < a["ci_lo"])


def _intervals_overlap(a, b):
    return np.maximum(a["ci_lo"], b["ci_lo"]) <= np.minimum(a["ci_hi"], b["ci_hi"])


def _compact_signature(summary):
    fields = []
    for j, component in enumerate(A.LIB):
        fields.append(f"{component}={summary['mean'][j]:+.4f}"
                      f"[{summary['ci_lo'][j]:+.4f},{summary['ci_hi'][j]:+.4f}]")
    return " ".join(fields)


def _write_csv(path, summaries, disjoint_some, overlap_all):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "sigma", "component", "mean", "ci_lo", "ci_hi",
                         "angular_ci_deg", "config_pair", "disjoint_on_some_component",
                         "overlap_on_all_components"])
        for config in CONFIGS:
            for sigma in SIGMAS:
                summary = summaries[(config, sigma)]
                for j, component in enumerate(A.LIB):
                    writer.writerow([
                        config,
                        f"{sigma:.2f}",
                        component,
                        f"{summary['mean'][j]:.8f}",
                        f"{summary['ci_lo'][j]:.8f}",
                        f"{summary['ci_hi'][j]:.8f}",
                        f"{summary['angular_ci_deg']:.8f}",
                        "",
                        "",
                        "",
                    ])
        writer.writerow(["summary", "", "", "", "", "", "",
                         "supg_vs_galerkin", int(disjoint_some), ""])
        writer.writerow(["summary", "", "", "", "", "", "",
                         "supg_vs_streamdiff_inc", "", int(overlap_all)])


def main():
    np.seterr(all="ignore")
    print("=" * 100)
    print("MEASUREMENT: BOOTSTRAP UNCERTAINTY INTERVALS FOR MODIFIED-EQUATION SIGNATURES")
    print("measurand = solver configuration; signal = unit signature over the verified 2D FD library")
    print("=" * 100)
    print(f"ICs: {A.N_IC}  |  working mesh: n_side=28, seed=2026  |  reference mesh: n_side=96, seed=7")
    print(f"noise levels: {SIGMAS}  |  bootstrap resamples per cell: B={B}")

    ics = [A.make_ic(1000 + i) for i in range(A.N_IC)]
    pts, elems, on_bnd = A.make_mesh(28, 2026)
    geom = A.mesh_geometry(pts, elems, D=A.D_PHYS)
    ref_pts, ref_elems, ref_bnd = A.make_mesh(96, 7)

    nu_art = A.added_diffusion_supg(pts, elems, tau_scale=1.0, geom=geom)
    print(f"working mesh: {len(pts)} nodes, {len(elems)} triangles; matched nu_art={nu_art:.6e}")

    print(f"\n[reference] fine SUPG solve once per IC ({A.N_IC} fields) ...")
    ref_grids = A.reference_grids(ics, ref_pts, ref_elems, ref_bnd)
    print("[reference] cached fine SUPG grids ready")

    print("[working solves] one clean FEM solve per configuration and IC ...")
    clean = {}
    for config in CONFIGS:
        clean[config] = [_solve_clean_grid(config, pts, elems, on_bnd, ic, geom, nu_art)
                         for ic in ics]
        print(f"   {config:<16} clean grids ready")

    signatures = {}
    summaries = {}
    for config_index, config in enumerate(CONFIGS):
        for sigma_index, sigma in enumerate(SIGMAS):
            noise_seed = 100 + 1000 * config_index + 100 * sigma_index
            bootstrap_seed = 10000 + 1000 * config_index + 100 * sigma_index
            S = _signatures_from_clean(config, clean[config], ref_grids, sigma, noise_seed)
            signatures[(config, sigma)] = S
            summaries[(config, sigma)] = _bootstrap_summary(S, bootstrap_seed)
        print(f"   {config:<16} sigma sweep + bootstrap CIs ready")

    print("\n[RESULT TABLE] mean unit direction [95% percentile CI]; angular column is 95th percentile (degrees)")
    header = f"{'config':<18} {'sigma':>7} " + " ".join(f"{c:>25}" for c in A.LIB) + f" {'angular_ci_deg':>15}"
    print(header)
    print("-" * len(header))
    for config in CONFIGS:
        for sigma in SIGMAS:
            summary = summaries[(config, sigma)]
            values = " ".join(
                f"{summary['mean'][j]:+.4f}[{summary['ci_lo'][j]:+.4f},{summary['ci_hi'][j]:+.4f}]"
                for j in range(len(A.LIB)))
            print(f"{config:<18} {sigma:>7.2f} {values} {summary['angular_ci_deg']:>15.4f}")

    supg = summaries[("supg", A.SIGMA_MAIN)]
    galerkin = summaries[("galerkin", A.SIGMA_MAIN)]
    streamdiff = summaries[("streamdiff_inc", A.SIGMA_MAIN)]
    disjoint_mask = _intervals_disjoint(supg, galerkin)
    overlap_mask = _intervals_overlap(supg, streamdiff)
    disjoint_some = bool(np.any(disjoint_mask))
    overlap_all = bool(np.all(overlap_mask))
    disjoint_components = [A.LIB[j] for j, value in enumerate(disjoint_mask) if value]
    overlap_components = [A.LIB[j] for j, value in enumerate(overlap_mask) if value]

    print("\n[SEPARATION CHECKS] sigma=0.01")
    print(f"supg vs galerkin: CIs DISJOINT on >=1 component = {int(disjoint_some)}"
          f" ({', '.join(disjoint_components) if disjoint_components else 'none'})")
    print(f"supg vs streamdiff_inc: CIs OVERLAP on ALL components = {int(overlap_all)}"
          f" ({len(overlap_components)}/{len(A.LIB)} components overlap)")

    chance_line = ("chance/permutation-floor: n/a for this bootstrap CI measurement "
                   "(no class labels; IC resampling supplies the uncertainty distribution)")
    print(f"\n{chance_line}")

    csv_path = os.path.join(TAB, "uncertainty_intervals.csv")
    _write_csv(csv_path, summaries, disjoint_some, overlap_all)
    print(f"CSV -> {csv_path}")

    print("\n[VERDICT] compact signatures +/- 95% CI")
    for config in CONFIGS:
        for sigma in SIGMAS:
            print(f"  {config:<16} sigma={sigma:.2f}  {_compact_signature(summaries[(config, sigma)])}"
                  f"  angle95={summaries[(config, sigma)]['angular_ci_deg']:.4f} deg")
    go = disjoint_some and overlap_all
    print(f"\n[{'GO' if go else 'NO-GO'}] measurement uncertainty deliverable: "
          f"supg distinguishable from galerkin with CIs separated={int(disjoint_some)}; "
          f"supg vs streamdiff_inc CIs overlap={int(overlap_all)} (collinearity limit)")
    return dict(signatures=signatures, summaries=summaries,
                disjoint_some=disjoint_some, overlap_all=overlap_all,
                csv_path=csv_path)


if __name__ == "__main__":
    main()
