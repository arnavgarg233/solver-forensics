import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src", "thermal"))
sys.path.insert(0, os.path.join(_ROOT, "src", "audit"))
import heated_channel as HC       # the validated thermal foundation
import supg_2d_engineering as A    # stat helpers: A.feats, A.cv_acc, A.perm_floor, A._clf, A.LIB
from sklearn.model_selection import GroupKFold, cross_val_predict
TAB = os.path.join(_ROOT, "results", "tables"); os.makedirs(TAB, exist_ok=True)

# The sweep must be usable on headless workers.  Keep the cache local to this
# repository so the plotting backend never relies on a home-directory setting.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_ROOT, ".mplcache"))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # CSV is the primary deliverable if plotting is unavailable.
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None

import csv
import time


PE_VALUES = (50, 100, 200, 500, 1000, 3000, 10000, 30000, 100000)
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)  # 0 and 2 are deliberately large-fault anchors.
NOISE = (0.0, 0.01)
N_ICS = 48
TARGET_FPR = 0.05
DETECTED_AT = 0.90
CSV_FIELDS = (
    "Pe", "alpha", "noise", "sig_dist_from_nominal", "detection_sensitivity",
    "Nu_bias_pct", "Twall_bias_pct", "Pe_h_max", "detected",
)


# signature detector: held-out classifier P(changed), threshold on nominal
# scores at 5% FPR, sensitivity=TPR
def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    X = A.feats(np.vstack([nominal, changed]))
    y = np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(changed))]
    k = min(5, len(nominal))
    proba = cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k),
                              method="predict_proba")[:, 1]
    thr = np.percentile(proba[y == 0], 100 * (1 - target_fpr))
    return float(np.mean(proba[y == 1] > thr))


# scalar detector for a 1-D thermal output v: two-sided |v - median(nominal)|,
# LOO-free threshold at 5% FPR
def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    nom = np.asarray(nom_vals, float); chg = np.asarray(chg_vals, float)
    center = np.median(nom)
    thr = np.percentile(np.abs(nom - center), 100 * (1 - target_fpr))
    return float(np.mean(np.abs(chg - center) > thr))


def _noise_seed(pe_index, alpha_index, noise_index, ic_index):
    """One deterministic, independent observation stream per solver grid."""
    return 91_000_000 + 1_000_000 * pe_index + 10_000 * alpha_index + 100 * noise_index + ic_index


def _add_grid_noise(Us, sigma, seed):
    """RMS-relative Gaussian observation noise; clean FEM solves are never repeated."""
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(Us ** 2))
    return Us + sigma * rms * rng.standard_normal(Us.shape)


def _element_pe_max(geom, a_th):
    return float(np.max(np.abs(geom["u_e"]) * geom["h_e"] / (2.0 * a_th)))


def _direction_distance(signatures, nominal):
    """Return 1-|cos| between the two ensemble-mean unit signatures."""
    left = np.mean(signatures, axis=0)
    right = np.mean(nominal, axis=0)
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    if scale == 0.0:
        return 0.0 if np.allclose(left, right) else 1.0
    cosine = float(np.clip(np.dot(left, right) / scale, -1.0, 1.0))
    return float(1.0 - abs(cosine))


def _solve_clean_configs(pts, elems, tags, pe, ics, geom):
    """Cache each working SUPG solution/grid/output exactly once per IC and alpha."""
    a_th = HC.thermal_diffusivity(pe)
    grids = {}
    outputs = {}
    health = {
        "finite": True,
        "max_linear_residual": 0.0,
        "max_energy_relerr": 0.0,
    }
    for alpha in ALPHAS:
        alpha = float(alpha)
        alpha_grids = np.empty((len(ics), A.GRID_OBS, A.GRID_OBS), dtype=float)
        nu = np.empty(len(ics), dtype=float)
        twall = np.empty(len(ics), dtype=float)
        for i, ic in enumerate(ics):
            temperature, active_tags, meta = HC.assemble_channel(
                "supg", pts, elems, tags, pe, ic=ic, alpha=alpha, geom=geom,
                return_meta=True,
            )
            out = HC.thermal_outputs(
                pts, temperature, active_tags, a_th, g_flux=ic["g_flux"])
            values = (temperature, out["Nu_mean"], out["Twall_max"], out["Tbulk_out"])
            if not all(np.all(np.isfinite(value)) for value in values):
                health["finite"] = False
                raise RuntimeError(f"non-finite thermal output at Pe={pe}, alpha={alpha}, IC={i}")
            alpha_grids[i] = HC.to_channel_grid(pts, temperature)
            nu[i] = out["Nu_mean"]
            twall[i] = out["Twall_max"]

            # The inlet ramp is antisymmetric under the symmetric Poiseuille
            # profile, so the exact inlet enthalpy is zero.  This mirrors the
            # foundation's heat-added versus outlet-enthalpy validation.
            heat_added = float(meta["heat_added"])
            outlet_enthalpy = HC.UBAR * active_tags["Ly"] * out["Tbulk_out"]
            energy_relerr = abs(outlet_enthalpy - heat_added) / max(abs(heat_added), 1.0e-30)
            health["max_energy_relerr"] = max(health["max_energy_relerr"], float(energy_relerr))
            health["max_linear_residual"] = max(
                health["max_linear_residual"], float(meta["linear_residual"]))
        grids[alpha] = alpha_grids
        outputs[alpha] = {"Nu_mean": nu, "Twall_max": twall}
    return grids, outputs, health


def _observed_signatures(clean_grids, refs, pe_index, noise_index, sigma):
    """Recover signatures from the cached clean grids after adding observation noise."""
    observed = {}
    for alpha_index, alpha in enumerate(ALPHAS):
        alpha = float(alpha)
        signatures = np.empty((len(refs), len(A.LIB)), dtype=float)
        for i, reference in enumerate(refs):
            observed_grid = _add_grid_noise(
                clean_grids[alpha][i], sigma,
                _noise_seed(pe_index, alpha_index, noise_index, i),
            )
            signatures[i] = HC.sig_from_grid(observed_grid, reference)
        observed[alpha] = signatures
    return observed


def _rows_for_condition(pe, pe_h_max, clean_outputs, observed):
    """Calculate all requested per-(Pe, alpha, noise) metrics."""
    nominal_sigs = observed[1.0]
    nominal_nu = float(np.mean(clean_outputs[1.0]["Nu_mean"]))
    nominal_twall = float(np.mean(clean_outputs[1.0]["Twall_max"]))
    rows = []
    for alpha in ALPHAS:
        alpha = float(alpha)
        nu_bias = 100.0 * (float(np.mean(clean_outputs[alpha]["Nu_mean"])) - nominal_nu) / nominal_nu
        twall_bias = 100.0 * (float(np.mean(clean_outputs[alpha]["Twall_max"])) - nominal_twall) / nominal_twall
        sensitivity = supervised_sensitivity(nominal_sigs, observed[alpha], TARGET_FPR)
        rows.append({
            "Pe": int(pe),
            "alpha": alpha,
            "sig_dist_from_nominal": _direction_distance(observed[alpha], nominal_sigs),
            "detection_sensitivity": sensitivity,
            "Nu_bias_pct": float(nu_bias),
            "Twall_bias_pct": float(twall_bias),
            "Pe_h_max": float(pe_h_max),
            "detected": int(sensitivity >= DETECTED_AT),
        })
    return rows


def _write_csv(rows):
    path = os.path.join(TAB, "pe_sweep.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
    return path


def _series(rows, alpha, noise, field):
    selected = [row for row in rows if np.isclose(row["alpha"], alpha)
                and np.isclose(row["noise"], noise)]
    selected.sort(key=lambda row: row["Pe"])
    return np.array([row["Pe"] for row in selected], dtype=float), np.array(
        [row[field] for row in selected], dtype=float)


def _make_figure(rows):
    """Write the required three-panel figure without making CSV success depend on it."""
    if plt is None:
        print(f"PLOT WARNING: matplotlib unavailable; CSV was written ({_PLOT_IMPORT_ERROR!r})")
        return None
    try:
        fig_dir = os.path.join(_ROOT, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
        colors = {0.5: "#4C72B0", 1.5: "#C44E52"}
        styles = {0.0: "-", 0.01: "--"}
        markers = {0.0: "o", 0.01: "s"}

        for alpha in (0.5, 1.5):
            for sigma in NOISE:
                pes, values = _series(rows, alpha, sigma, "sig_dist_from_nominal")
                axes[0].plot(pes, values, color=colors[alpha], ls=styles[sigma],
                             marker=markers[sigma], ms=4, lw=1.7,
                             label=fr"$\alpha={alpha:.1f}$, $\sigma={sigma:.2f}$")
                pes, values = _series(rows, alpha, sigma, "detection_sensitivity")
                axes[1].plot(pes, values, color=colors[alpha], ls=styles[sigma],
                             marker=markers[sigma], ms=4, lw=1.7,
                             label=fr"$\alpha={alpha:.1f}$, $\sigma={sigma:.2f}$")

        axes[1].axhline(DETECTED_AT, color="#555555", lw=1.0, ls=(0, (3, 2)))
        axes[1].text(0.02, DETECTED_AT + 0.025, "0.90 detection", fontsize=8,
                     transform=axes[1].get_yaxis_transform())

        for alpha in (0.5, 1.5):
            pes, nu = _series(rows, alpha, 0.0, "Nu_bias_pct")
            _, twall = _series(rows, alpha, 0.0, "Twall_bias_pct")
            axes[2].plot(pes, nu, color=colors[alpha], marker="o", ms=4, lw=1.7,
                         label=fr"Nu, $\alpha={alpha:.1f}$")
            axes[2].plot(pes, twall, color=colors[alpha], marker="s", ms=4, lw=1.5,
                         ls="--", label=fr"$T_{{wall}}$, $\alpha={alpha:.1f}$")

        titles = (
            "A  Signature-direction distance",
            "B  Detection sensitivity (TPR)",
            "C  Clean thermal-output bias",
        )
        ylabels = (
            r"$1-|\cos(\bar c_\alpha,\bar c_1)|$",
            "TPR at 5% FPR",
            "bias from nominal [%]",
        )
        for axis, title, ylabel in zip(axes, titles, ylabels):
            axis.set_xscale("log")
            axis.set_xlabel("Peclet number, Pe")
            axis.set_ylabel(ylabel)
            axis.set_title(title, fontsize=10)
            axis.grid(True, which="both", color="#d9d9d9", lw=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[1].set_ylim(-0.03, 1.03)
        axes[0].legend(frameon=False, fontsize=7.2, loc="best")
        axes[1].legend(frameon=False, fontsize=7.2, loc="best")
        axes[2].axhline(0.0, color="#555555", lw=0.8)
        axes[2].legend(frameon=False, fontsize=7.2, loc="best")
        fig.tight_layout()
        path = os.path.join(fig_dir, "fig_pe_sweep.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:
        print(f"PLOT WARNING: figure generation failed; CSV was written ({exc!r})")
        return None


def _print_result_table(rows):
    print("\nRESULT TABLE :: same geometry, per-Pe thermal audit metrics")
    print("   Pe  alpha  noise     sig-dist     sensitivity   Nu-bias[%]  Twall-bias[%]    Pe_h,max  detected")
    print("  " + "-" * 108)
    for row in rows:
        print(f"  {row['Pe']:6d}  {row['alpha']:5.1f}  {row['noise']:5.2f}"
              f"  {row['sig_dist_from_nominal']:11.4e}  {row['detection_sensitivity']:11.3f}"
              f"  {row['Nu_bias_pct']:11.4f}  {row['Twall_bias_pct']:13.4f}"
              f"  {row['Pe_h_max']:10.2f}  {row['detected']:8d}")


def _is_clean_decline(values):
    """Conservative discrete criterion: no material rise and a material endpoint fall."""
    values = np.asarray(values, dtype=float)
    tolerance = 1.0 / N_ICS + 1.0e-12  # one IC is sampling resolution, not a trend.
    return bool(np.all(np.diff(values) <= tolerance) and values[-1] < values[0] - tolerance)


def _has_window(values):
    """A window requires a material interior gain followed by a material loss."""
    values = np.asarray(values, dtype=float)
    tolerance = 1.0 / N_ICS + 1.0e-12
    if len(values) < 3:
        return False
    peak = float(np.max(values[1:-1]))
    return bool(peak > values[0] + tolerance and peak > values[-1] + tolerance
                and np.any(np.diff(values) > tolerance)
                and np.any(np.diff(values) < -tolerance))


def _verdict(rows, extreme_health):
    sequences = {}
    for alpha in (0.5, 1.5):
        pes, values = _series(rows, alpha, 0.01, "detection_sensitivity")
        sequences[alpha] = (pes, values)

    clean = all(_is_clean_decline(values) for _, values in sequences.values())
    window = all(_has_window(values) for _, values in sequences.values())
    if clean:
        outcome = "A CLEAN DECLINE"
        label = "GO"
        conclusion = ("Subtle-detuning sensitivity falls systematically with Pe; the Peclet "
                      "ceiling claim HOLDS for this same-geometry channel sweep.")
    elif window:
        outcome = "C NON-MONOTONIC"
        label = "WINDOW"
        conclusion = ("Sensitivity rises and then falls, so the evidence supports an auditability "
                      "WINDOW rather than a one-sided Peclet ceiling.")
    else:
        outcome = "B NO CLEAN DECLINE"
        label = "NO-GO"
        conclusion = ("The data do not show a systematic same-geometry decline; drop the Peclet "
                      "ceiling claim in favor of an application-dependent observability limit.")

    print("\nCHANCE / FALSE-POSITIVE RATE: the held-out signature classifier is evaluated as TPR at a"
          f" nominal {100.0 * TARGET_FPR:.1f}% FPR threshold (50% is only the balanced-class accuracy chance level).")
    print("\nSIGMA=0.01 SUBTLE-DETUNING SENSITIVITY :: Pe:TPR")
    for alpha, (pes, values) in sequences.items():
        pairs = ", ".join(f"{int(pe)}:{value:.3f}" for pe, value in zip(pes, values))
        print(f"  alpha={alpha:.1f} -> {pairs}")

    energy_limit = 3.0e-4  # the validated foundation's 0.03% balance benchmark
    health_good = (extreme_health["finite"]
                   and extreme_health["max_linear_residual"] <= 1.0e-8
                   and extreme_health["max_energy_relerr"] <= energy_limit)
    print("\nPe=100000 SOLVER HEALTH :: "
          f"finite={extreme_health['finite']}; max linear residual="
          f"{extreme_health['max_linear_residual']:.3e}; max output energy imbalance="
          f"{extreme_health['max_energy_relerr']:.3e}")
    if health_good:
        print("  No non-finite solve or energy-balance degradation beyond the foundation's 0.03% benchmark was observed.")
    else:
        print("  CAUTION: the extreme-Pe condition exceeds the foundation energy/residual health benchmark; interpret its trend point accordingly.")

    print("\n" + "=" * 104)
    print(f"[{label} / VERDICT] {outcome} :: same-geometry heated-channel Peclet sweep")
    print("=" * 104)
    print(f"  {conclusion}")
    return outcome, label


def main():
    start = time.perf_counter()
    print("=" * 104)
    print("ICHMT :: SAME-GEOMETRY PECLET SWEEP OF SILENT SUPG STABILIZATION DETUNING")
    print("Heated channel only: does subtle-detuning auditability decline when Pe alone changes?")
    print("=" * 104)
    print(f"Protocol: Pe={list(PE_VALUES)} | {N_ICS} ICs (seeds 1000..{1000 + N_ICS - 1}) | "
          "working mesh=60x20, seed=2026")
    print(f"          alphas={list(ALPHAS)} | RMS-relative grid noise={list(NOISE)} | "
          f"held-out target FPR={TARGET_FPR:.2f}")
    print("Reference rule: one fine nominal-SUPG reference per Pe/IC; each clean working solve is cached once,"
          " then only its 64x64 observation is noised.")

    ics = [HC.make_thermal_ic(1000 + i) for i in range(N_ICS)]
    pts, elems, tags = HC.make_channel_mesh(60, 20, seed=2026)
    rows = []
    extreme_health = None

    for pe_index, pe in enumerate(PE_VALUES):
        a_th = HC.thermal_diffusivity(pe)
        geom = HC.channel_mesh_geometry(pts, elems, a_th)
        pe_h_max = _element_pe_max(geom, a_th)
        print(f"\nPe={pe}: computing 48 fine nominal-SUPG reference grids (once for this Pe)")
        references = HC.reference_channel_grids(ics, Pe=pe)
        print(f"Pe={pe}: caching 5 x 48 clean working SUPG configurations; Pe_h,max={pe_h_max:.2f}")
        clean_grids, clean_outputs, health = _solve_clean_configs(pts, elems, tags, pe, ics, geom)
        if pe == 100000:
            extreme_health = health
        for noise_index, sigma in enumerate(NOISE):
            observed = _observed_signatures(clean_grids, references, pe_index, noise_index, sigma)
            for row in _rows_for_condition(pe, pe_h_max, clean_outputs, observed):
                row["noise"] = float(sigma)
                rows.append(row)

    rows.sort(key=lambda row: (row["Pe"], row["alpha"], row["noise"]))
    csv_path = _write_csv(rows)
    figure_path = _make_figure(rows)
    _print_result_table(rows)
    _verdict(rows, extreme_health)
    print(f"\nCSV -> {csv_path}")
    print(f"FIGURE -> {figure_path if figure_path is not None else 'not produced (see plot warning above)'}")
    print(f"RUNTIME: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
