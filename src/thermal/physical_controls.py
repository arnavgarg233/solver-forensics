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

import csv
import time

sys.path.insert(0, os.path.join(_ROOT, "src", "measurement"))
import abstention as AB


BASE_PE = 100.0
N_IC = HC.N_IC
SIGMA = 0.01
TARGET_FPR = 0.05
DETECT_THRESHOLD = 0.50
NO_FAULT_THRESHOLD = 0.15
WORKING_MESH = (60, 20, 2026)


def supervised_sensitivity(nominal, changed, target_fpr=0.05):
    X = A.feats(np.vstack([nominal, changed]))
    y = np.r_[np.zeros(len(nominal)), np.ones(len(changed))]
    g = np.r_[np.arange(len(nominal)), np.arange(len(changed))]
    k = min(5, len(nominal))
    proba = cross_val_predict(A._clf(), X, y, groups=g, cv=GroupKFold(k), method="predict_proba")[:, 1]
    thr = np.percentile(proba[y == 0], 100 * (1 - target_fpr))
    return float(np.mean(proba[y == 1] > thr))


def scalar_sensitivity(nom_vals, chg_vals, target_fpr=0.05):
    nom = np.asarray(nom_vals, float); chg = np.asarray(chg_vals, float)
    center = np.median(nom)
    thr = np.percentile(np.abs(nom - center), 100 * (1 - target_fpr))
    return float(np.mean(np.abs(chg - center) > thr))


def _mesh(nx, ny, seed):
    pts, elems, tags = HC.make_channel_mesh(nx, ny, seed=seed)
    return {
        "spec": f"{nx}x{ny}, seed={seed}",
        "pts": pts,
        "elems": elems,
        "tags": tags,
        "geoms": {},
    }


def _geometry(mesh, Pe):
    Pe = float(Pe)
    if Pe not in mesh["geoms"]:
        a_th = HC.thermal_diffusivity(Pe)
        mesh["geoms"][Pe] = HC.channel_mesh_geometry(
            mesh["pts"], mesh["elems"], a_th)
    return mesh["geoms"][Pe]


def _clean_grids(mesh, ics, Pes, scheme, alpha=1.0, nu_art=None):
    """Solve one clean field per IC, reusing geometry once per (mesh, Pe)."""
    if len(ics) != len(Pes):
        raise ValueError("ics and Pes must have the same length")

    pe_values = [float(Pe) for Pe in Pes]
    if scheme == "artvisc":
        art_by_pe = {}
        for Pe in sorted(set(pe_values)):
            geom = _geometry(mesh, Pe)
            art_by_pe[Pe] = (
                float(nu_art) if nu_art is not None
                else HC.matched_artificial_diffusion(geom, alpha=1.0))
    else:
        art_by_pe = {}

    grids = []
    for ic, Pe in zip(ics, pe_values):
        geom = _geometry(mesh, Pe)
        kwargs = {
            "scheme": scheme,
            "pts": mesh["pts"],
            "elems": mesh["elems"],
            "tags": mesh["tags"],
            "Pe": Pe,
            "ic": ic,
            "alpha": alpha,
            "geom": geom,
        }
        if scheme == "artvisc":
            kwargs["nu_art"] = art_by_pe[Pe]
        T = HC.assemble_channel(**kwargs)
        grids.append(HC.to_channel_grid(mesh["pts"], T))
    return grids


def _reference_grids(label, ics, Pes, cache):
    """Build fine nominal-SUPG references once for each labelled condition."""
    if len(ics) != len(Pes):
        raise ValueError("ics and Pes must have the same length")

    grouped = {}
    for index, Pe in enumerate(Pes):
        grouped.setdefault(float(Pe), []).append(index)

    refs = [None] * len(ics)
    for Pe, indices in grouped.items():
        key = (label, Pe)
        if key not in cache:
            subset = [ics[index] for index in indices]
            cache[key] = HC.reference_channel_grids(subset, Pe=Pe)
        for index, reference in zip(indices, cache[key]):
            refs[index] = reference
    return refs


def _signatures(clean_grids, reference_grids, seed):
    """Add deterministic field-relative observation noise after all solves."""
    rng = np.random.default_rng(seed)
    signatures = []
    for Us, Ur in zip(clean_grids, reference_grids):
        rms = np.sqrt(np.mean(Us**2))
        Us_noisy = Us + SIGMA * rms * rng.standard_normal(Us.shape)
        signatures.append(HC.sig_from_grid(Us_noisy, Ur))
    return np.asarray(signatures)


def _flux_control(ics):
    return [
        {**ic, "g_flux": float(ic["g_flux"] * (1.10 if index % 2 == 0 else 0.90))}
        for index, ic in enumerate(ics)
    ]


def _location_control(ics):
    alternate = [HC.make_thermal_ic(2000 + index) for index in range(N_IC)]
    return [
        {**ic, "xh0": alt["xh0"], "xh1": alt["xh1"]}
        for ic, alt in zip(ics, alternate)
    ]


def _inlet_control(ics):
    alternate = [HC.make_thermal_ic(3000 + index) for index in range(N_IC)]
    return [
        {**ic, "inlet_ramp": alt["inlet_ramp"]}
        for ic, alt in zip(ics, alternate)
    ]


def _decision(change_type, sensitivity):
    detected = sensitivity >= DETECT_THRESHOLD
    if change_type == "grid":
        if detected:
            # In abstention.py, this is decide(grid_controlled=False,
            # sensitivity_above_fpr=True), represented by acc > floor+margin.
            return AB.decide(
                grid_controlled=False, acc=sensitivity, floor=TARGET_FPR)
        return "NO-FAULT"
    return "DETECT" if detected else "NO-FAULT"


def _row(number, change_type, description, sensitivity):
    sensitivity = float(sensitivity)
    decision = _decision(change_type, sensitivity)
    if change_type == "numerical":
        desired = "DETECT"
        correct = sensitivity >= DETECT_THRESHOLD and decision == desired
    elif change_type == "thermal-operating":
        desired = "NO-FAULT"
        correct = sensitivity <= NO_FAULT_THRESHOLD and decision == desired
    else:
        desired = "INDETERMINATE"
        correct = sensitivity >= DETECT_THRESHOLD and decision == desired
    return {
        "row": number,
        "change_type": change_type,
        "description": description,
        "sensitivity": sensitivity,
        "desired": desired,
        "decision": decision,
        "correct": int(correct),
    }


def _write_csv(rows):
    path = os.path.join(TAB, "physical_controls.csv")
    fields = ["row", "change_type", "description", "sensitivity",
              "desired", "decision", "correct"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["sensitivity"] = f"{row['sensitivity']:.6f}"
            writer.writerow(record)
    return path


def main():
    started = time.perf_counter()
    if N_IC != 60:
        raise RuntimeError("the physical-controls protocol requires HC.N_IC == 60")

    print("=" * 100)
    print("ICHMT PHYSICAL-CONTROL AUDIT | heated-channel modified-equation signatures")
    print("measurand = solver numerics; matched fine-SUPG references isolate numerical error")
    print("=" * 100)
    print(f"N_IC={N_IC} | baseline Pe={BASE_PE:.0f} | sigma={SIGMA:.2f} | "
          f"working mesh={WORKING_MESH[0]}x{WORKING_MESH[1]}, seed={WORKING_MESH[2]}")
    print(f"detector = held-out GroupKFold-by-IC classifier, target FPR={TARGET_FPR:.2f}")

    ics = [HC.make_thermal_ic(1000 + index) for index in range(N_IC)]
    baseline_Pes = [BASE_PE] * N_IC
    references = {}

    working = _mesh(*WORKING_MESH)
    print("\nREFERENCE | fine nominal-SUPG baseline, one solve per IC")
    baseline_refs = HC.reference_channel_grids(ics, Pe=BASE_PE)
    references[("baseline", BASE_PE)] = baseline_refs
    print(f"  cached {len(baseline_refs)} references on the default fine mesh")

    print("WORKING SOLVES | baseline nominal SUPG alpha=1.0")
    baseline_clean = _clean_grids(
        working, ics, baseline_Pes, "supg", alpha=1.0)
    baseline_signatures = _signatures(baseline_clean, baseline_refs, seed=41001)

    rows = []

    print("  row 1: SUPG detuning alpha=0.5")
    half_clean = _clean_grids(
        working, ics, baseline_Pes, "supg", alpha=0.5)
    half_refs = references[("baseline", BASE_PE)]
    half_signatures = _signatures(half_clean, half_refs, seed=41002)
    rows.append(_row(
        1, "numerical", "SUPG detuning alpha=0.5 (under-stabilized)",
        supervised_sensitivity(baseline_signatures, half_signatures, TARGET_FPR)))

    print("  row 2: Galerkin replaces nominal SUPG")
    galerkin_clean = _clean_grids(
        working, ics, baseline_Pes, "galerkin", alpha=1.0)
    galerkin_signatures = _signatures(galerkin_clean, baseline_refs, seed=41003)
    rows.append(_row(
        2, "numerical", "Galerkin replaces nominal SUPG", 
        supervised_sensitivity(baseline_signatures, galerkin_signatures, TARGET_FPR)))

    print("  row 3: matched isotropic ArtVisc replaces nominal SUPG")
    baseline_geom = _geometry(working, BASE_PE)
    nu_art = HC.matched_artificial_diffusion(baseline_geom, alpha=1.0)
    artvisc_clean = _clean_grids(
        working, ics, baseline_Pes, "artvisc", alpha=1.0, nu_art=nu_art)
    artvisc_signatures = _signatures(artvisc_clean, baseline_refs, seed=41004)
    artvisc_sensitivity = supervised_sensitivity(
        baseline_signatures, artvisc_signatures, TARGET_FPR)
    rows.append(_row(
        3, "numerical", "matched isotropic ArtVisc (nu_art from nominal working mesh)",
        artvisc_sensitivity))

    print("  row 4: heat-flux control (+/-10%, matched references)")
    flux_ics = _flux_control(ics)
    flux_Pes = [BASE_PE] * N_IC
    flux_refs = _reference_grids("flux", flux_ics, flux_Pes, references)
    flux_clean = _clean_grids(
        working, flux_ics, flux_Pes, "supg", alpha=1.0)
    flux_signatures = _signatures(flux_clean, flux_refs, seed=41005)
    rows.append(_row(
        4, "thermal-operating", "heat flux varied by alternating +10%/-10%", 
        supervised_sensitivity(baseline_signatures, flux_signatures, TARGET_FPR)))

    print("  row 5: Peclet control (Pe=90/110, matched references)")
    pe_ics = list(ics)
    pe_Pes = [90.0 if index % 2 == 0 else 110.0 for index in range(N_IC)]
    pe_refs = _reference_grids("peclet", pe_ics, pe_Pes, references)
    pe_clean = _clean_grids(
        working, pe_ics, pe_Pes, "supg", alpha=1.0)
    pe_signatures = _signatures(pe_clean, pe_refs, seed=41006)
    rows.append(_row(
        5, "thermal-operating", "Peclet varied by alternating Pe=90/110", 
        supervised_sensitivity(baseline_signatures, pe_signatures, TARGET_FPR)))

    print("  row 6: heater-location control from alternate IC seed block")
    location_ics = _location_control(ics)
    location_Pes = [BASE_PE] * N_IC
    location_refs = _reference_grids("heater-location", location_ics,
                                     location_Pes, references)
    location_clean = _clean_grids(
        working, location_ics, location_Pes, "supg", alpha=1.0)
    location_signatures = _signatures(
        location_clean, location_refs, seed=41007)
    rows.append(_row(
        6, "thermal-operating", "heater location shifted using xh0/xh1 from seeds 2000+i",
        supervised_sensitivity(baseline_signatures, location_signatures, TARGET_FPR)))

    print("  row 7: inlet-temperature-ramp control from alternate IC seed block")
    inlet_ics = _inlet_control(ics)
    inlet_Pes = [BASE_PE] * N_IC
    inlet_refs = _reference_grids("inlet-ramp", inlet_ics, inlet_Pes, references)
    inlet_clean = _clean_grids(
        working, inlet_ics, inlet_Pes, "supg", alpha=1.0)
    inlet_signatures = _signatures(inlet_clean, inlet_refs, seed=41008)
    rows.append(_row(
        7, "thermal-operating", "inlet temperature ramp varied using seeds 3000+i",
        supervised_sensitivity(baseline_signatures, inlet_signatures, TARGET_FPR)))

    print("  row 8: grid-only control, coarse 44x14 vs fine 72x24")
    coarse = _mesh(44, 14, 3001)
    fine = _mesh(72, 24, 3002)
    coarse_clean = _clean_grids(
        coarse, ics, baseline_Pes, "supg", alpha=1.0)
    fine_clean = _clean_grids(
        fine, ics, baseline_Pes, "supg", alpha=1.0)
    # Keep separate cache labels so each mesh's signature population has its
    # own matched fine nominal-SUPG reference acquisition.
    coarse_refs = _reference_grids("grid-coarse", ics, baseline_Pes, references)
    fine_refs = _reference_grids("grid-fine", ics, baseline_Pes, references)
    coarse_signatures = _signatures(coarse_clean, coarse_refs, seed=41009)
    fine_signatures = _signatures(fine_clean, fine_refs, seed=41010)
    grid_sensitivity = supervised_sensitivity(
        fine_signatures, coarse_signatures, TARGET_FPR)
    rows.append(_row(
        8, "grid", "same nominal SUPG on coarse 44x14 vs fine 72x24 meshes",
        grid_sensitivity))

    print("\nRESULT TABLE | calibrated signature sensitivity")
    print(f"{'row':>3} {'change_type':<18} {'description':<66} "
          f"{'sensitivity':>11} {'desired':>15} {'decision':>17} {'ok':>3}")
    print("-" * 145)
    for result in rows:
        print(f"{result['row']:>3} {result['change_type']:<18} "
              f"{result['description']:<66} {result['sensitivity']:>11.3f} "
              f"{result['desired']:>15} {result['decision']:>17} "
              f"{result['correct']:>3d}")

    print("\nCHANCE / FALSE-POSITIVE LINE")
    print(f"  signature threshold calibrated on nominal scores at target FPR={TARGET_FPR:.2f}; "
          f"DETECT iff sensitivity >= {DETECT_THRESHOLD:.2f}")
    print(f"  thermal-operating NO-FAULT band: sensitivity <= {NO_FAULT_THRESHOLD:.2f}; "
          "grid detections use measurement/abstention.py with grid_controlled=False")
    print(f"  ArtVisc diagnostic: SUPG-vs-matched-ArtVisc sensitivity={artvisc_sensitivity:.3f} "
          "(same matched-reference protocol)")

    csv_path = _write_csv(rows)
    print(f"\nCSV -> {csv_path}")

    correct = sum(result["correct"] for result in rows)
    verdict_ok = correct == len(rows)
    print("\n" + "=" * 100)
    print(f"{'GO' if verdict_ok else 'CHECK'} / VERDICT | ICHMT physical-control audit matrix")
    print("=" * 100)
    print(f"  {correct}/{len(rows)} rows match the preregistered desired outcome")
    if verdict_ok:
        print("  Numerical changes DETECT; matched thermal-operating changes are NO-FAULT; "
              "the uncontrolled grid confound ABSTAINS as INDETERMINATE.")
    else:
        print("  MISMATCHES:")
        for result in rows:
            if not result["correct"]:
                print(f"    row {result['row']}: sensitivity={result['sensitivity']:.3f}, "
                      f"desired={result['desired']}, decision={result['decision']}")

    elapsed = time.perf_counter() - started
    print(f"  runtime: {elapsed:.2f} s")
    return rows


if __name__ == "__main__":
    main()
