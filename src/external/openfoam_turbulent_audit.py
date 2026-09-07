#!/usr/bin/env python3
"""Frozen turbulent RANS audit of an external finite-volume solver.

The reports asked for turbulent validation. This protocol audits OpenFOAM
``simpleFoam`` with the k-omega SST Reynolds-averaged turbulence model on a
three-dimensional duct, and asks whether the momentum divergence scheme can be
identified from the output velocity field alone.

The contrast is ``bounded Gauss upwind`` against ``bounded Gauss linearUpwind
grad(U)`` for ``div(phi,U)``. Everything else is held fixed inside an operating
case: mesh, boundary conditions, turbulence model, transport properties and
solver controls. The scheme actually used is parsed back from each case's
``fvSchemes`` file, so no run is labelled by intention alone.

Ten operating cases vary the inlet velocity and the molecular viscosity, giving
bulk Reynolds numbers from 5.0e4 to 2.0e5, and vary the inlet turbulence
intensity. Each operating case contributes six runs: the two schemes on a coarse
and a fine mesh, plus a finer reference and a separately checked reference.

A run only counts as turbulent if the model says so, so the audit measures the
ratio of maximum eddy viscosity to molecular viscosity in every case and gates on
it. Smoke mode executes one operating case and never returns a scientific
verdict, because grouped evaluation is impossible with a single group.

The mesh ladder was fixed by a smoke-mode design check before any scientific
run: the candidate meshes must be coarse enough, and the reference fine enough,
that the reference disagrees with its own check by well under the candidate
error. No mesh, threshold or operating case is changed after the full run.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import openfoam_3d_audit as OF3

PROTOCOL_ID = "openfoam-turbulent-rans-3d-v1"
SOLVER = "simpleFoam"
TURBULENCE_MODEL = "kOmegaSST"
IMAGE_REFERENCE = OF3.IMAGE_REFERENCE
EXPECTED_IMAGE_ID = OF3.EXPECTED_IMAGE_ID

SCHEMES = {
    "upwind": "bounded Gauss upwind",
    "linearUpwind": "bounded Gauss linearUpwind grad(U)",
}
DUCT_LENGTH, DUCT_HEIGHT, DUCT_WIDTH = 4.0, 1.0, 1.0
C_MU = 0.09
TURB_LENGTH_SCALE = 0.07 * DUCT_HEIGHT
END_TIME = 400
MIN_NUT_RATIO = 2.0


@dataclass(frozen=True)
class Mesh:
    name: str
    nx: int
    ny: int
    nz: int


@dataclass(frozen=True)
class Op:
    case_id: str
    velocity: float
    nu: float
    intensity: float

    @property
    def reynolds(self) -> float:
        return self.velocity * DUCT_HEIGHT / self.nu

    @property
    def k_inlet(self) -> float:
        return 1.5 * (self.intensity * self.velocity) ** 2

    @property
    def omega_inlet(self) -> float:
        return math.sqrt(self.k_inlet) / (C_MU ** 0.25 * TURB_LENGTH_SCALE)


@dataclass(frozen=True)
class Run:
    op: Op
    mesh: Mesh
    scheme_key: str
    role: str


OPS = (
    Op("op00", 1.0, 2.0e-5, 0.05), Op("op01", 1.0, 1.0e-5, 0.05),
    Op("op02", 1.2, 2.0e-5, 0.04), Op("op03", 1.2, 1.2e-5, 0.06),
    Op("op04", 1.5, 2.0e-5, 0.05), Op("op05", 1.5, 1.0e-5, 0.03),
    Op("op06", 0.8, 1.6e-5, 0.05), Op("op07", 0.8, 1.0e-5, 0.07),
    Op("op08", 2.0, 2.0e-5, 0.04), Op("op09", 2.0, 1.2e-5, 0.05),
)
FULL_MESHES = {
    "coarse": Mesh("coarse", 20, 10, 10), "fine": Mesh("fine", 28, 14, 14),
    "reference": Mesh("reference", 80, 40, 40),
    "reference_check": Mesh("reference_check", 96, 48, 48),
}
SMOKE_MESHES = {
    "coarse": Mesh("coarse", 16, 8, 8), "fine": Mesh("fine", 24, 12, 12),
    "reference": Mesh("reference", 48, 24, 24),
    "reference_check": Mesh("reference_check", 60, 30, 30),
}
FEATURE_NAMES = (
    "mean_abs", "rms", "max_abs", "mean_signed", "near_wall_mean", "core_mean",
    "station_25", "station_50", "station_75", "wall_normal_gradient", "spanwise_gradient",
)
GATES = (
    OF3.Gate("all 60 turbulent cases completed", "solver_failure_count", "le", upper=0.0),
    OF3.Gate("every parsed fvSchemes receipt matches", "receipt_failure_count", "le", upper=0.0),
    OF3.Gate("turbulence model is genuinely active", "minimum_nut_ratio", "ge", lower=MIN_NUT_RATIO),
    OF3.Gate("scheme contrast changes every paired field", "minimum_scheme_delta", "ge", lower=1.0e-6),
    OF3.Gate("grouped leave-one-operating-case-out accuracy", "grouped_accuracy", "ge", lower=0.80),
    OF3.Gate("coarse-library to fine-field transfer", "coarse_to_fine_accuracy", "ge", lower=0.75),
    OF3.Gate("fine-library to coarse-field transfer", "fine_to_coarse_accuracy", "ge", lower=0.75),
    OF3.Gate("duplicate-field negative control is chance", "negative_control_accuracy", "between", 0.49, 0.51),
    OF3.Gate("reference ladder median disagreement", "reference_median_relative_l2", "le", upper=0.02),
    OF3.Gate("reference ladder maximum disagreement", "reference_max_relative_l2", "le", upper=0.05),
    OF3.Gate("reference error is below candidate error", "reference_to_candidate_error_ratio", "le", upper=0.50),
)


def _hdr(cls: str, obj: str) -> str:
    return OF3._foam_header(cls, obj)


def case_files(run: Run) -> dict[str, str]:
    op, mesh = run.op, run.mesh
    k, w = op.k_inlet, op.omega_inlet
    block = _hdr("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
    (0 0 0) ({DUCT_LENGTH:g} 0 0) ({DUCT_LENGTH:g} {DUCT_HEIGHT:g} 0) (0 {DUCT_HEIGHT:g} 0)
    (0 0 {DUCT_WIDTH:g}) ({DUCT_LENGTH:g} 0 {DUCT_WIDTH:g}) ({DUCT_LENGTH:g} {DUCT_HEIGHT:g} {DUCT_WIDTH:g}) (0 {DUCT_HEIGHT:g} {DUCT_WIDTH:g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({mesh.nx} {mesh.ny} {mesh.nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  {{ type patch; faces ( (0 4 7 3) ); }}
    outlet {{ type patch; faces ( (1 2 6 5) ); }}
    walls  {{ type wall;  faces ( (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }}
);
mergePatchPairs ();
"""
    U = _hdr("volVectorField", "U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({op.velocity:.10g} 0 0);
boundaryField
{{
    inlet  {{ type fixedValue; value uniform ({op.velocity:.10g} 0 0); }}
    outlet {{ type zeroGradient; }}
    walls  {{ type noSlip; }}
}}
"""
    p = _hdr("volScalarField", "p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls  { type zeroGradient; }
}
"""
    kf = _hdr("volScalarField", "k") + f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k:.10g};
boundaryField
{{
    inlet  {{ type fixedValue; value uniform {k:.10g}; }}
    outlet {{ type zeroGradient; }}
    walls  {{ type kqRWallFunction; value uniform {k:.10g}; }}
}}
"""
    om = _hdr("volScalarField", "omega") + f"""
dimensions [0 0 -1 0 0 0 0];
internalField uniform {w:.10g};
boundaryField
{{
    inlet  {{ type fixedValue; value uniform {w:.10g}; }}
    outlet {{ type zeroGradient; }}
    walls  {{ type omegaWallFunction; value uniform {w:.10g}; }}
}}
"""
    nut = _hdr("volScalarField", "nut") + """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    walls  { type nutkWallFunction; value uniform 0; }
}
"""
    transport = _hdr("dictionary", "transportProperties") + f"""
transportModel Newtonian;
nu {op.nu:.10g};
"""
    turb = _hdr("dictionary", "turbulenceProperties") + f"""
simulationType RAS;
RAS {{ RASModel {TURBULENCE_MODEL}; turbulence on; printCoeffs off; }}
"""
    control = _hdr("dictionary", "controlDict") + f"""
application {SOLVER};
startFrom startTime; startTime 0;
stopAt endTime; endTime {END_TIME};
deltaT 1; writeControl timeStep; writeInterval {END_TIME};
purgeWrite 0; writeFormat ascii; writePrecision 12;
writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable false;
"""
    schemes = _hdr("dictionary", "fvSchemes") + f"""
ddtSchemes {{ default steadyState; }}
gradSchemes {{ default Gauss linear; }}
divSchemes
{{
    default none;
    div(phi,U) {SCHEMES[run.scheme_key]};
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
"""
    solution = _hdr("dictionary", "fvSolution") + """
solvers
{
    p { solver GAMG; tolerance 1e-08; relTol 0.01; smoother GaussSeidel; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; }
}
relaxationFactors { equations { p 0.7; U 0.9; ".*" 0.9; } }
"""
    return {
        "system/blockMeshDict": block, "system/controlDict": control,
        "system/fvSchemes": schemes, "system/fvSolution": solution,
        "constant/transportProperties": transport,
        "constant/turbulenceProperties": turb,
        "0/U": U, "0/p": p, "0/k": kf, "0/omega": om, "0/nut": nut,
    }


def build_case(case_dir: Path, run: Run) -> None:
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for rel, text in case_files(run).items():
        target = case_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def execute_case(case_dir: Path, timeout: float = 1800.0) -> dict[str, object]:
    """Execute one generated turbulent case in the digest-pinned container."""
    case_dir = Path(case_dir).resolve()
    script = f"""source /opt/OpenFOAM-v2506/etc/bashrc
set -euo pipefail
cd /case
blockMesh > log.blockMesh 2>&1
foamDictionary system/fvSchemes -entry divSchemes -value > effectiveDivSchemes.txt
{SOLVER} > log.{SOLVER} 2>&1
postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1
"""
    command = ["docker", "run", "--rm", "--entrypoint", "/bin/bash",
               "-v", f"{case_dir}:/case", IMAGE_REFERENCE, "--noprofile", "--norc", "-c", script]
    started = time.monotonic()
    completed = OF3._run_command(command, timeout)
    log = case_dir / f"log.{SOLVER}"
    text = log.read_text(errors="replace") if log.exists() else ""
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "converged": "SIMPLE solution converged" in text,
        "container_output": completed.stdout[-1200:],
    }


def receipt_matches(case_dir: Path, run: Run) -> bool:
    path = Path(case_dir) / "effectiveDivSchemes.txt"
    if not path.exists():
        return False
    text = " ".join(path.read_text(errors="replace").split())
    expected = " ".join(f"div(phi,U) {SCHEMES[run.scheme_key]}".split())
    return expected in text


def load_speed_field(case_dir: Path) -> OF3.StructuredField:
    latest = OF3._latest_time_dir(Path(case_dir))
    centres = OF3.parse_internal_vector(latest / "C")
    velocity = OF3.parse_internal_vector(latest / "U")
    return OF3.structured_field(centres, np.linalg.norm(velocity, axis=1))


def nut_ratio(case_dir: Path, op: Op) -> float:
    latest = OF3._latest_time_dir(Path(case_dir))
    nut = OF3.parse_internal_scalar(latest / "nut")
    return float(np.max(nut) / op.nu)


def signature(candidate: OF3.StructuredField, reference: OF3.StructuredField, op: Op) -> np.ndarray:
    truth = OF3.interpolate_field(reference, candidate)
    error = (candidate.values - truth) / op.velocity
    ny = candidate.y.size
    near = np.zeros(ny, dtype=bool)
    near[: max(1, ny // 5)] = True
    near[-max(1, ny // 5):] = True
    core = ~near
    feats = [
        np.mean(np.abs(error)), np.sqrt(np.mean(error ** 2)), np.max(np.abs(error)),
        np.mean(error), np.mean(np.abs(error[:, near, :])), np.mean(np.abs(error[:, core, :])),
    ]
    for fraction in (0.25, 0.50, 0.75):
        index = int(np.argmin(np.abs(candidate.x / DUCT_LENGTH - fraction)))
        feats.append(np.mean(np.abs(error[index])))
    dy = np.gradient(candidate.values, candidate.y, axis=1) - np.gradient(truth, candidate.y, axis=1)
    dz = np.gradient(candidate.values, candidate.z, axis=2) - np.gradient(truth, candidate.z, axis=2)
    feats.append(np.mean(np.abs(dy)) / op.velocity)
    feats.append(np.mean(np.abs(dz)) / op.velocity)
    out = np.asarray(feats, dtype=float)
    if out.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(out)):
        raise ValueError("turbulent signature is non-finite or wrongly shaped")
    return out


def relative_l2(a: OF3.StructuredField, b: OF3.StructuredField) -> float:
    other = OF3.interpolate_field(b, a)
    return float(np.linalg.norm(a.values - other) / np.linalg.norm(a.values))


def run_audit(mode: str, work_dir: Path, timeout_per_case: float = 1800.0) -> dict[str, object]:
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    meshes = FULL_MESHES if mode == "full" else SMOKE_MESHES
    ops = OPS if mode == "full" else OPS[:1]
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    runtime = OF3.inspect_runtime()
    rows, refs, nut_ratios, deltas, failures, receipts = [], [], [], [], 0, 0
    ref_ratio = []
    for op in ops:
        specs = [
            Run(op, meshes["coarse"], "upwind", "upwind_coarse"),
            Run(op, meshes["coarse"], "linearUpwind", "linearUpwind_coarse"),
            Run(op, meshes["fine"], "upwind", "upwind_fine"),
            Run(op, meshes["fine"], "linearUpwind", "linearUpwind_fine"),
            Run(op, meshes["reference"], "linearUpwind", "reference"),
            Run(op, meshes["reference_check"], "linearUpwind", "reference_check"),
        ]
        fields = {}
        for spec in specs:
            case_dir = work_dir / f"{op.case_id}_{spec.role}"
            build_case(case_dir, spec)
            result = execute_case(case_dir, timeout_per_case)
            ok = result["returncode"] == 0 and result["converged"]
            if not ok:
                failures += 1
                print(f"  [FAIL] {op.case_id} {spec.role}: rc={result['returncode']} conv={result['converged']}")
                continue
            if not receipt_matches(case_dir, spec):
                receipts += 1
            fields[spec.role] = load_speed_field(case_dir)
            nut_ratios.append(nut_ratio(case_dir, op))
        if len(fields) < 6:
            continue
        rl = relative_l2(fields["reference"], fields["reference_check"])
        refs.append(rl)
        ref_err = relative_l2(fields["reference"], fields["reference_check"])
        cand_err = relative_l2(fields["linearUpwind_fine"], fields["reference"])
        ref_ratio.append(ref_err / cand_err if cand_err > 0 else np.inf)
        for mesh_name in ("coarse", "fine"):
            up, lin = fields["upwind_" + mesh_name], fields["linearUpwind_" + mesh_name]
            deltas.append(relative_l2(up, lin))
            rows.append({"group": op.case_id, "mesh": mesh_name, "scheme": "upwind", "label": 0,
                         "features": signature(up, fields["reference"], op).tolist()})
            rows.append({"group": op.case_id, "mesh": mesh_name, "scheme": "linearUpwind", "label": 1,
                         "features": signature(lin, fields["reference"], op).tolist()})
        print(f"  [ok] {op.case_id} Re={op.reynolds:.3g} nut/nu_max={max(nut_ratios):.1f} refL2={rl:.2e}")

    metrics = {
        "solver_failure_count": float(failures),
        "receipt_failure_count": float(receipts),
        "minimum_nut_ratio": float(np.min(nut_ratios)) if nut_ratios else float("nan"),
        "minimum_scheme_delta": float(np.min(deltas)) if deltas else float("nan"),
        "grouped_accuracy": OF3.grouped_accuracy(rows) if mode == "full" else float("nan"),
        "coarse_to_fine_accuracy": OF3.grouped_accuracy(rows, train_mesh="coarse") if mode == "full" else float("nan"),
        "fine_to_coarse_accuracy": OF3.grouped_accuracy(rows, train_mesh="fine") if mode == "full" else float("nan"),
        "reference_median_relative_l2": float(np.median(refs)) if refs else float("nan"),
        "reference_max_relative_l2": float(np.max(refs)) if refs else float("nan"),
        "reference_to_candidate_error_ratio": float(np.median(ref_ratio)) if ref_ratio else float("nan"),
        "completed_case_count": float(len(ops) * 6 - failures),
    }
    if mode == "full":
        neg = [dict(r) for r in rows]
        for r in neg:
            if r["scheme"] == "linearUpwind":
                twin = next(x for x in rows if x["group"] == r["group"] and x["mesh"] == r["mesh"]
                            and x["scheme"] == "upwind")
                r["features"] = list(twin["features"])
        metrics["negative_control_accuracy"] = OF3.grouped_accuracy(neg)
    else:
        metrics["negative_control_accuracy"] = float("nan")

    gates = []
    if mode == "full":
        for gate in GATES:
            value = metrics[gate.metric]
            value = float(value)
            if gate.comparison == "ge":
                passed = np.isfinite(value) and value >= float(gate.lower)
            elif gate.comparison == "le":
                passed = np.isfinite(value) and value <= float(gate.upper)
            elif gate.comparison == "between":
                passed = np.isfinite(value) and float(gate.lower) <= value <= float(gate.upper)
            else:
                raise ValueError(f"unknown comparison {gate.comparison!r}")
            gates.append({"gate": {"name": gate.name, "metric": gate.metric,
                                   "comparison": gate.comparison, "lower": gate.lower,
                                   "upper": gate.upper},
                          "value": value, "passed": bool(passed)})
        verdict = "GO" if all(g["passed"] for g in gates) else "FAIL"
    else:
        verdict = "SMOKE_PASS" if failures == 0 and receipts == 0 else "SMOKE_FAIL"
    return {"manifest": {"protocol_id": PROTOCOL_ID, "mode": mode, "solver": SOLVER,
                         "turbulence_model": TURBULENCE_MODEL, "schemes": SCHEMES,
                         "image_reference": IMAGE_REFERENCE, "runtime": runtime,
                         "meshes": {k: vars(v) for k, v in meshes.items()},
                         "operating_cases": [vars(o) | {"reynolds": o.reynolds} for o in ops]},
            "metrics": metrics, "gates": gates, "verdict": verdict}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/openfoam_turbulent_audit"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/openfoam_turbulent_audit.json"))
    ap.add_argument("--timeout-per-case", type=float, default=1800.0)
    a = ap.parse_args(argv)
    report = run_audit(a.mode, a.work_dir, a.timeout_per_case)
    print(f"protocol: {report['manifest']['protocol_id']} ({a.mode})")
    print(f"solver: {SOLVER} / {TURBULENCE_MODEL}")
    for k, v in report["metrics"].items():
        print(f"{k}: {v}")
    for g in report["gates"]:
        print(f"[{'PASS' if g['passed'] else 'FAIL'}] {g['gate']['name']}: {g['value']}")
    print(f"verdict: {report['verdict']}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(OF3._json_compatible(report), indent=2, sort_keys=True) + "\n")
    print(f"report: {a.output}")
    return 0 if report["verdict"] in {"GO", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
