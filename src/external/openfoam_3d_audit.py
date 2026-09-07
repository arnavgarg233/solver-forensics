#!/usr/bin/env python3
"""Frozen 3D external-solver audit using OpenFOAM scalarTransportFoam.

The full protocol contains 10 operating groups and six OpenFOAM runs per
operating group, for 60 solver cases.  It compares the fvSchemes entries
``Gauss upwind`` and ``Gauss linearUpwind grad(T)`` on two controlled meshes.
Two finer linearUpwind runs provide a separately checked reference ladder.
The scalar inlet is split at z=0.5 and forced to two different values, so a
run is invalid if the resulting volume field is invariant in z.

Smoke mode executes one small six-run group.  It checks the external solver,
mesh, scheme receipts, reference ordering, and genuine z variation, but it
never returns a scientific GO verdict because grouped evaluation is impossible
with one operating group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


PROTOCOL_ID = "openfoam-scalar-transport-3d-v1"
IMAGE_NAME = "openeuler/openfoam"
IMAGE_DIGEST = "sha256:a9e2ee499bca06b43bfe330de779e675c496687420a4ab3bec2316f20a693d4a"
IMAGE_REFERENCE = f"{IMAGE_NAME}@{IMAGE_DIGEST}"
EXPECTED_IMAGE_ID = "sha256:a7e4fa3dc52cfb323a4079beab1969dff013c8e195d76e0d21bc778a181147d2"
EXPECTED_OPENFOAM_VERSION = "v2506"
EXPECTED_OPENFOAM_BUILD = "_615aae61d7-20250627"
SOLVER = "scalarTransportFoam"
DUCT_LENGTH = 4.0
DUCT_WIDTH = 1.0
DUCT_HEIGHT = 1.0
CFL = 0.40
SCHEMES = {
    "upwind": "Gauss upwind",
    "linearUpwind": "Gauss linearUpwind grad(T)",
}
FEATURE_NAMES = (
    "relative_l1",
    "relative_l2",
    "relative_linf",
    "signed_mean",
    "lower_half_bias",
    "upper_half_bias",
    "z_contrast_error",
    "quarter_duct_l1",
    "mid_duct_l1",
    "three_quarter_duct_l1",
    "z_gradient_l1",
    "range_excursion",
)


@dataclass(frozen=True)
class Gate:
    name: str
    metric: str
    comparison: str
    lower: float | None = None
    upper: float | None = None


# Frozen before any audit result is computed.  Smoke mode reports these metrics
# but does not apply a GO verdict to its one-group diagnostic run.
GO_CRITERIA = (
    Gate("all 60 OpenFOAM cases completed", "solver_health_failure_count", "le", upper=0.0),
    Gate("every parsed fvSchemes receipt matches", "scheme_receipt_failure_count", "le", upper=0.0),
    Gate("all fields are genuinely z-dependent", "minimum_z_dependence", "ge", lower=0.02),
    Gate("scheme contrast changes every paired field", "minimum_scheme_field_delta", "ge", lower=1.0e-6),
    Gate("grouped leave-one-operating-case-out accuracy", "grouped_accuracy", "ge", lower=0.80),
    Gate("coarse-library to fine-field transfer", "coarse_to_fine_accuracy", "ge", lower=0.75),
    Gate("fine-library to coarse-field transfer", "fine_to_coarse_accuracy", "ge", lower=0.75),
    Gate("duplicate-field negative control is chance", "negative_control_accuracy", "between", 0.49, 0.51),
    Gate("reference ladder median disagreement", "reference_median_relative_l2", "le", upper=0.02),
    Gate("reference ladder maximum disagreement", "reference_max_relative_l2", "le", upper=0.05),
    Gate("reference error is below candidate error", "reference_to_candidate_error_ratio", "le", upper=0.50),
)


@dataclass(frozen=True)
class OperatingCase:
    case_id: str
    velocity: float
    diffusivity: float
    lower_inlet: float
    upper_inlet: float

    @property
    def forcing_span(self) -> float:
        return abs(self.upper_inlet - self.lower_inlet)

    def validate(self) -> None:
        if self.velocity <= 0.0 or self.diffusivity <= 0.0:
            raise ValueError("velocity and diffusivity must be positive")
        if self.forcing_span <= 0.0:
            raise ValueError("the two z inlet patches must have different forcing")


OPERATING_CASES = (
    OperatingCase("op00", 0.80, 0.0040, 0.00, 1.00),
    OperatingCase("op01", 0.80, 0.0080, 0.10, 1.10),
    OperatingCase("op02", 0.80, 0.0120, -0.10, 0.90),
    OperatingCase("op03", 1.00, 0.0050, 0.00, 0.80),
    OperatingCase("op04", 1.00, 0.0100, 0.20, 1.20),
    OperatingCase("op05", 1.00, 0.0150, -0.20, 1.00),
    OperatingCase("op06", 1.20, 0.0060, 0.00, 1.20),
    OperatingCase("op07", 1.20, 0.0120, 0.15, 1.05),
    OperatingCase("op08", 1.40, 0.0070, -0.10, 1.10),
    OperatingCase("op09", 1.40, 0.0140, 0.05, 0.95),
)


@dataclass(frozen=True)
class MeshSpec:
    name: str
    nx: int
    ny: int
    nz: int

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nz) < 2 or self.nz % 2:
            raise ValueError("mesh counts must be at least two and nz must be even")
        if self.nx != 4 * self.ny or self.ny != self.nz:
            raise ValueError("the frozen meshes use cubic cells in the 4:1:1 duct")


FULL_MESHES = {
    "coarse": MeshSpec("coarse", 24, 6, 6),
    "fine": MeshSpec("fine", 32, 8, 8),
    "reference": MeshSpec("reference", 48, 12, 12),
    "reference_check": MeshSpec("reference_check", 64, 16, 16),
}
SMOKE_MESHES = {
    "coarse": MeshSpec("coarse", 8, 2, 2),
    "fine": MeshSpec("fine", 16, 4, 4),
    "reference": MeshSpec("reference", 24, 6, 6),
    "reference_check": MeshSpec("reference_check", 32, 8, 8),
}


@dataclass(frozen=True)
class RunSpec:
    operating: OperatingCase
    mesh: MeshSpec
    scheme_key: str
    role: str

    @property
    def run_id(self) -> str:
        return f"{self.operating.case_id}__{self.role}"


@dataclass(frozen=True)
class StructuredField:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values: np.ndarray


def criteria_sha256() -> str:
    payload = [asdict(gate) for gate in GO_CRITERIA]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def protocol_manifest(mode: str = "full") -> dict[str, object]:
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    meshes = FULL_MESHES if mode == "full" else SMOKE_MESHES
    operating = OPERATING_CASES if mode == "full" else OPERATING_CASES[:1]
    for item in operating:
        item.validate()
    for item in meshes.values():
        item.validate()
    runs: list[RunSpec] = []
    for op in operating:
        runs.extend((
            RunSpec(op, meshes["coarse"], "upwind", "upwind_coarse"),
            RunSpec(op, meshes["coarse"], "linearUpwind", "linearUpwind_coarse"),
            RunSpec(op, meshes["fine"], "upwind", "upwind_fine"),
            RunSpec(op, meshes["fine"], "linearUpwind", "linearUpwind_fine"),
            RunSpec(op, meshes["reference"], "linearUpwind", "reference"),
            RunSpec(op, meshes["reference_check"], "linearUpwind", "reference_check"),
        ))
    return {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "image_reference": IMAGE_REFERENCE,
        "expected_image_id": EXPECTED_IMAGE_ID,
        "expected_openfoam_version": EXPECTED_OPENFOAM_VERSION,
        "expected_openfoam_build": EXPECTED_OPENFOAM_BUILD,
        "solver": SOLVER,
        "solver_equation": "fvm::ddt(T) + fvm::div(phi,T) - fvm::laplacian(DT,T)",
        "schemes": dict(SCHEMES),
        "operating_cases": [asdict(item) for item in operating],
        "meshes": {key: asdict(value) for key, value in meshes.items()},
        "runs": runs,
        "run_count": len(runs),
        "feature_names": FEATURE_NAMES,
        "criteria_sha256": criteria_sha256(),
        "go_criteria": [asdict(gate) for gate in GO_CRITERIA],
        "positive_control": "upwind versus linearUpwind grouped by operating case",
        "negative_control": "duplicate upwind signatures assigned to both labels",
        "grouping_rule": "hold out every mesh and scheme row from one operating case together",
        "reference_rule": "linearUpwind on 48x12x12 checked against 64x16x16",
    }


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {class_name};\n    object {object_name};\n}}\n"
    )


def _block_mesh_dict(mesh: MeshSpec) -> str:
    half_nz = mesh.nz // 2
    return _foam_header("dictionary", "blockMeshDict") + f"""
convertToMeters 1;
vertices
(
    (0 0 0) ({DUCT_LENGTH:g} 0 0) ({DUCT_LENGTH:g} {DUCT_WIDTH:g} 0) (0 {DUCT_WIDTH:g} 0)
    (0 0 0.5) ({DUCT_LENGTH:g} 0 0.5) ({DUCT_LENGTH:g} {DUCT_WIDTH:g} 0.5) (0 {DUCT_WIDTH:g} 0.5)
    (0 0 {DUCT_HEIGHT:g}) ({DUCT_LENGTH:g} 0 {DUCT_HEIGHT:g})
    ({DUCT_LENGTH:g} {DUCT_WIDTH:g} {DUCT_HEIGHT:g}) (0 {DUCT_WIDTH:g} {DUCT_HEIGHT:g})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({mesh.nx} {mesh.ny} {half_nz}) simpleGrading (1 1 1)
    hex (4 5 6 7 8 9 10 11) ({mesh.nx} {mesh.ny} {half_nz}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inletLower {{ type patch; faces ((0 4 7 3)); }}
    inletUpper {{ type patch; faces ((4 8 11 7)); }}
    outlet {{ type patch; faces ((1 2 6 5) (5 6 10 9)); }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4) (4 5 9 8)
            (3 7 6 2) (7 11 10 6)
            (0 3 2 1) (8 9 10 11)
        );
    }}
);
mergePatchPairs ();
"""


def _vector_field(op: OperatingCase) -> str:
    velocity = f"({op.velocity:.12g} 0 0)"
    return _foam_header("volVectorField", "U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform {velocity};
boundaryField
{{
    inletLower {{ type fixedValue; value uniform {velocity}; }}
    inletUpper {{ type fixedValue; value uniform {velocity}; }}
    outlet {{ type zeroGradient; }}
    walls {{ type fixedValue; value uniform (0 0 0); }}
}}
"""


def _scalar_field(op: OperatingCase) -> str:
    initial = 0.5 * (op.lower_inlet + op.upper_inlet)
    return _foam_header("volScalarField", "T") + f"""
dimensions [0 0 0 0 0 0 0];
internalField uniform {initial:.12g};
boundaryField
{{
    inletLower {{ type fixedValue; value uniform {op.lower_inlet:.12g}; }}
    inletUpper {{ type fixedValue; value uniform {op.upper_inlet:.12g}; }}
    outlet {{ type zeroGradient; }}
    walls {{ type zeroGradient; }}
}}
"""


def _control_dict(spec: RunSpec) -> tuple[str, float, int]:
    dx = DUCT_LENGTH / spec.mesh.nx
    nominal_dt = CFL * dx / spec.operating.velocity
    target_time = 1.10 * DUCT_LENGTH / spec.operating.velocity
    n_steps = int(math.ceil(target_time / nominal_dt))
    end_time = target_time
    dt = end_time / n_steps
    text = _foam_header("dictionary", "controlDict") + f"""
application {SOLVER};
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end_time:.12g};
deltaT {dt:.12g};
writeControl runTime;
writeInterval {end_time:.12g};
purgeWrite 0;
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
functions {{}}
"""
    return text, dt, n_steps


def _fv_schemes(scheme_key: str) -> str:
    if scheme_key not in SCHEMES:
        raise ValueError(f"unknown convection scheme {scheme_key!r}")
    scheme = SCHEMES[scheme_key]
    return _foam_header("dictionary", "fvSchemes") + f"""
ddtSchemes {{ default Euler; }}
gradSchemes
{{
    default Gauss linear;
    grad(T) Gauss linear;
}}
divSchemes
{{
    default none;
    div(phi,T) {scheme};
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
"""


def _fv_solution() -> str:
    return _foam_header("dictionary", "fvSolution") + """
solvers
{
    T
    {
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-10;
        relTol 0;
        maxIter 1000;
    }
}
relaxationFactors { equations { T 1; } }
"""


def build_case(case_dir: str | Path, spec: RunSpec) -> dict[str, object]:
    """Generate a complete scalarTransportFoam case without running Docker."""
    case_dir = Path(case_dir)
    for relative in ("0", "constant", "system"):
        (case_dir / relative).mkdir(parents=True, exist_ok=True)
    control, dt, n_steps = _control_dict(spec)
    files = {
        "0/U": _vector_field(spec.operating),
        "0/T": _scalar_field(spec.operating),
        "constant/transportProperties": _foam_header("dictionary", "transportProperties")
        + f"\nDT [0 2 -1 0 0 0 0] {spec.operating.diffusivity:.12g};\n",
        "system/blockMeshDict": _block_mesh_dict(spec.mesh),
        "system/controlDict": control,
        "system/fvSchemes": _fv_schemes(spec.scheme_key),
        "system/fvSolution": _fv_solution(),
    }
    for relative, text in files.items():
        (case_dir / relative).write_text(text, encoding="utf-8")
    metadata = {
        "run_id": spec.run_id,
        "operating": asdict(spec.operating),
        "mesh": asdict(spec.mesh),
        "role": spec.role,
        "scheme_key": spec.scheme_key,
        "fv_scheme": SCHEMES[spec.scheme_key],
        "delta_t": dt,
        "n_steps": n_steps,
        "end_time": dt * n_steps,
        "max_nominal_courant": spec.operating.velocity * dt / (DUCT_LENGTH / spec.mesh.nx),
    }
    (case_dir / "audit_case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _run_command(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False,
    )


def inspect_runtime(timeout: float = 120.0) -> dict[str, str]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker executable not found")
    inspected = _run_command(
        ["docker", "image", "inspect", IMAGE_REFERENCE, "--format", "{{json .}}"], timeout
    )
    if inspected.returncode:
        raise RuntimeError(f"cannot inspect pinned image: {inspected.stdout.strip()}")
    image = json.loads(inspected.stdout)
    architecture = image.get("Architecture", "")
    image_id = image.get("Id", "")
    if architecture != "arm64":
        raise RuntimeError(f"pinned image is not arm64: {architecture!r}")
    if image_id != EXPECTED_IMAGE_ID:
        raise RuntimeError(f"image ID mismatch: expected {EXPECTED_IMAGE_ID}, found {image_id}")
    probe = _run_command([
        "docker", "run", "--rm", "--entrypoint", "/bin/bash", IMAGE_REFERENCE,
        "--noprofile", "--norc", "-c",
        "source /opt/OpenFOAM-v2506/etc/bashrc; scalarTransportFoam -help",
    ], timeout)
    version = re.search(r"Using:\s+OpenFOAM-(v\d+)", probe.stdout)
    build = re.search(r"Build:\s+(\S+)", probe.stdout)
    if probe.returncode or not version or not build:
        raise RuntimeError(f"cannot probe OpenFOAM runtime: {probe.stdout.strip()}")
    if version.group(1) != EXPECTED_OPENFOAM_VERSION or build.group(1) != EXPECTED_OPENFOAM_BUILD:
        raise RuntimeError(
            f"OpenFOAM build mismatch: found {version.group(1)} {build.group(1)}"
        )
    return {
        "image_reference": IMAGE_REFERENCE,
        "image_id": image_id,
        "architecture": architecture,
        "openfoam_version": version.group(1),
        "openfoam_build": build.group(1),
        "solver": SOLVER,
    }


def execute_case(case_dir: str | Path, timeout: float = 900.0) -> dict[str, object]:
    """Execute one generated case in the digest-pinned OpenFOAM container."""
    case_dir = Path(case_dir).resolve()
    script = """source /opt/OpenFOAM-v2506/etc/bashrc
set -euo pipefail
cd /case
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
foamDictionary system/fvSchemes -entry divSchemes -value > effectiveDivSchemes.txt
scalarTransportFoam > log.scalarTransportFoam 2>&1
postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1
"""
    command = [
        "docker", "run", "--rm", "--entrypoint", "/bin/bash",
        "-v", f"{case_dir}:/case", IMAGE_REFERENCE,
        "--noprofile", "--norc", "-c", script,
    ]
    started = time.monotonic()
    completed = _run_command(command, timeout)
    elapsed = time.monotonic() - started
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "container_output": completed.stdout,
    }


def _latest_time_dir(case_dir: Path) -> Path:
    candidates = []
    for child in case_dir.iterdir():
        if child.is_dir():
            try:
                value = float(child.name)
            except ValueError:
                continue
            if value > 0.0:
                candidates.append((value, child))
    if not candidates:
        raise ValueError("no positive OpenFOAM time directory was written")
    return max(candidates, key=lambda item: item[0])[1]


def parse_internal_scalar(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text, flags=re.DOTALL,
    )
    if not match:
        uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if uniform:
            return np.asarray([float(uniform.group(1))])
        raise ValueError(f"cannot parse scalar internalField from {path}")
    values = np.fromstring(match.group(2), sep=" ")
    if values.size != int(match.group(1)):
        raise ValueError(f"scalar count mismatch in {path}")
    return values


def parse_internal_vector(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text, flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"cannot parse vector internalField from {path}")
    rows = re.findall(r"\(([^()]+)\)", match.group(2))
    values = np.asarray([[float(token) for token in row.split()] for row in rows])
    if values.shape != (int(match.group(1)), 3):
        raise ValueError(f"vector count mismatch in {path}")
    return values


def structured_field(centres: np.ndarray, values: np.ndarray) -> StructuredField:
    centres = np.asarray(centres, dtype=float)
    values = np.asarray(values, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 3 or values.shape != (centres.shape[0],):
        raise ValueError("centres and values have incompatible shapes")
    rounded = np.round(centres, decimals=10)
    axes = tuple(np.unique(rounded[:, axis]) for axis in range(3))
    expected = int(np.prod([axis.size for axis in axes]))
    if expected != values.size or min(axis.size for axis in axes) < 2:
        raise ValueError("field is not a complete three-dimensional Cartesian grid")
    array = np.full(tuple(axis.size for axis in axes), np.nan)
    indices = tuple(np.searchsorted(axes[axis], rounded[:, axis]) for axis in range(3))
    array[indices] = values
    if not np.all(np.isfinite(array)):
        raise ValueError("structured field contains missing or non-finite cells")
    return StructuredField(axes[0], axes[1], axes[2], array)


def load_case_field(case_dir: str | Path) -> StructuredField:
    latest = _latest_time_dir(Path(case_dir))
    return structured_field(
        parse_internal_vector(latest / "C"), parse_internal_scalar(latest / "T")
    )


def interpolate_field(reference: StructuredField, target: StructuredField) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (reference.x, reference.y, reference.z), reference.values,
        method="linear", bounds_error=True,
    )
    grid = np.meshgrid(target.x, target.y, target.z, indexing="ij")
    points = np.column_stack([item.ravel() for item in grid])
    return np.asarray(interpolator(points)).reshape(target.values.shape)


def field_observations(field: StructuredField, forcing_span: float) -> dict[str, object]:
    if forcing_span <= 0.0:
        raise ValueError("forcing_span must be positive")
    z_profile = np.mean(field.values, axis=(0, 1))
    lower = field.z < 0.5 * DUCT_HEIGHT
    upper = ~lower
    if not np.any(lower) or not np.any(upper):
        raise ValueError("both z-forced halves must contain cells")
    outlet = field.values[-1]
    return {
        "minimum": float(np.min(field.values)),
        "maximum": float(np.max(field.values)),
        "mean": float(np.mean(field.values)),
        "z_dependence": float(np.ptp(z_profile) / forcing_span),
        "z_contrast": float((np.mean(field.values[:, :, upper]) - np.mean(field.values[:, :, lower])) / forcing_span),
        "outlet_mean": float(np.mean(outlet)),
        "outlet_z_contrast": float((np.mean(outlet[:, upper]) - np.mean(outlet[:, lower])) / forcing_span),
        "z_profile": z_profile.tolist(),
    }


def signature(candidate: StructuredField, reference: StructuredField, op: OperatingCase) -> np.ndarray:
    truth = interpolate_field(reference, candidate)
    span = op.forcing_span
    error = (candidate.values - truth) / span
    lower = candidate.z < 0.5 * DUCT_HEIGHT
    upper = ~lower
    features = [
        np.mean(np.abs(error)),
        np.sqrt(np.mean(error**2)),
        np.max(np.abs(error)),
        np.mean(error),
        np.mean(error[:, :, lower]),
        np.mean(error[:, :, upper]),
        (np.mean(candidate.values[:, :, upper]) - np.mean(candidate.values[:, :, lower])
         - np.mean(truth[:, :, upper]) + np.mean(truth[:, :, lower])) / span,
    ]
    for fraction in (0.25, 0.50, 0.75):
        index = int(np.argmin(np.abs(candidate.x / DUCT_LENGTH - fraction)))
        features.append(np.mean(np.abs(error[index])))
    dz = np.gradient(candidate.values, candidate.z, axis=2)
    dz_ref = np.gradient(truth, candidate.z, axis=2)
    features.append(np.mean(np.abs(dz - dz_ref)) / span)
    lo = min(op.lower_inlet, op.upper_inlet)
    hi = max(op.lower_inlet, op.upper_inlet)
    excursion = max(0.0, lo - float(np.min(candidate.values))) + max(
        0.0, float(np.max(candidate.values)) - hi
    )
    features.append(excursion / span)
    result = np.asarray(features, dtype=float)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("signature is non-finite or has the wrong length")
    return result


def _nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    scale = np.std(train_x, axis=0)
    scale[scale < 1.0e-12] = 1.0
    center = np.mean(train_x, axis=0)
    z_train = (train_x - center) / scale
    z_test = (test_x - center) / scale
    centroids = np.vstack([np.mean(z_train[train_y == label], axis=0) for label in (0, 1)])
    distances = np.linalg.norm(z_test[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def grouped_accuracy(rows: Sequence[Mapping[str, object]], train_mesh: str | None = None,
                     test_mesh: str | None = None) -> float:
    groups = sorted({str(row["group"]) for row in rows})
    if len(groups) < 3:
        return math.nan
    correct = total = 0
    for held in groups:
        train = [row for row in rows if row["group"] != held and (train_mesh is None or row["mesh"] == train_mesh)]
        test = [row for row in rows if row["group"] == held and (test_mesh is None or row["mesh"] == test_mesh)]
        if not train or not test or {int(row["label"]) for row in train} != {0, 1}:
            return math.nan
        train_x = np.vstack([np.asarray(row["features"], dtype=float) for row in train])
        train_y = np.asarray([int(row["label"]) for row in train])
        test_x = np.vstack([np.asarray(row["features"], dtype=float) for row in test])
        test_y = np.asarray([int(row["label"]) for row in test])
        prediction = _nearest_centroid(train_x, train_y, test_x)
        correct += int(np.sum(prediction == test_y))
        total += len(test)
    return correct / total


def evaluate_go(metrics: Mapping[str, float]) -> list[dict[str, object]]:
    evaluations = []
    for gate in GO_CRITERIA:
        value = float(metrics.get(gate.metric, math.nan))
        if gate.comparison == "ge":
            passed = np.isfinite(value) and value >= float(gate.lower)
        elif gate.comparison == "le":
            passed = np.isfinite(value) and value <= float(gate.upper)
        elif gate.comparison == "between":
            passed = np.isfinite(value) and float(gate.lower) <= value <= float(gate.upper)
        else:
            raise ValueError(f"unknown comparison {gate.comparison!r}")
        evaluations.append({"gate": asdict(gate), "value": value, "passed": bool(passed)})
    return evaluations


def _solver_health(case_dir: Path, metadata: Mapping[str, object], execution: Mapping[str, object],
                   field: StructuredField | None) -> tuple[list[str], str]:
    failures = []
    if int(execution["returncode"]) != 0:
        failures.append(f"container return code {execution['returncode']}")
    mesh_log = (case_dir / "log.checkMesh").read_text(encoding="utf-8") if (case_dir / "log.checkMesh").exists() else ""
    solver_log = (case_dir / "log.scalarTransportFoam").read_text(encoding="utf-8") if (case_dir / "log.scalarTransportFoam").exists() else ""
    receipt = (case_dir / "effectiveDivSchemes.txt").read_text(encoding="utf-8") if (case_dir / "effectiveDivSchemes.txt").exists() else ""
    if "Mesh OK." not in mesh_log:
        failures.append("checkMesh did not report Mesh OK")
    if "End" not in solver_log or "FOAM FATAL" in solver_log or re.search(r"\bnan\b", solver_log, re.I):
        failures.append("solver log lacks a clean End marker or contains a fatal/non-finite marker")
    residuals = [float(value) for value in re.findall(r"Final residual = ([0-9.eE+-]+)", solver_log)]
    if not residuals or max(residuals) > 1.0e-8:
        failures.append("linear solver final residual exceeded 1e-8 or was absent")
    expected = str(metadata["fv_scheme"])
    compact_receipt = " ".join(receipt.split())
    if expected not in compact_receipt or "div(phi,T)" not in compact_receipt:
        failures.append("effective fvSchemes receipt does not contain the requested div(phi,T) scheme")
    if field is None or not np.all(np.isfinite(field.values)):
        failures.append("finite three-dimensional T field was not parsed")
    return failures, compact_receipt


def run_audit(mode: str, work_dir: str | Path, timeout_per_case: float = 900.0) -> dict[str, object]:
    """Run smoke or the frozen 60-case full protocol and return a JSON-ready report."""
    manifest = protocol_manifest(mode)
    runtime = inspect_runtime()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    fields: dict[str, StructuredField] = {}
    health_failures: list[str] = []
    receipt_failures: list[str] = []

    for spec in manifest["runs"]:
        if not isinstance(spec, RunSpec):
            raise TypeError("manifest run is not a RunSpec")
        case_dir = work_dir / spec.run_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
        metadata = build_case(case_dir, spec)
        execution = execute_case(case_dir, timeout_per_case)
        field = None
        parse_error = ""
        try:
            field = load_case_field(case_dir)
            fields[spec.run_id] = field
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        failures, receipt = _solver_health(case_dir, metadata, execution, field)
        if "requested div(phi,T) scheme" in " ".join(failures):
            receipt_failures.append(spec.run_id)
        if parse_error:
            failures.append(parse_error)
        health_failures.extend(f"{spec.run_id}: {failure}" for failure in failures)
        observations = field_observations(field, spec.operating.forcing_span) if field is not None else None
        records[spec.run_id] = {
            "metadata": metadata,
            "execution": execution,
            "effective_div_schemes": receipt,
            "observations": observations,
            "health_failures": failures,
        }

    library_rows: list[dict[str, object]] = []
    scheme_deltas = []
    reference_errors = []
    candidate_errors = []
    z_dependence = []
    negative_rows: list[dict[str, object]] = []
    for op in (OPERATING_CASES if mode == "full" else OPERATING_CASES[:1]):
        prefix = op.case_id + "__"
        needed = [prefix + role for role in (
            "upwind_coarse", "linearUpwind_coarse", "upwind_fine", "linearUpwind_fine",
            "reference", "reference_check",
        )]
        if not all(run_id in fields for run_id in needed):
            continue
        reference = fields[prefix + "reference"]
        reference_check = fields[prefix + "reference_check"]
        reference_signature = signature(reference, reference_check, op)
        reference_errors.append(float(reference_signature[1]))
        for mesh_name in ("coarse", "fine"):
            upwind = fields[prefix + "upwind_" + mesh_name]
            linear = fields[prefix + "linearUpwind_" + mesh_name]
            upwind_signature = signature(upwind, reference_check, op)
            linear_signature = signature(linear, reference_check, op)
            candidate_errors.extend((float(upwind_signature[1]), float(linear_signature[1])))
            library_rows.extend((
                {"group": op.case_id, "mesh": mesh_name, "scheme": "upwind", "label": 0, "features": upwind_signature.tolist()},
                {"group": op.case_id, "mesh": mesh_name, "scheme": "linearUpwind", "label": 1, "features": linear_signature.tolist()},
            ))
            negative_rows.extend((
                {"group": op.case_id, "mesh": mesh_name, "label": 0, "features": upwind_signature.tolist()},
                {"group": op.case_id, "mesh": mesh_name, "label": 1, "features": upwind_signature.tolist()},
            ))
            scheme_deltas.append(float(np.sqrt(np.mean((upwind.values - linear.values) ** 2)) / op.forcing_span))
        for run_id in needed:
            observation = records[run_id]["observations"]
            if observation is not None:
                z_dependence.append(float(observation["z_dependence"]))

    ref_median = float(np.median(reference_errors)) if reference_errors else math.nan
    candidate_median = float(np.median(candidate_errors)) if candidate_errors else math.nan
    metrics = {
        "solver_health_failure_count": float(len(health_failures)),
        "scheme_receipt_failure_count": float(len(receipt_failures)),
        "minimum_z_dependence": float(min(z_dependence)) if z_dependence else math.nan,
        "minimum_scheme_field_delta": float(min(scheme_deltas)) if scheme_deltas else math.nan,
        "grouped_accuracy": grouped_accuracy(library_rows),
        "coarse_to_fine_accuracy": grouped_accuracy(library_rows, "coarse", "fine"),
        "fine_to_coarse_accuracy": grouped_accuracy(library_rows, "fine", "coarse"),
        "negative_control_accuracy": grouped_accuracy(negative_rows),
        "reference_median_relative_l2": ref_median,
        "reference_max_relative_l2": float(max(reference_errors)) if reference_errors else math.nan,
        "reference_to_candidate_error_ratio": ref_median / candidate_median if candidate_median > 0 else math.nan,
        "completed_case_count": float(sum(not row["health_failures"] for row in records.values())),
    }
    gates = evaluate_go(metrics)
    if mode == "full":
        verdict = "GO" if all(item["passed"] for item in gates) else "FAIL"
    else:
        smoke_ok = (
            metrics["solver_health_failure_count"] == 0
            and metrics["scheme_receipt_failure_count"] == 0
            and metrics["minimum_z_dependence"] >= 0.02
            and metrics["minimum_scheme_field_delta"] >= 1.0e-6
            and np.isfinite(metrics["reference_max_relative_l2"])
        )
        verdict = "SMOKE_PASS" if smoke_ok else "SMOKE_FAIL"
    serializable_manifest = dict(manifest)
    serializable_manifest["runs"] = [
        {"run_id": spec.run_id, "operating_case": spec.operating.case_id,
         "mesh": spec.mesh.name, "scheme_key": spec.scheme_key, "role": spec.role}
        for spec in manifest["runs"]
    ]
    return {
        "manifest": serializable_manifest,
        "runtime": runtime,
        "metrics": metrics,
        "gates": gates,
        "verdict": verdict,
        "health_failures": health_failures,
        "signature_library": library_rows,
        "case_records": records,
    }


def _json_compatible(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def write_report(report: Mapping[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_compatible(report)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def print_report(report: Mapping[str, object]) -> None:
    manifest = report["manifest"]
    runtime = report["runtime"]
    print(f"protocol: {manifest['protocol_id']} ({manifest['mode']})")
    print(f"image: {runtime['image_reference']} ({runtime['image_id']}, {runtime['architecture']})")
    print(f"OpenFOAM: {runtime['openfoam_version']} build {runtime['openfoam_build']}")
    print(f"solver cases: {manifest['run_count']}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value}")
    if manifest["mode"] == "full":
        for item in report["gates"]:
            print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['gate']['name']}: {item['value']}")
    else:
        print("smoke mode does not issue a scientific GO verdict")
    for failure in report["health_failures"]:
        print(f"[HEALTH FAILURE] {failure}")
    print(f"verdict: {report['verdict']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/openfoam_3d_audit"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/openfoam_3d_audit.json"))
    parser.add_argument("--timeout-per-case", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args.mode, args.work_dir, args.timeout_per_case)
    print_report(report)
    print(f"report: {write_report(report, args.output)}")
    return 0 if report["verdict"] in {"GO", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
