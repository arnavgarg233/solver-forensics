"""Multi-resolution audit of documented py-pde time integrators.

The protocol compares ``EulerSolver`` with ``RungeKuttaSolver`` on one periodic
advection-diffusion equation. Both solvers receive the same central-difference
spatial right-hand side, Fourier initial conditions, grid ladder, and time-step
schedule. The schedule is fixed as ``dt = cfl * dx``. Thus Euler's first-order
time error is O(dx), while the common central spatial error is O(dx**2) and the
higher-order Runge-Kutta time error is smaller. Only per-IC convergence rates,
not single-grid errors or grid labels, enter the detector.

The full configuration uses 60 paired initial conditions. GO_CRITERIA is defined
at import time and is evaluated without adaptation. Failed numerical solves and
failed gates remain in the returned report and in CSV output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from pde import CartesianGrid, PDEBase, ScalarField
from pde.solvers import Controller, EulerSolver, RungeKuttaSolver


PROTOCOL_ID = "library-multires-euler-rk-v1"
REQUIRED_PY_PDE_VERSION = "0.56.0"
SOLVERS = {"euler": EulerSolver, "runge_kutta": RungeKuttaSolver}


@dataclass(frozen=True)
class Gate:
    """One immutable, pre-specified decision criterion."""

    name: str
    metric: str
    comparison: str
    lower: float | None = None
    upper: float | None = None


# Frozen before any numerical result is computed. These values must not be
# selected or relaxed after inspecting an audit outcome.
GO_CRITERIA = (
    Gate("all numerical solves completed", "solver_failure_count", "le", upper=0.0),
    Gate("paired grouped-LOO accuracy", "accuracy", "ge", lower=0.90),
    Gate("Euler median rate is first order", "euler_median_rate", "between", 0.75, 1.25),
    Gate("Runge-Kutta median rate is second order", "rk_median_rate", "between", 1.70, 2.30),
    Gate("median paired rate gap", "median_paired_rate_gap", "ge", lower=0.60),
    Gate("per-IC rate ordering fraction", "rate_ordering_fraction", "ge", lower=0.90),
    Gate("coarsest-pair accuracy", "coarse_pair_accuracy", "ge", lower=0.80),
    Gate("finest-pair accuracy", "fine_pair_accuracy", "ge", lower=0.80),
)


@dataclass(frozen=True)
class AuditConfig:
    """Numerical design for the controlled refinement experiment."""

    n_ics: int = 60
    grid_sizes: tuple[int, ...] = (32, 48, 72, 108)
    length: float = 1.0
    speed: float = 1.0
    diffusivity: float = 0.01
    t_final: float = 0.10
    cfl: float = 0.40
    max_mode: int = 5
    seed: int = 2026

    def validate(self) -> None:
        if self.n_ics < 3:
            raise ValueError("n_ics must be at least 3 for grouped leave-one-IC-out scoring")
        if len(self.grid_sizes) < 3:
            raise ValueError("at least three grids are required for a multi-resolution audit")
        if any(n < 8 for n in self.grid_sizes):
            raise ValueError("all grid sizes must be at least 8")
        if any(b <= a for a, b in zip(self.grid_sizes, self.grid_sizes[1:])):
            raise ValueError("grid_sizes must be strictly increasing")
        if min(self.length, self.diffusivity, self.t_final, self.cfl) <= 0:
            raise ValueError("length, diffusivity, t_final, and cfl must be positive")
        if self.max_mode < 1 or 2 * self.max_mode >= min(self.grid_sizes):
            raise ValueError("max_mode must be resolved by every grid")

        # A fixed advective CFL does not guarantee explicit diffusion stability.
        # Verify the full Euler amplification factor on every proposed grid.
        theta = np.linspace(0.0, np.pi, 8193)
        for n in self.grid_sizes:
            dx = self.length / n
            dt = self.cfl * dx
            lam = (
                -4.0 * self.diffusivity * np.sin(theta / 2.0) ** 2 / dx**2
                - 1j * self.speed * np.sin(theta) / dx
            )
            if float(np.max(np.abs(1.0 + dt * lam))) > 1.0 + 1e-12:
                raise ValueError(
                    f"controlled dt=cfl*dx schedule is unstable for Euler at N={n}; "
                    "the design cannot isolate integrator order"
                )
            steps = self.t_final / dt
            if not np.isclose(steps, round(steps), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"t_final requires a partial final step at N={n}; "
                    "the dt=cfl*dx schedule would not remain exact"
                )


@dataclass(frozen=True)
class FourierIC:
    """A grid-independent periodic initial condition with analytic evolution."""

    modes: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray

    def values(self, x: np.ndarray, t: float, config: AuditConfig) -> np.ndarray:
        k = 2.0 * np.pi * self.modes / config.length
        phase = k[:, None] * (x[None, :] - config.speed * t) + self.phases[:, None]
        decay = np.exp(-config.diffusivity * k**2 * t)
        waves = (self.amplitudes * decay)[:, None] * np.sin(phase)
        return 1.0 + 0.25 * np.sum(waves, axis=0)


class CentralAdvectionDiffusion(PDEBase):
    """Periodic central spatial RHS shared by both documented integrators."""

    def __init__(self, speed: float, diffusivity: float):
        super().__init__()
        self.speed = speed
        self.diffusivity = diffusivity

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        u = state.data
        dx = float(state.grid.discretization[0])
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
        return ScalarField(state.grid, -self.speed * ux + self.diffusivity * uxx)


def installed_py_pde_version() -> str:
    return importlib.metadata.version("py-pde")


def require_pinned_version() -> str:
    version = installed_py_pde_version()
    if version != REQUIRED_PY_PDE_VERSION:
        raise RuntimeError(
            f"this audit requires py-pde=={REQUIRED_PY_PDE_VERSION}, found {version}"
        )
    return version


def criteria_sha256() -> str:
    payload = [gate.__dict__ for gate in GO_CRITERIA]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cell_centers(n: int, length: float) -> np.ndarray:
    return (np.arange(n, dtype=float) + 0.5) * length / n


def generate_initial_conditions(config: AuditConfig) -> list[FourierIC]:
    rng = np.random.default_rng(config.seed)
    modes = np.arange(1, config.max_mode + 1, dtype=float)
    out = []
    for _ in range(config.n_ics):
        amplitudes = rng.normal(size=config.max_mode)
        amplitudes /= np.linalg.norm(amplitudes) + 1e-15
        phases = rng.uniform(0.0, 2.0 * np.pi, size=config.max_mode)
        out.append(FourierIC(modes.copy(), amplitudes, phases))
    return out


def solve_one(solver_key: str, n: int, ic: FourierIC, config: AuditConfig) -> np.ndarray:
    """Advance one IC with a documented py-pde solver and the shared RHS."""

    if solver_key not in SOLVERS:
        raise ValueError(f"unknown solver {solver_key!r}")
    grid = CartesianGrid([[0.0, config.length]], n, periodic=True)
    x = np.asarray(grid.axes_coords[0])
    state = ScalarField(grid, ic.values(x, 0.0, config))
    equation = CentralAdvectionDiffusion(config.speed, config.diffusivity)
    solver = SOLVERS[solver_key](equation, backend="numpy", adaptive=False)
    dt = config.cfl * config.length / n
    result = Controller(solver, t_range=config.t_final, tracker=None).run(state, dt=dt)
    return np.asarray(result.data, dtype=float).copy()


def convergence_rate_features(errors: np.ndarray, grid_sizes: Sequence[int]) -> np.ndarray:
    """Return adjacent-grid convergence rates for each IC.

    Multiplying all errors of one IC by any positive constant leaves these
    features unchanged. This removes absolute single-grid error magnitude as a
    possible mesh or IC confound.
    """

    errors = np.asarray(errors, dtype=float)
    grids = np.asarray(grid_sizes, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != grids.size:
        raise ValueError("errors must have shape (n_ics, len(grid_sizes))")
    rates = np.full((errors.shape[0], grids.size - 1), np.nan)
    valid = (
        np.isfinite(errors[:, :-1])
        & np.isfinite(errors[:, 1:])
        & (errors[:, :-1] > 0.0)
        & (errors[:, 1:] > 0.0)
    )
    denominator = np.log(grids[1:] / grids[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.log(errors[:, :-1] / errors[:, 1:]) / denominator[None, :]
    rates[valid] = raw[valid]
    return rates


def grouped_loo_accuracy(euler: np.ndarray, rk: np.ndarray) -> float:
    """Nearest-centroid accuracy with both rows of each held-out IC excluded."""

    euler = np.asarray(euler, dtype=float)
    rk = np.asarray(rk, dtype=float)
    if euler.shape != rk.shape or euler.ndim != 2 or euler.shape[0] < 3:
        raise ValueError("paired feature arrays must have equal (n_ics, n_features) shapes")
    if not np.all(np.isfinite(euler)) or not np.all(np.isfinite(rk)):
        return 0.0

    correct = 0
    for held_out in range(euler.shape[0]):
        keep = np.arange(euler.shape[0]) != held_out
        train = np.vstack([euler[keep], rk[keep]])
        scale = np.std(train, axis=0)
        scale[scale < 1e-12] = 1.0
        center = np.mean(train, axis=0)
        centroid_e = np.mean((euler[keep] - center) / scale, axis=0)
        centroid_r = np.mean((rk[keep] - center) / scale, axis=0)
        for expected, row in enumerate((euler[held_out], rk[held_out])):
            z = (row - center) / scale
            distances = (np.linalg.norm(z - centroid_e), np.linalg.norm(z - centroid_r))
            correct += int(int(np.argmin(distances)) == expected)
    return correct / (2.0 * euler.shape[0])


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else math.nan


def evaluate_go(metrics: Mapping[str, float]) -> list[dict[str, object]]:
    """Apply the immutable GO criteria without changing or dropping failures."""

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
        evaluations.append({"gate": gate, "value": value, "passed": bool(passed)})
    return evaluations


SolveFunction = Callable[[str, int, FourierIC, AuditConfig], np.ndarray]


def run_audit(
    config: AuditConfig = AuditConfig(), solve_fn: SolveFunction = solve_one
) -> dict[str, object]:
    """Run the paired ladder and return all metrics, gates, and failures."""

    config.validate()
    version = require_pinned_version()
    initial_conditions = generate_initial_conditions(config)
    errors = {
        key: np.full((config.n_ics, len(config.grid_sizes)), np.inf)
        for key in SOLVERS
    }
    failures: list[str] = []

    for solver_key in SOLVERS:
        for ic_index, ic in enumerate(initial_conditions):
            for grid_index, n in enumerate(config.grid_sizes):
                x = cell_centers(n, config.length)
                truth = ic.values(x, config.t_final, config)
                try:
                    numerical = np.asarray(solve_fn(solver_key, n, ic, config), dtype=float)
                    if numerical.shape != truth.shape or not np.all(np.isfinite(numerical)):
                        raise ValueError("solver returned a non-finite field or wrong shape")
                    errors[solver_key][ic_index, grid_index] = float(
                        np.linalg.norm(numerical - truth) / np.linalg.norm(truth)
                    )
                except Exception as exc:  # preserved as a failed gate and CSV metadata
                    failures.append(
                        f"{solver_key},IC={ic_index},N={n}: {type(exc).__name__}: {exc}"
                    )

    features = {
        key: convergence_rate_features(value, config.grid_sizes)
        for key, value in errors.items()
    }
    per_ic_euler = np.array([_finite_median(row) for row in features["euler"]])
    per_ic_rk = np.array([_finite_median(row) for row in features["runge_kutta"]])
    metrics = {
        "solver_failure_count": float(len(failures)),
        "accuracy": grouped_loo_accuracy(features["euler"], features["runge_kutta"]),
        "euler_median_rate": _finite_median(per_ic_euler),
        "rk_median_rate": _finite_median(per_ic_rk),
        "median_paired_rate_gap": _finite_median(per_ic_rk - per_ic_euler),
        "rate_ordering_fraction": float(np.mean(per_ic_rk > per_ic_euler)),
        "coarse_pair_accuracy": grouped_loo_accuracy(
            features["euler"][:, :1], features["runge_kutta"][:, :1]
        ),
        "fine_pair_accuracy": grouped_loo_accuracy(
            features["euler"][:, -1:], features["runge_kutta"][:, -1:]
        ),
    }
    gates = evaluate_go(metrics)
    return {
        "protocol_id": PROTOCOL_ID,
        "py_pde_version": version,
        "required_py_pde_version": REQUIRED_PY_PDE_VERSION,
        "criteria_sha256": criteria_sha256(),
        "config": config,
        "dt_policy": "dt=cfl*dx",
        "metrics": metrics,
        "gates": gates,
        "verdict": "GO" if all(item["passed"] for item in gates) else "FAIL",
        "failures": failures,
        "errors": errors,
        "features": features,
    }


def write_csv(report: Mapping[str, object], path: str | Path) -> Path:
    """Write gate outcomes with complete protocol and version metadata."""

    config = report["config"]
    if not isinstance(config, AuditConfig):
        raise TypeError("report config must be an AuditConfig")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "protocol_id", "py_pde_version", "required_py_pde_version", "criteria_sha256",
        "dt_policy", "n_ics", "grid_sizes", "dt_values", "step_counts", "cfl", "t_final",
        "length", "speed", "diffusivity", "max_mode", "seed", "solver_a", "solver_b",
        "gate", "metric", "comparison", "lower", "upper", "value", "passed", "verdict",
        "numerical_failures",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in report["gates"]:
            gate = item["gate"]
            writer.writerow({
                "protocol_id": report["protocol_id"],
                "py_pde_version": report["py_pde_version"],
                "required_py_pde_version": report["required_py_pde_version"],
                "criteria_sha256": report["criteria_sha256"],
                "dt_policy": report["dt_policy"],
                "n_ics": config.n_ics,
                "grid_sizes": ";".join(map(str, config.grid_sizes)),
                "dt_values": ";".join(
                    f"{config.cfl * config.length / n:.17g}" for n in config.grid_sizes
                ),
                "step_counts": ";".join(
                    str(round(config.t_final / (config.cfl * config.length / n)))
                    for n in config.grid_sizes
                ),
                "cfl": config.cfl,
                "t_final": config.t_final,
                "length": config.length,
                "speed": config.speed,
                "diffusivity": config.diffusivity,
                "max_mode": config.max_mode,
                "seed": config.seed,
                "solver_a": "EulerSolver",
                "solver_b": "RungeKuttaSolver",
                "gate": gate.name,
                "metric": gate.metric,
                "comparison": gate.comparison,
                "lower": "" if gate.lower is None else gate.lower,
                "upper": "" if gate.upper is None else gate.upper,
                "value": item["value"],
                "passed": item["passed"],
                "verdict": report["verdict"],
                "numerical_failures": " | ".join(report["failures"]),
            })
    return output


def print_report(report: Mapping[str, object]) -> None:
    config = report["config"]
    print(f"protocol: {report['protocol_id']}")
    print(f"py-pde version: {report['py_pde_version']} (required {report['required_py_pde_version']})")
    print(f"paired ICs: {config.n_ics}; grids: {config.grid_sizes}; dt policy: {report['dt_policy']}")
    print(f"frozen GO criteria SHA256: {report['criteria_sha256']}")
    for item in report["gates"]:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"[{mark}] {item['gate'].name}: {item['value']:.6g}")
    for failure in report["failures"]:
        print(f"[NUMERICAL FAILURE] {failure}")
    print(f"verdict: {report['verdict']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("library_multires_audit.csv"))
    parser.add_argument("--n-ics", type=int, default=AuditConfig.n_ics)
    args = parser.parse_args(argv)
    report = run_audit(AuditConfig(n_ics=args.n_ics))
    print_report(report)
    print(f"CSV: {write_csv(report, args.output)}")
    return 0 if report["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
