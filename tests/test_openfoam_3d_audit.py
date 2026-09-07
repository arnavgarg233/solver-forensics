import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "external"))

import openfoam_3d_audit as audit


class FrozenProtocolTests(unittest.TestCase):
    def test_full_protocol_is_exactly_sixty_digest_pinned_solver_cases(self):
        manifest = audit.protocol_manifest("full")
        self.assertEqual(manifest["run_count"], 60)
        self.assertEqual(len(manifest["operating_cases"]), 10)
        self.assertEqual(manifest["image_reference"],
                         "openeuler/openfoam@sha256:a9e2ee499bca06b43bfe330de779e675c496687420a4ab3bec2316f20a693d4a")
        self.assertEqual(manifest["expected_image_id"],
                         "sha256:a7e4fa3dc52cfb323a4079beab1969dff013c8e195d76e0d21bc778a181147d2")
        self.assertEqual(manifest["expected_openfoam_version"], "v2506")
        self.assertEqual(manifest["expected_openfoam_build"], "_615aae61d7-20250627")
        self.assertEqual(manifest["solver"], "scalarTransportFoam")
        self.assertIn("fvm::div(phi,T)", manifest["solver_equation"])
        self.assertEqual(len(audit.criteria_sha256()), 64)
        self.assertIsInstance(audit.GO_CRITERIA, tuple)

        roles_by_group = {}
        for spec in manifest["runs"]:
            roles_by_group.setdefault(spec.operating.case_id, set()).add(spec.role)
        expected = {
            "upwind_coarse", "linearUpwind_coarse", "upwind_fine",
            "linearUpwind_fine", "reference", "reference_check",
        }
        self.assertTrue(all(roles == expected for roles in roles_by_group.values()))

    def test_smoke_protocol_uses_same_six_roles_on_small_three_dimensional_meshes(self):
        manifest = audit.protocol_manifest("smoke")
        self.assertEqual(manifest["run_count"], 6)
        for mesh in manifest["meshes"].values():
            self.assertGreaterEqual(mesh["ny"], 2)
            self.assertGreaterEqual(mesh["nz"], 2)
            self.assertEqual(mesh["nx"], 4 * mesh["ny"])
            self.assertEqual(mesh["ny"], mesh["nz"])
        self.assertEqual(manifest["grouping_rule"],
                         "hold out every mesh and scheme row from one operating case together")
        self.assertIn("duplicate upwind", manifest["negative_control"])

    def test_every_operating_case_has_real_z_dependent_forcing(self):
        for operating in audit.OPERATING_CASES:
            operating.validate()
            self.assertNotEqual(operating.lower_inlet, operating.upper_inlet)
            self.assertGreater(operating.forcing_span, 0.0)

    def test_smoke_report_serializes_unavailable_group_metrics_as_json_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = audit.write_report({"grouped_accuracy": float("nan")},
                                        Path(temporary) / "report.json")
            loaded = json.loads(output.read_text())
        self.assertIsNone(loaded["grouped_accuracy"])


class CaseGenerationTests(unittest.TestCase):
    def setUp(self):
        self.op = audit.OPERATING_CASES[0]
        self.mesh = audit.SMOKE_MESHES["fine"]

    def _spec(self, scheme):
        return audit.RunSpec(self.op, self.mesh, scheme, f"{scheme}_fine")

    def test_generated_case_uses_split_z_inlets_and_requested_fv_scheme(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upwind = root / "upwind"
            linear = root / "linear"
            up_meta = audit.build_case(upwind, self._spec("upwind"))
            linear_meta = audit.build_case(linear, self._spec("linearUpwind"))

            blocks = (upwind / "system" / "blockMeshDict").read_text()
            scalar = (upwind / "0" / "T").read_text()
            up_schemes = (upwind / "system" / "fvSchemes").read_text()
            linear_schemes = (linear / "system" / "fvSchemes").read_text()
            self.assertEqual(blocks.count("hex ("), 2)
            self.assertIn("inletLower", blocks)
            self.assertIn("inletUpper", blocks)
            self.assertIn("value uniform 0;", scalar)
            self.assertIn("value uniform 1;", scalar)
            self.assertIn("div(phi,T) Gauss upwind;", up_schemes)
            self.assertIn("div(phi,T) Gauss linearUpwind grad(T);", linear_schemes)
            self.assertNotEqual(up_schemes, linear_schemes)
            self.assertLessEqual(up_meta["max_nominal_courant"], audit.CFL + 1e-12)
            self.assertEqual(linear_meta["fv_scheme"], "Gauss linearUpwind grad(T)")

            metadata = json.loads((linear / "audit_case.json").read_text())
            self.assertEqual(metadata["scheme_key"], "linearUpwind")
            self.assertEqual(metadata["mesh"]["nz"], 4)

    def test_execute_command_uses_digest_and_clean_bash_environment(self):
        completed = mock.Mock(returncode=0, stdout="")
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(audit, "_run_command", return_value=completed) as runner:
                result = audit.execute_case(temporary, timeout=12.0)
        command = runner.call_args.args[0]
        self.assertIn(audit.IMAGE_REFERENCE, command)
        self.assertIn("--noprofile", command)
        self.assertIn("--norc", command)
        self.assertIn("scalarTransportFoam", command[-1])
        self.assertIn("foamDictionary", command[-1])
        self.assertEqual(result["returncode"], 0)


class FieldAndSignatureTests(unittest.TestCase):
    @staticmethod
    def _field(nx=4, ny=3, nz=4, offset=0.0):
        x = np.linspace(0.5, 3.5, nx)
        y = np.linspace(0.1, 0.9, ny)
        z = np.linspace(0.1, 0.9, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        values = 0.05 * xx + 0.02 * yy + 0.5 * zz + offset
        return audit.StructuredField(x, y, z, values)

    def test_openfoam_ascii_parsers_and_structuring_preserve_3d_values(self):
        centres = np.array([
            [0.25, 0.25, 0.25], [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25], [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25], [0.75, 0.75, 0.75],
        ])
        values = np.arange(8, dtype=float)
        vector_rows = "\n".join(f"({x} {y} {z})" for x, y, z in centres)
        scalar_rows = "\n".join(str(value) for value in values)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scalar_path = root / "T"
            vector_path = root / "C"
            scalar_path.write_text(
                f"internalField nonuniform List<scalar>\n8\n(\n{scalar_rows}\n);\n")
            vector_path.write_text(
                f"internalField nonuniform List<vector>\n8\n(\n{vector_rows}\n);\n")
            parsed_values = audit.parse_internal_scalar(scalar_path)
            parsed_centres = audit.parse_internal_vector(vector_path)
        field = audit.structured_field(parsed_centres, parsed_values)
        self.assertEqual(field.values.shape, (2, 2, 2))
        np.testing.assert_array_equal(parsed_values, values)
        np.testing.assert_array_equal(parsed_centres, centres)

    def test_two_dimensional_or_z_invariant_outputs_cannot_pass_3d_gate(self):
        centres_2d = np.array([[x, y, 0.5] for x in (0.25, 0.75) for y in (0.25, 0.75)])
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            audit.structured_field(centres_2d, np.arange(4.0))

        invariant = self._field()
        invariant = audit.StructuredField(
            invariant.x, invariant.y, invariant.z,
            np.broadcast_to(invariant.values.mean(axis=2, keepdims=True), invariant.values.shape).copy(),
        )
        observations = audit.field_observations(invariant, forcing_span=1.0)
        self.assertAlmostEqual(observations["z_dependence"], 0.0)
        metrics = {gate.metric: 1.0 for gate in audit.GO_CRITERIA}
        metrics["minimum_z_dependence"] = observations["z_dependence"]
        failed = {item["gate"]["metric"] for item in audit.evaluate_go(metrics) if not item["passed"]}
        self.assertIn("minimum_z_dependence", failed)

    def test_signature_has_fixed_observation_library_and_detects_error(self):
        reference = self._field()
        exact = self._field()
        shifted = self._field(offset=0.05)
        op = audit.OPERATING_CASES[0]
        np.testing.assert_allclose(audit.signature(exact, reference, op), 0.0, atol=1e-14)
        shifted_signature = audit.signature(shifted, reference, op)
        self.assertEqual(shifted_signature.shape, (len(audit.FEATURE_NAMES),))
        self.assertGreater(shifted_signature[0], 0.0)
        self.assertGreater(shifted_signature[1], 0.0)


class GroupedEvaluationTests(unittest.TestCase):
    @staticmethod
    def _rows(duplicate=False):
        rows = []
        for group_index in range(6):
            for mesh, mesh_offset in (("coarse", 0.0), ("fine", 0.1)):
                low = np.array([0.0 + mesh_offset, group_index * 0.001])
                high = low if duplicate else np.array([5.0 + mesh_offset, group_index * 0.001])
                rows.extend((
                    {"group": f"g{group_index}", "mesh": mesh, "label": 0, "features": low},
                    {"group": f"g{group_index}", "mesh": mesh, "label": 1, "features": high},
                ))
        return rows

    def test_grouped_evaluation_holds_out_operating_groups_and_transfers_meshes(self):
        rows = self._rows()
        self.assertEqual(audit.grouped_accuracy(rows), 1.0)
        self.assertEqual(audit.grouped_accuracy(rows, "coarse", "fine"), 1.0)
        self.assertEqual(audit.grouped_accuracy(rows, "fine", "coarse"), 1.0)

    def test_duplicate_field_negative_control_is_exactly_chance(self):
        self.assertEqual(audit.grouped_accuracy(self._rows(duplicate=True)), 0.5)


@unittest.skipUnless(os.environ.get("RUN_OPENFOAM_INTEGRATION") == "1",
                     "set RUN_OPENFOAM_INTEGRATION=1 to run Docker smoke integration")
class DockerIntegrationTests(unittest.TestCase):
    def test_digest_pinned_openfoam_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = audit.run_audit("smoke", Path(temporary) / "cases", timeout_per_case=300.0)
        self.assertEqual(report["verdict"], "SMOKE_PASS")
        self.assertEqual(report["metrics"]["completed_case_count"], 6.0)
        self.assertGreater(report["metrics"]["minimum_z_dependence"], 0.02)
        self.assertGreater(report["metrics"]["minimum_scheme_field_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
