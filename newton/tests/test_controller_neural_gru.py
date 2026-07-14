# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Anchor schema-v3 GRU actuator contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import types
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, ClampingMaxEffort, ControllerNeuralGRU, Delay

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch

    class _ProbeGRU(torch.nn.Module):
        """Expose a real GRU while projecting assembled inputs deterministically."""

        def __init__(self, weights: torch.Tensor, bias: torch.Tensor):
            super().__init__()
            input_size = int(weights.shape[1])
            output_size = int(weights.shape[0])
            self.gru = torch.nn.GRU(input_size, 2, batch_first=True)
            self.head = torch.nn.Linear(input_size, output_size)
            with torch.no_grad():
                self.head.weight.copy_(weights)
                self.head.bias.copy_(bias)

        def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Project each feature row and preserve hidden state."""
            return self.head(x), hidden


def _axis_values(width: int, value: float) -> float | list[float]:
    """Return the scalar-or-vector representation used by Anchor metadata."""
    return value if width == 1 else [value] * width


def _metadata(
    input_joints: tuple[str, ...] = ("joint_a",),
    output_joints: tuple[str, ...] = ("joint_a",),
    *,
    mappings: list[dict[str, list[str]]] | None = None,
    features: tuple[str, ...] = ("position",),
    target: str = "torque",
    kp: list[float] | None = None,
    kd: list[float] | None = None,
    dry: list[float] | None = None,
    viscous: list[float] | None = None,
) -> dict:
    """Build realistic Anchor schema-v3 metadata for one model artifact."""
    if mappings is None:
        mappings = [
            {
                "input_joints": list(input_joints),
                "output_joints": list(output_joints),
            }
        ]
    input_width = len(mappings[0]["input_joints"])
    output_width = len(mappings[0]["output_joints"])
    topology = "siso" if input_width == output_width == 1 else ("miso" if output_width == 1 else "mimo")
    specs = [
        {
            "name": feature,
            "domain": "output_joints" if feature == "previous_torque" else "input_joints",
            "channels": ["value"],
        }
        for feature in features
    ]
    feature_width = {feature: output_width if feature == "previous_torque" else input_width for feature in features}
    metadata = {
        "schema_version": 3,
        "model_type": "gru",
        "joint_mappings": mappings,
        "input_columns": list(features),
        "input_feature_specs": specs,
        "input_size": sum(feature_width.values()),
        "output_size": output_width,
        "target_columns": [target],
        "sample_dt_s": 0.002,
        "normalization": {
            "inputs": {
                "mean": {name: _axis_values(width, 0.0) for name, width in feature_width.items()},
                "std": {name: _axis_values(width, 1.0) for name, width in feature_width.items()},
            },
            "targets": {
                "mean": {target: _axis_values(output_width, 0.0)},
                "std": {target: _axis_values(output_width, 1.0)},
            },
        },
        "target_normalization": "identity",
        "output_head": {"activation": "tanh", "scale": 15.0},
        "delay": {"handling": "learned", "external_delay_s": 0.0},
        "training_topology": topology,
    }
    if "previous_torque" in features:
        metadata["previous_torque_derivation"] = {
            "source": "previous_raw_network_output_pre_clamp",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset_boundaries": ["recorded_episode", "joint_mapping"],
            "value_space": "residual_torque" if target == "torque_residual" else "physical_torque",
        }
    if "solver_pd" in features or target == "torque_residual":
        pd_width = input_width if "solver_pd" in features else output_width
        metadata["pd_baseline"] = {
            "domain": "input_joints" if "solver_pd" in features else "output_joints",
            "kp": kp if kp is not None else [40.0] * pd_width,
            "kd": kd if kd is not None else [1.0] * pd_width,
            "velocity_target": "zero",
        }
    if target == "torque_residual":
        metadata["friction_baseline"] = {
            "domain": "output_joints",
            "dry": dry if dry is not None else [0.0] * output_width,
            "viscous": viscous if viscous is not None else [0.0] * output_width,
        }
    return metadata


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNeuralGRU(unittest.TestCase):
    """Exercise schema parsing, tensor assembly, state, and builder hooks."""

    def setUp(self):
        """Create a CPU-only temporary checkpoint workspace."""
        self.device = wp.get_device("cpu")
        self.tmp = tempfile.TemporaryDirectory()
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("ignore", DeprecationWarning)
        self.checkpoint_index = 0

    def tearDown(self):
        """Remove temporary checkpoints and restore warning filters."""
        self.warning_context.__exit__(None, None, None)
        self.tmp.cleanup()

    def _save(
        self,
        metadata: dict,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> str:
        """Save a deterministic TorchScript GRU with embedded metadata."""
        input_size = int(metadata.get("input_size", 1))
        output_size = int(metadata.get("output_size", 1))
        if weights is None:
            weights = np.zeros((output_size, input_size), dtype=np.float32)
        if bias is None:
            bias = np.zeros(output_size, dtype=np.float32)
        network = _ProbeGRU(torch.tensor(weights), torch.tensor(bias))
        path = os.path.join(self.tmp.name, f"gru_{self.checkpoint_index}.pt")
        self.checkpoint_index += 1
        torch.jit.save(
            torch.jit.script(network),
            path,
            _extra_files={"metadata.json": json.dumps(metadata)},
        )
        return path

    def _arrays(self, values, *, dtype=wp.float32):
        """Create a one-dimensional Warp array on the test CPU device."""
        return wp.array(values, dtype=dtype, device=self.device)

    def test_shared_siso_mapping_selection(self):
        """Select one physical mapping while retaining one scalar model contract."""
        mappings = [{"input_joints": [name], "output_joints": [name]} for name in ("Rotation", "Shoulder", "Elbow")]
        path = self._save(_metadata(mappings=mappings))

        with self.assertRaisesRegex(ValueError, "requires 'mapping_index'"):
            ControllerNeuralGRU(path)
        controller = ControllerNeuralGRU(path, mapping_index=1)

        self.assertEqual(controller.input_joints, ("Shoulder",))
        self.assertEqual(controller.output_joints, ("Shoulder",))
        controller.validate_io(input_count=3, output_count=3)
        with self.assertRaisesRegex(ValueError, "out of range"):
            ControllerNeuralGRU(path, mapping_index=3)

    def test_shared_siso_mappings_batch_in_one_controller(self):
        """Batch distinct physical SISO mappings while preserving recurrent rows."""
        mappings = [{"input_joints": [name], "output_joints": [name]} for name in ("Rotation", "Shoulder")]
        path = self._save(
            _metadata(mappings=mappings),
            weights=np.array([[1.0]], dtype=np.float32),
        )
        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in mappings]
        joints = [
            builder.add_joint_revolute(
                parent=-1 if index == 0 else links[index - 1],
                child=link,
                axis=newton.Axis.Z,
            )
            for index, link in enumerate(links)
        ]
        builder.add_articulation(joints)
        for mapping_index, joint in enumerate(joints):
            dof = builder.joint_qd_start[joint]
            coord = builder.joint_q_start[joint]
            builder.add_actuator_group(
                ControllerNeuralGRU,
                input_indices=[dof],
                output_indices=[dof],
                input_pos_indices=[coord],
                output_pos_indices=[coord],
                model_path=path,
                mapping_index=mapping_index,
            )

        model = builder.finalize(device=self.device)
        self.assertEqual(len(model.actuators), 1)
        controller = model.actuators[0].controller
        self.assertEqual(controller.mapping_indices, (0, 1))
        state = controller.state(output_count=2, device=self.device)
        self.assertEqual(tuple(state.hidden.shape), (1, 2, 2))
        self.assertEqual(tuple(state.previous_effort.shape), (2, 1))

        indices = self._arrays([0, 1], dtype=wp.uint32)
        forces = self._arrays([0.0, 0.0])
        controller.compute(
            self._arrays([2.0, 3.0]),
            self._arrays([0.0, 0.0]),
            self._arrays([0.0, 0.0]),
            self._arrays([0.0, 0.0]),
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state,
            0.002,
            self.device,
        )
        np.testing.assert_allclose(forces.numpy(), [2.0, 3.0])

    def test_mapping_index_arrays_are_validated(self):
        """Reject malformed per-output mapping arrays during finalization."""
        shared_path = self._save(
            _metadata(
                mappings=[
                    {"input_joints": ["Rotation"], "output_joints": ["Rotation"]},
                    {"input_joints": ["Shoulder"], "output_joints": ["Shoulder"]},
                ]
            )
        )
        shared_cases = (
            ([0.0], "must contain 2 values"),
            ([0.0, 2.0], "out of range"),
            ([0.0, 0.5], "finite integers"),
        )
        for values, message in shared_cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                ControllerNeuralGRU(
                    shared_path,
                    mapping_index=self._arrays(values),
                ).finalize(self.device, output_count=2)

        coupled_path = self._save(
            _metadata(
                mappings=[
                    {"input_joints": ["a", "b"], "output_joints": ["a", "b"]},
                    {"input_joints": ["c", "d"], "output_joints": ["c", "d"]},
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "repeated across every output"):
            ControllerNeuralGRU(
                coupled_path,
                mapping_index=self._arrays([0.0, 1.0]),
            ).finalize(self.device, output_count=2)

    def test_group_indices_follow_selected_mapping_slots(self):
        """Reject outputs that do not identify their selected physical inputs."""
        path = self._save(_metadata(("a", "b", "c"), ("c",)))
        args = ControllerNeuralGRU.resolve_arguments({"model_path": path, "mapping_index": 0})
        ControllerNeuralGRU.validate_resolved_group(args, [4, 5, 6], [6])
        with self.assertRaisesRegex(ValueError, "must reference input slot 2"):
            ControllerNeuralGRU.validate_resolved_group(args, [4, 5, 6], [5])
        with self.assertRaisesRegex(ValueError, "requires 3 input and 1 output"):
            ControllerNeuralGRU.validate_resolved_group(args, [4, 5], [5])

    def test_actuator_input_axis_is_all_or_none(self):
        """Reject partial low-level input-axis wiring."""
        path = self._save(_metadata())
        indices = self._arrays([0], dtype=wp.uint32)
        names = ("input_indices", "input_pos_indices", "input_target_pos_indices")

        for mask in range(1, 7):
            kwargs = {name: indices for bit, name in enumerate(names) if mask & (1 << bit)}
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "must be provided together"):
                Actuator(indices, ControllerNeuralGRU(path), **kwargs)

    def test_actuator_routes_explicit_miso_input_axis(self):
        """Read an explicit two-joint input axis and write one output effort."""
        metadata = _metadata(("shoulder", "elbow"), ("elbow",), features=("position",))
        controller = ControllerNeuralGRU(self._save(metadata, weights=np.array([[1.0, 10.0]], dtype=np.float32)))
        actuator = Actuator(
            indices=self._arrays([1], dtype=wp.uint32),
            controller=controller,
            input_indices=self._arrays([0, 1], dtype=wp.uint32),
            input_pos_indices=self._arrays([2, 0], dtype=wp.uint32),
            input_target_pos_indices=self._arrays([0, 1], dtype=wp.uint32),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state = types.SimpleNamespace(
            joint_q=self._arrays([3.0, 99.0, 2.0]),
            joint_qd=self._arrays([0.0, 0.0]),
        )
        sim_control = types.SimpleNamespace(
            joint_target_q=self._arrays([0.0, 0.0]),
            joint_target_qd=self._arrays([0.0, 0.0]),
            joint_act=self._arrays([0.0, 0.0]),
            joint_f=self._arrays([0.0, 0.0]),
        )

        actuator.step(sim_state, sim_control, actuator.state(), actuator.state(), 0.002)

        np.testing.assert_allclose(sim_control.joint_f.numpy(), [0.0, 32.0])

    def test_gru_rejects_external_delay(self):
        """Reject external Delay through both builder and direct construction."""
        path = self._save(_metadata())
        builder = newton.ModelBuilder()
        link = builder.add_link()
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        dof = builder.joint_qd_start[joint]
        with self.assertRaisesRegex(ValueError, "cannot be combined with a Newton Delay"):
            builder.add_actuator(
                ControllerNeuralGRU,
                index=dof,
                model_path=path,
                delay_steps=1,
            )

        indices = self._arrays([0], dtype=wp.uint32)
        delay = Delay(self._arrays([1], dtype=wp.int32), max_delay=1)
        with self.assertRaisesRegex(ValueError, "cannot be combined with a Newton Delay"):
            Actuator(indices, ControllerNeuralGRU(path), delay=delay)

    def test_miso_feature_major_assembly_includes_dynamic_bias(self):
        """Assemble all feature blocks in Anchor order for a shuffled 3-to-1 mapping."""
        features = (
            "position",
            "position_error",
            "velocity",
            "solver_pd",
            "dynamic_bias",
            "previous_torque",
        )
        metadata = _metadata(
            ("a", "b", "c"),
            ("c",),
            features=features,
            kp=[2.0, 3.0, 4.0],
            kd=[0.5, 1.0, 1.5],
        )
        weights = np.arange(1, 17, dtype=np.float32).reshape(1, 16)
        controller = ControllerNeuralGRU(self._save(metadata, weights=weights))
        controller.finalize(self.device, output_count=1)
        state = controller.state(output_count=1, device=self.device)
        state.previous_effort.fill_(7.0)
        indices = self._arrays([2, 0, 3], dtype=wp.uint32)
        forces = self._arrays([0.0])
        compute_args = (
            self._arrays([1.0, 99.0, 2.0, 3.0]),
            self._arrays([0.2, 99.0, 0.4, 0.6]),
            self._arrays([1.5, 99.0, 2.5, 4.0]),
            self._arrays([9.0, 9.0, 9.0, 9.0]),
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state,
            0.002,
            self.device,
        )
        with self.assertRaisesRegex(ValueError, "requires dynamic_bias"):
            controller.compute(*compute_args)
        controller.compute(*compute_args, dynamic_bias=self._arrays([10.0, 99.0, 20.0, 30.0]))
        raw_features = np.array([2.0, 1.0, 3.0, 0.5, 0.5, 1.0, 0.4, 0.2, 0.6, 0.8, 1.3, 3.1, 20.0, 10.0, 30.0, 7.0])
        self.assertAlmostEqual(float(forces.numpy()[0]), float(raw_features @ weights[0]), places=4)

    def test_previous_torque_uses_raw_pre_clamp_effort_and_resets(self):
        """Feed back raw effort across steps even when applied effort is clamped."""
        metadata = _metadata(features=("previous_torque",))
        controller = ControllerNeuralGRU(
            self._save(metadata, weights=np.array([[1.0]], np.float32), bias=np.array([2.0], np.float32))
        )
        indices = self._arrays([0], dtype=wp.uint32)
        actuator = Actuator(
            indices,
            controller,
            clamping=[ClampingMaxEffort(self._arrays([1.0]))],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state = types.SimpleNamespace(joint_q=self._arrays([0.0]), joint_qd=self._arrays([0.0]))
        sim_control = types.SimpleNamespace(
            joint_target_q=self._arrays([0.0]),
            joint_target_qd=self._arrays([0.0]),
            joint_act=self._arrays([0.0]),
            joint_f=self._arrays([0.0]),
        )
        current, next_state = actuator.state(), actuator.state()

        actuator.step(sim_state, sim_control, current, next_state, 0.002)
        self.assertAlmostEqual(float(sim_control.joint_f.numpy()[0]), 1.0)
        self.assertAlmostEqual(float(next_state.controller_state.previous_effort[0, 0]), 2.0)

        sim_control.joint_f.zero_()
        final_state = actuator.state()
        actuator.step(sim_state, sim_control, next_state, final_state, 0.002)
        self.assertAlmostEqual(float(sim_control.joint_f.numpy()[0]), 1.0)
        self.assertAlmostEqual(float(final_state.controller_state.previous_effort[0, 0]), 4.0)
        final_state.controller_state.reset()
        self.assertEqual(float(final_state.controller_state.previous_effort[0, 0]), 0.0)

    def test_full_and_residual_joint_configurations(self):
        """Configure full-torque outputs as explicit and residual outputs as implicit PD."""
        full_path = self._save(_metadata())
        full = ControllerNeuralGRU.resolve_joint_configurations(
            {"model_path": full_path, "mapping_index": 0}, output_count=1
        )
        self.assertEqual([(item.target_ke, item.target_kd) for item in full], [(0.0, 0.0)])
        full_builder = newton.ModelBuilder()
        full_link = full_builder.add_link()
        full_joint = full_builder.add_joint_revolute(parent=-1, child=full_link, axis=newton.Axis.Z)
        full_dof = full_builder.joint_qd_start[full_joint]
        full_coord = full_builder.joint_q_start[full_joint]
        full_builder.add_actuator(
            ControllerNeuralGRU,
            index=full_dof,
            pos_index=full_coord,
            model_path=full_path,
            mapping_index=0,
        )
        self.assertEqual(full_builder.joint_target_ke[full_dof], 0.0)
        self.assertEqual(full_builder.joint_target_kd[full_dof], 0.0)
        self.assertEqual(full_builder.joint_target_mode[full_dof], int(newton.JointTargetMode.EFFORT))

        residual = _metadata(
            ("a", "b", "c"),
            ("b", "c"),
            features=("solver_pd",),
            target="torque_residual",
            kp=[10.0, 20.0, 30.0],
            kd=[1.0, 2.0, 3.0],
            dry=[0.1, 0.2],
            viscous=[0.3, 0.4],
        )
        residual_path = self._save(residual)
        configs = ControllerNeuralGRU.resolve_joint_configurations(
            {"model_path": residual_path, "mapping_index": 0}, output_count=2
        )
        self.assertEqual(
            [(item.target_ke, item.target_kd, item.dry_friction, item.viscous_friction) for item in configs],
            [(20.0, 2.0, 0.1, 0.3), (30.0, 3.0, 0.2, 0.4)],
        )

        template = newton.ModelBuilder()
        links = [template.add_link() for _ in range(3)]
        joints = [
            template.add_joint_revolute(parent=-1 if index == 0 else links[index - 1], child=link, axis=newton.Axis.Z)
            for index, link in enumerate(links)
        ]
        template.add_articulation(joints)
        dofs = [template.joint_qd_start[joint] for joint in joints]
        coords = [template.joint_q_start[joint] for joint in joints]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="add_actuator_group:.*overwrote")
            template.add_actuator_group(
                ControllerNeuralGRU,
                input_indices=dofs,
                output_indices=dofs[1:],
                input_pos_indices=coords,
                output_pos_indices=coords[1:],
                model_path=residual_path,
                mapping_index=0,
            )
        np.testing.assert_allclose(np.asarray(template.joint_target_ke)[dofs[1:]], [20.0, 30.0])
        np.testing.assert_allclose(np.asarray(template.joint_target_kd)[dofs[1:]], [2.0, 3.0])
        self.assertEqual(
            [template.joint_target_mode[dof] for dof in dofs[1:]],
            [int(newton.JointTargetMode.POSITION)] * 2,
        )
        np.testing.assert_allclose(np.asarray(template.joint_friction)[dofs[1:]], [0.1, 0.2])
        np.testing.assert_allclose(np.asarray(template.joint_damping)[dofs[1:]], [0.3, 0.4])

        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        actuator = builder.finalize(device=self.device).actuators[0]
        np.testing.assert_array_equal(actuator.input_indices.numpy(), [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(actuator.indices.numpy(), [1, 2, 4, 5])

    def test_timestep_must_match_artifact(self):
        """Reject control timesteps that differ from the training sample period."""
        controller = ControllerNeuralGRU(self._save(_metadata()))
        controller.finalize(self.device, output_count=1)
        state = controller.state(output_count=1, device=self.device)
        values = self._arrays([0.0])
        indices = self._arrays([0], dtype=wp.uint32)
        compute_args = (
            values,
            values,
            values,
            values,
            None,
            indices,
            indices,
            indices,
            indices,
            self._arrays([0.0]),
            state,
        )
        controller.compute(*compute_args, 0.002, self.device)
        with self.assertRaisesRegex(ValueError, "does not match model sample_dt_s"):
            controller.compute(*compute_args, 0.001, self.device)

    def test_malformed_schema_v3_contracts_are_rejected(self):
        """Reject inconsistent topology, dimensions, provenance, and statistics."""
        base = _metadata(("a", "b"), ("b",), features=("position", "previous_torque"))
        cases = []

        bad = copy.deepcopy(base)
        bad["joint_mappings"][0]["output_joints"] = ["missing"]
        cases.append((bad, "subset of input_joints"))
        bad = copy.deepcopy(base)
        bad["input_size"] += 1
        cases.append((bad, "input_size.*must be"))
        bad = copy.deepcopy(base)
        bad["input_feature_specs"][0]["domain"] = "output_joints"
        cases.append((bad, "must use domain 'input_joints'"))
        bad = copy.deepcopy(base)
        bad["previous_torque_derivation"]["runtime_source"] = "teacher_forced"
        cases.append((bad, "runtime_source"))
        bad = copy.deepcopy(base)
        bad["normalization"]["inputs"]["std"]["position"] = [1.0, 0.0]
        cases.append((bad, "greater than zero"))

        for metadata, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ControllerNeuralGRU.resolve_arguments({"model_path": self._save(metadata), "mapping_index": 0})


if __name__ == "__main__":
    unittest.main()
