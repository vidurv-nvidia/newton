# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import load_checkpoint, load_metadata
from .base import Controller

if typing.TYPE_CHECKING:
    import torch


_SUPPORTED_FEATURES = {
    "dynamic_bias",
    "position",
    "position_error",
    "previous_torque",
    "solver_pd",
    "velocity",
}
_SUPPORTED_TARGETS = {"torque", "torque_residual"}


@dataclass(frozen=True)
class _JointMapping:
    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]


@dataclass(frozen=True)
class _InputFeatureSpec:
    name: str
    domain: str
    channels: tuple[str, ...]
    width: int


@dataclass(frozen=True)
class _GRUMetadata:
    joint_mappings: tuple[_JointMapping, ...]
    input_columns: tuple[str, ...]
    input_feature_specs: tuple[_InputFeatureSpec, ...]
    input_width: int
    output_width: int
    input_size: int
    output_size: int
    target: str
    sample_dt_s: float
    input_mean: tuple[float, ...]
    input_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    pd_domain: str | None
    kp: tuple[float, ...] | None
    kd: tuple[float, ...] | None
    dry_friction: tuple[float, ...] | None
    viscous_friction: tuple[float, ...] | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"GRU metadata '{name}' must be an object")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"GRU metadata '{name}' must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GRU metadata '{name}' must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"GRU metadata '{name}' must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"GRU metadata '{name}' must be greater than zero")
    if nonnegative and result < 0.0:
        raise ValueError(f"GRU metadata '{name}' must be nonnegative")
    return result


def _finite_values(
    value: Any,
    name: str,
    count: int,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if count == 1 and not isinstance(value, list):
        return (_finite_float(value, name, positive=positive, nonnegative=nonnegative),)
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"GRU metadata '{name}' must be a list of {count} numbers")
    return tuple(
        _finite_float(item, f"{name}[{index}]", positive=positive, nonnegative=nonnegative)
        for index, item in enumerate(value)
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"GRU metadata '{name}' must be a positive integer")
    return value


def _joint_names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"GRU metadata '{name}' must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"GRU metadata '{name}' entries must be non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"GRU metadata '{name}' must not contain duplicates")
    return result


def _parse_metadata(metadata: dict[str, Any], model_path: str) -> _GRUMetadata:
    if not metadata:
        raise ValueError(f"GRU checkpoint at '{model_path}' has no embedded metadata.json")
    schema_version = metadata.get("schema_version")
    if schema_version != 3:
        raise ValueError(
            f"GRU checkpoint at '{model_path}' has unsupported schema_version {schema_version!r}; expected 3"
        )
    if metadata.get("model_type") != "gru":
        raise ValueError(
            f"GRU checkpoint at '{model_path}' has model_type {metadata.get('model_type')!r}; expected 'gru'"
        )

    input_columns_value = metadata.get("input_columns")
    if not isinstance(input_columns_value, list) or not input_columns_value:
        raise ValueError("GRU metadata 'input_columns' must be a non-empty list")
    if not all(isinstance(column, str) for column in input_columns_value):
        raise ValueError("GRU metadata 'input_columns' entries must be strings")
    input_columns = tuple(input_columns_value)
    if len(set(input_columns)) != len(input_columns):
        raise ValueError("GRU metadata 'input_columns' must not contain duplicates")
    unsupported = sorted(set(input_columns) - _SUPPORTED_FEATURES)
    if unsupported:
        raise ValueError(f"GRU metadata contains unsupported input features: {', '.join(unsupported)}")

    target_columns = metadata.get("target_columns")
    if not isinstance(target_columns, list) or len(target_columns) != 1:
        raise ValueError("GRU metadata 'target_columns' must contain exactly one target")
    target = target_columns[0]
    if target not in _SUPPORTED_TARGETS:
        raise ValueError(f"GRU metadata target {target!r} is unsupported; expected one of {sorted(_SUPPORTED_TARGETS)}")

    mappings_value = metadata.get("joint_mappings")
    if not isinstance(mappings_value, list) or not mappings_value:
        raise ValueError("GRU metadata 'joint_mappings' must be a non-empty list")
    parsed_mappings: list[_JointMapping] = []
    for index, value in enumerate(mappings_value):
        item = _mapping(value, f"joint_mappings[{index}]")
        inputs = _joint_names(item.get("input_joints"), f"joint_mappings[{index}].input_joints")
        outputs = _joint_names(item.get("output_joints"), f"joint_mappings[{index}].output_joints")
        missing = [name for name in outputs if name not in inputs]
        if missing:
            raise ValueError(
                f"GRU metadata 'joint_mappings[{index}].output_joints' must be a subset of input_joints: "
                + ", ".join(missing)
            )
        parsed_mappings.append(_JointMapping(inputs, outputs))
    joint_mappings = tuple(parsed_mappings)
    identities = [(item.input_joints, item.output_joints) for item in joint_mappings]
    if len(set(identities)) != len(identities):
        raise ValueError("GRU metadata 'joint_mappings' must not contain duplicates")
    input_width = len(joint_mappings[0].input_joints)
    output_width = len(joint_mappings[0].output_joints)
    if any(len(item.input_joints) != input_width or len(item.output_joints) != output_width for item in joint_mappings):
        raise ValueError("GRU metadata 'joint_mappings' must have identical input and output widths")

    expected_topology = "siso" if input_width == output_width == 1 else ("miso" if output_width == 1 else "mimo")
    if metadata.get("training_topology") != expected_topology:
        raise ValueError(
            f"GRU metadata 'training_topology' must be {expected_topology!r} for "
            f"{input_width}-to-{output_width} mappings"
        )

    specs_value = metadata.get("input_feature_specs")
    if not isinstance(specs_value, list) or len(specs_value) != len(input_columns):
        raise ValueError("GRU metadata 'input_feature_specs' must contain one entry for every input column")
    parsed_specs: list[_InputFeatureSpec] = []
    for index, (value, column) in enumerate(zip(specs_value, input_columns, strict=True)):
        item = _mapping(value, f"input_feature_specs[{index}]")
        if item.get("name") != column:
            raise ValueError(
                f"GRU metadata 'input_feature_specs[{index}].name' must match input_columns[{index}]={column!r}"
            )
        expected_domain = "output_joints" if column == "previous_torque" else "input_joints"
        domain = item.get("domain")
        if domain != expected_domain:
            raise ValueError(f"GRU metadata input feature {column!r} must use domain {expected_domain!r}")
        channels = item.get("channels")
        if channels != ["value"]:
            raise ValueError(f"GRU metadata input feature {column!r} supports only channels=['value']")
        width = output_width if domain == "output_joints" else input_width
        parsed_specs.append(_InputFeatureSpec(column, domain, ("value",), width))
    input_feature_specs = tuple(parsed_specs)
    input_size = _positive_int(metadata.get("input_size"), "input_size")
    expected_input_size = sum(spec.width for spec in input_feature_specs)
    if input_size != expected_input_size:
        raise ValueError(
            f"GRU metadata 'input_size' must be {expected_input_size} from input_feature_specs; got {input_size}"
        )
    output_size = _positive_int(metadata.get("output_size"), "output_size")
    if output_size != output_width:
        raise ValueError(
            f"GRU metadata 'output_size' must match the mapping output width {output_width}; got {output_size}"
        )

    if "previous_torque" in input_columns:
        derivation = _mapping(metadata.get("previous_torque_derivation"), "previous_torque_derivation")
        expected_derivation = {
            "source": "previous_raw_network_output_pre_clamp",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset_boundaries": ["recorded_episode", "joint_mapping"],
            "value_space": "residual_torque" if target == "torque_residual" else "physical_torque",
        }
        for key, expected in expected_derivation.items():
            if derivation.get(key) != expected:
                raise ValueError(f"GRU metadata 'previous_torque_derivation.{key}' must be {expected!r}")

    sample_dt_s = _finite_float(metadata.get("sample_dt_s"), "sample_dt_s", positive=True)
    normalization = _mapping(metadata.get("normalization"), "normalization")
    inputs = _mapping(normalization.get("inputs"), "normalization.inputs")
    input_means = _mapping(inputs.get("mean"), "normalization.inputs.mean")
    input_stds = _mapping(inputs.get("std"), "normalization.inputs.std")
    targets = _mapping(normalization.get("targets"), "normalization.targets")
    target_means = _mapping(targets.get("mean"), "normalization.targets.mean")
    target_stds = _mapping(targets.get("std"), "normalization.targets.std")
    input_mean = tuple(
        value
        for spec in input_feature_specs
        for value in _finite_values(
            input_means.get(spec.name),
            f"normalization.inputs.mean.{spec.name}",
            spec.width,
        )
    )
    input_std = tuple(
        value
        for spec in input_feature_specs
        for value in _finite_values(
            input_stds.get(spec.name),
            f"normalization.inputs.std.{spec.name}",
            spec.width,
            positive=True,
        )
    )
    target_mean = _finite_values(
        target_means.get(target),
        f"normalization.targets.mean.{target}",
        output_width,
    )
    target_std = _finite_values(
        target_stds.get(target),
        f"normalization.targets.std.{target}",
        output_width,
        positive=True,
    )

    delay = _mapping(metadata.get("delay"), "delay")
    if delay.get("handling") != "learned":
        raise ValueError("GRU metadata 'delay.handling' must be 'learned'")
    external_delay_s = _finite_float(
        delay.get("external_delay_s"),
        "delay.external_delay_s",
        nonnegative=True,
    )
    if external_delay_s != 0.0:
        raise ValueError("GRU metadata with delay.handling='learned' must have external_delay_s=0")

    pd_domain = None
    kp = kd = None
    needs_pd = target == "torque_residual" or "solver_pd" in input_columns
    if needs_pd:
        pd_baseline = _mapping(metadata.get("pd_baseline"), "pd_baseline")
        expected_pd_domain = "input_joints" if "solver_pd" in input_columns else "output_joints"
        pd_domain = pd_baseline.get("domain")
        if pd_domain != expected_pd_domain:
            raise ValueError(f"GRU metadata 'pd_baseline.domain' must be {expected_pd_domain!r}")
        pd_width = input_width if pd_domain == "input_joints" else output_width
        kp = _finite_values(pd_baseline.get("kp"), "pd_baseline.kp", pd_width, nonnegative=True)
        kd = _finite_values(pd_baseline.get("kd"), "pd_baseline.kd", pd_width, nonnegative=True)
        if pd_baseline.get("velocity_target") != "zero":
            raise ValueError("GRU metadata supports only pd_baseline.velocity_target='zero'")

    dry_friction = viscous_friction = None
    if target == "torque_residual":
        friction = _mapping(metadata.get("friction_baseline"), "friction_baseline")
        if friction.get("domain") != "output_joints":
            raise ValueError("GRU metadata 'friction_baseline.domain' must be 'output_joints'")
        dry_friction = _finite_values(friction.get("dry"), "friction_baseline.dry", output_width, nonnegative=True)
        viscous_friction = _finite_values(
            friction.get("viscous"),
            "friction_baseline.viscous",
            output_width,
            nonnegative=True,
        )

    return _GRUMetadata(
        joint_mappings=joint_mappings,
        input_columns=input_columns,
        input_feature_specs=input_feature_specs,
        input_width=input_width,
        output_width=output_width,
        input_size=input_size,
        output_size=output_size,
        target=target,
        sample_dt_s=sample_dt_s,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        pd_domain=pd_domain,
        kp=kp,
        kd=kd,
        dry_friction=dry_friction,
        viscous_friction=viscous_friction,
    )


def _resolve_mapping_index(metadata: _GRUMetadata, value: Any) -> int:
    if value is None:
        if len(metadata.joint_mappings) > 1:
            raise ValueError("ControllerNeuralGRU requires 'mapping_index' for artifacts with multiple joint_mappings")
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("ControllerNeuralGRU 'mapping_index' must be an integer")
    if value < 0 or value >= len(metadata.joint_mappings):
        raise ValueError(
            f"ControllerNeuralGRU 'mapping_index' {value} is out of range for "
            f"{len(metadata.joint_mappings)} joint mapping(s)"
        )
    return value


def _resolve_group_mapping_indices(
    metadata: _GRUMetadata,
    value: int | None | wp.array,
    group_count: int,
) -> tuple[int, ...]:
    """Resolve one mapping index per recurrent group.

    ModelBuilder authors array parameters once per output DOF, so each mapping
    index is repeated ``output_width`` times. Direct scalar construction applies
    one mapping to every group.
    """
    if not isinstance(value, wp.array):
        mapping_index = _resolve_mapping_index(metadata, value)
        return (mapping_index,) * group_count

    if value.ndim != 1:
        raise ValueError("ControllerNeuralGRU 'mapping_index' array must be one-dimensional")
    expected_count = group_count * metadata.output_width
    if len(value) != expected_count:
        raise ValueError(
            f"ControllerNeuralGRU 'mapping_index' array must contain {expected_count} values "
            f"({metadata.output_width} per group); got {len(value)}"
        )

    authored_values = value.numpy()
    mapping_indices: list[int] = []
    for group_index in range(group_count):
        group_values = authored_values[group_index * metadata.output_width : (group_index + 1) * metadata.output_width]
        resolved_group: list[int] = []
        for output_index, authored_value in enumerate(group_values):
            numeric_value = float(authored_value)
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(
                    "ControllerNeuralGRU 'mapping_index' array values must be finite integers; "
                    f"got {authored_value!r} at output {group_index * metadata.output_width + output_index}"
                )
            resolved_group.append(_resolve_mapping_index(metadata, int(numeric_value)))
        if len(set(resolved_group)) != 1:
            raise ValueError("ControllerNeuralGRU 'mapping_index' must be repeated across every output in a group")
        mapping_indices.append(resolved_group[0])
    return tuple(mapping_indices)


class ControllerNeuralGRU(Controller):
    """Stateful GRU controller with an embedded runtime contract.

    The TorchScript archive must contain Anchor schema-v3 metadata describing
    feature order, normalization, output semantics, training timestep, learned
    delay, joint mappings, and any analytical PD/friction baseline. The selected
    mapping's feature widths are determined by each ``input_feature_specs``
    domain. Feature blocks are flattened in
    ``input_columns`` order. The network input and output shapes are
    ``(G, 1, input_size)`` and ``(G, 1, output_size)``; hidden state has shape
    ``(L, G, H)``.

    ``dynamic_bias`` is supplied explicitly to :meth:`Actuator.step` and is
    never added directly to effort. ``previous_torque`` is the preceding raw
    network output before clamping and is initialized to zero.

    Args:
        model_path: Path to a TorchScript GRU archive.
        mapping_index: Joint mapping to execute. It may be omitted only when
            the artifact contains one mapping.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}
    PER_GROUP_PARAMS: ClassVar[set[str]] = {"mapping_index"}
    supports_external_delay: ClassVar[bool] = False

    @dataclass
    class State(Controller.State):
        """GRU hidden state and previous raw network effort."""

        hidden: torch.Tensor | None = None
        """Hidden state with shape (num_layers, group_count, hidden_size)."""

        previous_effort: torch.Tensor | None = None
        """Previous raw effort with shape (group_count, output_width)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            hidden = self.hidden
            previous_effort = self.previous_effort
            if mask is None:
                hidden.zero_()
                previous_effort.zero_()
            else:
                torch_mask = wp.to_torch(mask).bool()
                group_count, joint_count = previous_effort.shape
                if len(torch_mask) == group_count:
                    group_mask = torch_mask
                    previous_effort[group_mask, :] = 0.0
                elif len(torch_mask) == group_count * joint_count:
                    effort_mask = torch_mask.reshape(group_count, joint_count)
                    group_mask = effort_mask.all(dim=1)
                    if (effort_mask.any(dim=1) != group_mask).any().item():
                        raise ValueError("ControllerNeuralGRU reset mask must select complete joint groups")
                    previous_effort[group_mask, :] = 0.0
                else:
                    raise ValueError(
                        "ControllerNeuralGRU reset mask length must match the group count or the actuator count"
                    )
                hidden[:, group_mask, :] = 0.0

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "model_path" not in args:
            raise ValueError("ControllerNeuralGRU requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("ControllerNeuralGRU requires a non-empty 'model_path'")
        metadata = _parse_metadata(load_metadata(model_path), model_path)
        mapping_index = _resolve_mapping_index(metadata, args.get("mapping_index"))
        return {"model_path": model_path, "mapping_index": mapping_index}

    @classmethod
    def validate_resolved_group(
        cls,
        args: dict[str, Any],
        input_indices: typing.Sequence[int],
        output_indices: typing.Sequence[int],
    ) -> None:
        """Validate one authored group against its selected physical mapping."""
        model_path = args["model_path"]
        metadata = _parse_metadata(load_metadata(model_path), model_path)
        mapping_index = _resolve_mapping_index(metadata, args.get("mapping_index"))
        if len(input_indices) != metadata.input_width or len(output_indices) != metadata.output_width:
            raise ValueError(
                f"ControllerNeuralGRU mapping {mapping_index} requires {metadata.input_width} input and "
                f"{metadata.output_width} output indices; got {len(input_indices)} and {len(output_indices)}"
            )

        mapping = metadata.joint_mappings[mapping_index]
        output_input_slots = tuple(mapping.input_joints.index(name) for name in mapping.output_joints)
        for output_slot, input_slot in enumerate(output_input_slots):
            if output_indices[output_slot] != input_indices[input_slot]:
                raise ValueError(
                    f"ControllerNeuralGRU mapping {mapping_index} output slot {output_slot} must reference "
                    f"input slot {input_slot} (physical index {input_indices[input_slot]}); "
                    f"got physical index {output_indices[output_slot]}"
                )

    @classmethod
    def resolve_joint_configurations(
        cls,
        args: dict[str, Any],
        output_count: int,
    ) -> tuple[Controller.JointConfiguration, ...] | None:
        """Resolve one joint configuration per selected mapping output."""
        model_path = args["model_path"]
        metadata = _parse_metadata(load_metadata(model_path), model_path)
        mapping_index = _resolve_mapping_index(metadata, args.get("mapping_index"))
        if output_count != metadata.output_width:
            raise ValueError(
                f"ControllerNeuralGRU expected {metadata.output_width} output configuration(s) "
                f"for one mapping group; got {output_count}"
            )
        if metadata.target == "torque":
            return tuple(Controller.JointConfiguration(target_ke=0.0, target_kd=0.0) for _ in range(output_count))

        assert metadata.kp is not None and metadata.kd is not None
        assert metadata.dry_friction is not None and metadata.viscous_friction is not None
        mapping = metadata.joint_mappings[mapping_index]
        if metadata.pd_domain == "input_joints":
            pd_slots = tuple(mapping.input_joints.index(name) for name in mapping.output_joints)
        else:
            pd_slots = tuple(range(output_count))
        return tuple(
            Controller.JointConfiguration(
                target_ke=metadata.kp[pd_slot],
                target_kd=metadata.kd[pd_slot],
                dry_friction=metadata.dry_friction[output_slot],
                viscous_friction=metadata.viscous_friction[output_slot],
            )
            for output_slot, pd_slot in enumerate(pd_slots)
        )

    def __init__(self, model_path: str, mapping_index: int | wp.array | None = None):
        import torch

        self.model_path = model_path
        self._torch_device = torch.device("cpu")
        self.network, raw_metadata = load_checkpoint(model_path)
        self.metadata = _parse_metadata(raw_metadata, model_path)
        self.requires_dynamic_bias = "dynamic_bias" in self.metadata.input_columns
        self.mapping_index = mapping_index
        self.mapping_indices: tuple[int, ...] = ()
        if not isinstance(mapping_index, wp.array):
            self.mapping_index = _resolve_mapping_index(self.metadata, mapping_index)
            self.joint_mapping = self.metadata.joint_mappings[self.mapping_index]
            self.input_joints = self.joint_mapping.input_joints
            self.output_joints = self.joint_mapping.output_joints

        if not hasattr(self.network, "gru"):
            raise ValueError("network must expose a 'gru' attribute (torch.nn.GRU)")
        gru = self.network.gru
        gru_parameters = dict(gru.named_parameters())
        layer_indices = set()
        bidirectional = False
        for name in gru_parameters:
            if not name.startswith("weight_ih_l"):
                continue
            suffix = name.removeprefix("weight_ih_l")
            if suffix.endswith("_reverse"):
                bidirectional = True
                suffix = suffix.removesuffix("_reverse")
            if suffix.isdigit():
                layer_indices.add(int(suffix))
        if not layer_indices or layer_indices != set(range(max(layer_indices) + 1)):
            raise ValueError("network.gru must expose contiguous torch.nn.GRU layer parameters")
        if bidirectional:
            raise ValueError("network.gru must not be bidirectional")

        weight_ih_l0 = gru_parameters["weight_ih_l0"]
        weight_hh_l0 = gru_parameters.get("weight_hh_l0")
        if weight_hh_l0 is None or weight_ih_l0.ndim != 2 or weight_hh_l0.ndim != 2:
            raise ValueError("network.gru has invalid torch.nn.GRU parameter shapes")
        input_size = int(weight_ih_l0.shape[1])
        hidden_size = int(weight_hh_l0.shape[1])
        expected_gate_rows = 3 * hidden_size
        if (
            tuple(weight_hh_l0.shape) != (expected_gate_rows, hidden_size)
            or int(weight_ih_l0.shape[0]) != expected_gate_rows
        ):
            raise ValueError("network.gru has invalid torch.nn.GRU gate dimensions")

        self._input_width = self.metadata.input_width
        self._output_width = self.metadata.output_width
        expected_input_size = self.metadata.input_size
        if input_size != expected_input_size:
            raise ValueError(
                f"network.gru.input_size must match metadata input_size {expected_input_size}; got {input_size}"
            )

        self._num_layers = len(layer_indices)
        self._hidden_size = hidden_size
        test_group_count = 2
        with torch.inference_mode():
            try:
                test_output, test_hidden = self.network(
                    torch.zeros(test_group_count, 1, expected_input_size),
                    torch.zeros(self._num_layers, test_group_count, self._hidden_size),
                )
            except Exception as exc:
                raise ValueError("ControllerNeuralGRU could not evaluate the batch-first network interface") from exc
        expected_test_output_shape = (test_group_count, 1, self.metadata.output_size)
        if tuple(test_output.shape) != expected_test_output_shape:
            raise ValueError(
                f"ControllerNeuralGRU network output shape must be {expected_test_output_shape} for "
                f"{test_group_count} groups; got {tuple(test_output.shape)}"
            )
        expected_test_hidden_shape = (self._num_layers, test_group_count, self._hidden_size)
        if tuple(test_hidden.shape) != expected_test_hidden_shape:
            raise ValueError(
                f"ControllerNeuralGRU hidden shape must be {expected_test_hidden_shape} for "
                f"{test_group_count} groups; got {tuple(test_hidden.shape)}"
            )

        self._group_count = 0
        self._torch_pos_indices: torch.Tensor | None = None
        self._torch_vel_indices: torch.Tensor | None = None
        self._torch_target_pos_indices: torch.Tensor | None = None
        self._input_mean: torch.Tensor | None = None
        self._input_std: torch.Tensor | None = None
        self._target_mean: torch.Tensor | None = None
        self._target_std: torch.Tensor | None = None
        self._kp: torch.Tensor | None = None
        self._kd: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._previous_effort: torch.Tensor | None = None

    def validate_io(self, input_count: int, output_count: int) -> None:
        if input_count % self._input_width != 0:
            raise ValueError(
                f"ControllerNeuralGRU input count {input_count} is not divisible by "
                f"mapping input width {self._input_width}"
            )
        if output_count % self._output_width != 0:
            raise ValueError(
                f"ControllerNeuralGRU output count {output_count} is not divisible by "
                f"mapping output width {self._output_width}"
            )
        input_groups = input_count // self._input_width
        output_groups = output_count // self._output_width
        if input_groups != output_groups:
            raise ValueError(
                f"ControllerNeuralGRU input/output counts describe different group counts: "
                f"{input_groups} input group(s) and {output_groups} output group(s)"
            )

    def _output_group_count(self, output_count: int) -> int:
        if output_count % self._output_width != 0:
            raise ValueError(
                f"ControllerNeuralGRU output count {output_count} is not divisible by "
                f"mapping output width {self._output_width}"
            )
        return output_count // self._output_width

    def finalize(self, device: wp.Device, output_count: int) -> None:
        import torch

        self._group_count = self._output_group_count(output_count)
        self.mapping_indices = _resolve_group_mapping_indices(
            self.metadata,
            self.mapping_index,
            self._group_count,
        )
        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
        self.network = self.network.to(self._torch_device)
        self._input_mean = torch.tensor(self.metadata.input_mean, dtype=torch.float32, device=self._torch_device)
        self._input_std = torch.tensor(self.metadata.input_std, dtype=torch.float32, device=self._torch_device)
        self._target_mean = torch.tensor(self.metadata.target_mean, dtype=torch.float32, device=self._torch_device)
        self._target_std = torch.tensor(self.metadata.target_std, dtype=torch.float32, device=self._torch_device)
        if self.metadata.kp is not None:
            self._kp = torch.tensor(self.metadata.kp, dtype=torch.float32, device=self._torch_device)
            self._kd = torch.tensor(self.metadata.kd, dtype=torch.float32, device=self._torch_device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, output_count: int, device: wp.Device) -> ControllerNeuralGRU.State:
        import torch

        group_count = self._output_group_count(output_count)
        return ControllerNeuralGRU.State(
            hidden=torch.zeros(self._num_layers, group_count, self._hidden_size, device=self._torch_device),
            previous_effort=torch.zeros(group_count, self._output_width, device=self._torch_device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerNeuralGRU.State,
        dt: float,
        device: wp.Device | None = None,
        dynamic_bias: wp.array[float] | None = None,
    ) -> None:
        import torch

        if not math.isclose(dt, self.metadata.sample_dt_s, rel_tol=1.0e-5, abs_tol=1.0e-9):
            raise ValueError(
                f"ControllerNeuralGRU timestep {dt} s does not match model sample_dt_s {self.metadata.sample_dt_s} s"
            )

        input_count = len(pos_indices)
        output_count = len(forces)
        self.validate_io(input_count, output_count)
        for name, indices in (
            ("vel_indices", vel_indices),
            ("target_pos_indices", target_pos_indices),
            ("target_vel_indices", target_vel_indices),
        ):
            if len(indices) != input_count:
                raise ValueError(
                    f"ControllerNeuralGRU {name} length {len(indices)} must match input count {input_count}"
                )
        group_count = output_count // self._output_width
        if group_count != self._group_count:
            raise ValueError(
                f"ControllerNeuralGRU received {group_count} group(s), but was finalized for "
                f"{self._group_count} group(s)"
            )

        if self._torch_pos_indices is None:
            self._torch_pos_indices = wp.to_torch(pos_indices).to(dtype=torch.long)
            self._torch_vel_indices = wp.to_torch(vel_indices).to(dtype=torch.long)
            self._torch_target_pos_indices = wp.to_torch(target_pos_indices).to(dtype=torch.long)

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target_p = wp.to_torch(target_pos)
        position = current_pos[self._torch_pos_indices].reshape(self._group_count, self._input_width)
        velocity = current_vel[self._torch_vel_indices].reshape(self._group_count, self._input_width)
        position_error = (
            target_p[self._torch_target_pos_indices].reshape(self._group_count, self._input_width) - position
        )
        feature_values = {
            "position": position,
            "position_error": position_error,
            "previous_torque": state.previous_effort,
            "velocity": velocity,
        }
        if "dynamic_bias" in self.metadata.input_columns:
            if dynamic_bias is None:
                raise ValueError(
                    "ControllerNeuralGRU input feature 'dynamic_bias' requires dynamic_bias values in Actuator.step"
                )
            feature_values["dynamic_bias"] = wp.to_torch(dynamic_bias)[self._torch_vel_indices].reshape(
                self._group_count, self._input_width
            )

        if "solver_pd" in self.metadata.input_columns:
            feature_values["solver_pd"] = self._kp * position_error - self._kd * velocity
        expected_previous_shape = (self._group_count, self._output_width)
        if tuple(state.previous_effort.shape) != expected_previous_shape:
            raise ValueError(
                f"ControllerNeuralGRU previous effort shape must be {expected_previous_shape}; "
                f"got {tuple(state.previous_effort.shape)}"
            )
        raw_input = torch.cat([feature_values[spec.name] for spec in self.metadata.input_feature_specs], dim=1)
        if tuple(raw_input.shape) != (self._group_count, self.metadata.input_size):
            raise ValueError(
                f"ControllerNeuralGRU assembled input shape must be "
                f"{(self._group_count, self.metadata.input_size)}; got {tuple(raw_input.shape)}"
            )
        net_input = ((raw_input - self._input_mean) / self._input_std).unsqueeze(1)

        with torch.inference_mode():
            effort, self._hidden = self.network(net_input, state.hidden)

        expected_effort_shape = (self._group_count, 1, self._output_width)
        if tuple(effort.shape) != expected_effort_shape:
            raise ValueError(
                f"ControllerNeuralGRU network output shape must be {expected_effort_shape}; got {tuple(effort.shape)}"
            )
        expected_hidden_shape = (self._num_layers, self._group_count, self._hidden_size)
        if tuple(self._hidden.shape) != expected_hidden_shape:
            raise ValueError(
                f"ControllerNeuralGRU hidden shape must be {expected_hidden_shape}; got {tuple(self._hidden.shape)}"
            )

        effort = effort[:, 0, :] * self._target_std + self._target_mean
        self._previous_effort = effort
        effort_wp = wp.from_torch(effort.reshape(-1).contiguous(), dtype=wp.float32)
        wp.copy(forces, effort_wp)

    def update_state(
        self,
        current_state: ControllerNeuralGRU.State,
        next_state: ControllerNeuralGRU.State,
    ) -> None:
        if next_state is not None:
            next_state.hidden = self._hidden
            next_state.previous_effort = self._previous_effort
