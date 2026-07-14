# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp


@wp.kernel
def _masked_zero_1d(data: wp.array[float], mask: wp.array[wp.bool]):
    i = wp.tid()
    if mask[i]:
        data[i] = 0.0


class Controller:
    """Base class for actuator control laws.

    Control laws compute actuator output effort from authored controller
    parameters, commanded inputs (targets, feedforward), and simulation
    state. The output may still be constrained by one or more
    :class:`~newton.actuators.Clamping` objects.

    Subclasses must override ``compute`` and ``resolve_arguments``.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values (e.g. ``kp >= 0``) before they are batched into Warp
    arrays.  ``__init__`` receives pre-built arrays and validates shapes
    only — reading back array contents for value checks would force a
    synchronous device-to-host copy on every construction.
    """

    @dataclass
    class State:
        """Base state for controllers.

        Subclass this in concrete controllers that maintain internal
        state (e.g. integral accumulators, history buffers).
        """

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset state to initial values.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """

    @dataclass(frozen=True)
    class JointConfiguration:
        """Joint properties a controller requires ModelBuilder to author."""

        target_ke: float | None = None
        target_kd: float | None = None
        dry_friction: float | None = None
        viscous_friction: float | None = None

    SHARED_PARAMS: ClassVar[set[str]] = set()

    requires_dynamic_bias: bool = False
    """Whether :class:`Actuator` should pass ``dynamic_bias`` to :meth:`compute`."""

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    @classmethod
    def resolve_joint_configuration(
        cls,
        args: dict[str, Any],
        delay_steps: int | None = None,
    ) -> JointConfiguration | None:
        """Return joint properties required by resolved controller arguments.

        This optional build-time hook is called while the target DOF is still
        mutable. Controllers that do not require joint configuration changes
        return None.

        Args:
            args: Arguments returned by resolve_arguments.
            delay_steps: Authored external actuator delay, if any.

        Returns:
            Required joint configuration, or None.
        """
        return None

    @classmethod
    def resolve_joint_configurations(
        cls,
        args: dict[str, Any],
        output_count: int,
        delay_steps: int | None = None,
    ) -> tuple[JointConfiguration, ...] | None:
        """Return ordered joint properties for an actuator output group.

        The default repeats the scalar configuration returned by
        :meth:`resolve_joint_configuration`. Controllers with slot-specific
        output properties may override this hook.

        Args:
            args: Arguments returned by :meth:`resolve_arguments`.
            output_count: Number of output DOFs in the authored group.
            delay_steps: Authored external actuator delay, if any.

        Returns:
            One configuration per output DOF, or ``None``.
        """
        configuration = cls.resolve_joint_configuration(args, delay_steps)
        if configuration is None:
            return None
        return (configuration,) * output_count

    @classmethod
    def validate_resolved_group(
        cls,
        args: dict[str, Any],
        input_indices: Sequence[int],
        output_indices: Sequence[int],
    ) -> None:
        """Validate one resolved controller group before authoring joint properties.

        Args:
            args: Arguments returned by :meth:`resolve_arguments`.
            input_indices: Ordered DOF indices read by the controller.
            output_indices: Ordered DOF indices receiving controller effort.
        """

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Called by :class:`Actuator` after construction to set up device-specific resources.

        Override in subclasses that need to place tensors or networks
        on a specific device, or pre-compute index tensors.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs) this controller manages.
        """
        pass

    def validate_io(self, input_count: int, output_count: int) -> None:
        """Validate the number of controller inputs and actuator outputs.

        Controllers whose input and output axes have different sizes must
        override this method and validate their own shape contract.

        Args:
            input_count: Number of state and target inputs read by the controller.
            output_count: Number of efforts written by the controller.

        Raises:
            ValueError: If the input and output counts differ.
        """
        if input_count != output_count:
            raise ValueError(
                f"{type(self).__name__} requires equal input and output counts; "
                f"got {input_count} inputs and {output_count} outputs"
            )

    def validate_delay(self, delay: Any | None) -> None:
        """Validate an optional runtime Delay component.

        Controllers whose artifact contract constrains delay may override this
        hook. The default accepts either presence or absence of delay.

        Args:
            delay: Runtime Delay component, or ``None``.
        """

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
        state: Controller.State | None,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        """Compute actuator output effort and write to ``forces[i]``.

        Args:
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            target_pos: Target positions [m or rad].
            target_vel: Target velocities [m/s or rad/s].
            feedforward: Feedforward effort [N or N·m] (may be ``None``).
            pos_indices: Input-axis indices into *positions*.
            vel_indices: Input-axis indices into *velocities*.
            target_pos_indices: Input-axis indices into *target_pos*.
            target_vel_indices: Input-axis indices into *target_vel* and
                *feedforward*.
            forces: Output-axis scratch buffer to write effort [N or N·m].
                Shape ``(output_count,)``.
            state: Controller state (``None`` if stateless).
            dt: Timestep [s].
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")

    def is_stateful(self) -> bool:
        """Return True if this controller maintains internal state."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_stateful")

    def is_graphable(self) -> bool:
        """Return True if compute() can be captured in a CUDA graph."""
        raise NotImplementedError(f"{type(self).__name__} must implement is_graphable")

    def state(self, num_actuators: int, device: wp.Device) -> Controller.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        current_state: Controller.State,
        next_state: Controller.State,
    ) -> None:
        """Advance internal state after a compute step.

        Args:
            current_state: Current controller state.
            next_state: Next controller state to write.
        """
        pass
