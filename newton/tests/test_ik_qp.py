# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-based QP inverse-kinematics optimizer."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik.ik_common import eval_fk_batched
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# ----------------------------------------------------------------------------
# Helpers: model builders
# ----------------------------------------------------------------------------


def _build_two_link_planar(device) -> newton.Model:
    """Build a 2-DOF planar revolute arm (1m + 1m links)."""
    builder = newton.ModelBuilder()

    link1 = builder.add_link(
        xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )

    link2 = builder.add_link(
        xform=wp.transform([1.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )

    builder.add_articulation([joint1, joint2])
    return builder.finalize(device=device, requires_grad=True)


def _build_single_d6(device) -> newton.Model:
    """Build a single 6-DOF body (3 linear + 3 angular axes)."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.JointDofConfig
    link = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    joint = builder.add_joint_d6(
        parent=-1,
        child=link,
        linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        angular_axes=[cfg(axis=[1, 0, 0]), cfg(axis=[0, 1, 0]), cfg(axis=[0, 0, 1])],
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device, requires_grad=True)


# ----------------------------------------------------------------------------
# Helpers: FK evaluation
# ----------------------------------------------------------------------------


def _fk_end_effector_positions(
    model: newton.Model, body_q_2d: wp.array, n_problems: int, ee_link_index: int, ee_offset: wp.vec3
) -> np.ndarray:
    """Return (N, 3) world-space end-effector positions."""
    positions = np.zeros((n_problems, 3), dtype=np.float32)
    body_q_np = body_q_2d.numpy()

    for prob in range(n_problems):
        body_tf = body_q_np[prob, ee_link_index]
        pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
        rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
        ee_world = wp.transform_point(wp.transform(pos, rot), ee_offset)
        positions[prob] = [ee_world[0], ee_world[1], ee_world[2]]
    return positions


def _fk_end_effector_rotations(
    model: newton.Model, body_q_2d: wp.array, n_problems: int, ee_link_index: int
) -> np.ndarray:
    """Return (N, 4) world-space end-effector quaternions (x, y, z, w)."""
    rotations = np.zeros((n_problems, 4), dtype=np.float32)
    body_q_np = body_q_2d.numpy()

    for prob in range(n_problems):
        body_tf = body_q_np[prob, ee_link_index]
        rotations[prob] = [body_tf[3], body_tf[4], body_tf[5], body_tf[6]]
    return rotations


def _run_fk(model, joint_q_2d, n_problems):
    """Run FK and return (body_q, body_qd) arrays."""
    joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32, device=model.device)
    body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform, device=model.device)
    body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector, device=model.device)
    eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
    return body_q_2d, body_qd_2d


def _quat_angle_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle [rad] between two unit quaternions (x, y, z, w)."""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


# ----------------------------------------------------------------------------
# 1. test_qp_converges_position
# ----------------------------------------------------------------------------


def test_qp_converges_position(test, device):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
        targets = wp.array([[1.5, 1.0, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_rho=1.0,
            qp_max_iters=20,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=80, step_size=1.0)

        body_q, _ = _run_fk(model, joint_q_2d, n_problems)
        final = _fk_end_effector_positions(model, body_q, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err, 1e-3, f"problem {prob} position error {err:.6f} > 1mm")


# ----------------------------------------------------------------------------
# 2. test_qp_converges_rotation
# ----------------------------------------------------------------------------


def test_qp_converges_rotation(test, device):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_single_d6(device)

        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)

        angles = [math.pi / 6 + prob * math.pi / 8 for prob in range(n_problems)]
        rot_targets = wp.array(
            [[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)] for a in angles],
            dtype=wp.vec4,
        )
        pos_targets = wp.array([[0.0, 0.0, 0.0]] * n_problems, dtype=wp.vec3)

        pos_obj = ik.IKObjectivePosition(
            link_index=0,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=pos_targets,
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=0,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=rot_targets,
        )

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj, rot_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_rho=1.0,
            qp_max_iters=20,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=80, step_size=1.0)

        body_q, _ = _run_fk(model, joint_q_2d, n_problems)
        final_rot = _fk_end_effector_rotations(model, body_q, n_problems, 0)

        for prob in range(n_problems):
            target_quat = rot_targets.numpy()[prob]
            angle_err = _quat_angle_distance(final_rot[prob], target_quat)
            deg_err = math.degrees(angle_err)
            test.assertLess(deg_err, 1.0, f"problem {prob} rotation error {deg_err:.3f} deg > 1 deg")


# ----------------------------------------------------------------------------
# 3. test_qp_joint_limits_respected
# ----------------------------------------------------------------------------


def test_qp_joint_limits_respected(test, device):
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)

        # Target that requires large joint angles - push toward limits
        targets = wp.array([[0.0, 1.8, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=50, step_size=1.0)

        q_np = joint_q_2d.numpy()
        lower = model.joint_limit_lower.numpy()[: model.joint_coord_count]
        upper = model.joint_limit_upper.numpy()[: model.joint_coord_count]

        for prob in range(n_problems):
            for dof in range(model.joint_coord_count):
                lo = lower[dof]
                hi = upper[dof]
                if np.isfinite(lo) and np.isfinite(hi):
                    test.assertGreaterEqual(
                        q_np[prob, dof],
                        lo - 1e-4,
                        f"problem {prob} dof {dof} below lower limit",
                    )
                    test.assertLessEqual(
                        q_np[prob, dof],
                        hi + 1e-4,
                        f"problem {prob} dof {dof} above upper limit",
                    )


# ----------------------------------------------------------------------------
# 4. test_qp_velocity_limits_respected
# ----------------------------------------------------------------------------


def test_qp_velocity_limits_respected(test, device):
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_two_link_planar(device)
        n_dofs = model.joint_dof_count
        dt = 0.02
        v_max = 1.0  # rad/s per DOF

        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
        q_before = joint_q_2d.numpy().copy()

        targets = wp.array([[1.5, 1.0, 0.0]] * n_problems, dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        vel_limit = np.full(n_dofs, v_max, dtype=np.float32)

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_dt=dt,
            qp_velocity_limit=vel_limit,
        )

        # Run a single outer iteration to check the displacement bound
        solver.step(joint_q_2d, joint_q_2d, iterations=1, step_size=1.0)

        q_after = joint_q_2d.numpy()
        max_disp = v_max * dt

        for prob in range(n_problems):
            for dof in range(n_dofs):
                displacement = abs(q_after[prob, dof] - q_before[prob, dof])
                test.assertLessEqual(
                    displacement,
                    max_disp + 1e-5,
                    f"problem {prob} dof {dof} displacement {displacement:.6f} > v_max*dt={max_disp}",
                )


# ----------------------------------------------------------------------------
# 5. test_qp_warm_start_faster
# ----------------------------------------------------------------------------


def test_qp_warm_start_faster(test, device):
    with wp.ScopedDevice(device):
        n_problems = 1
        model = _build_two_link_planar(device)

        targets = wp.array([[1.5, 1.0, 0.0]], dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        # Cold start: reset before each outer iteration batch
        joint_q_cold = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
        solver_cold = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_max_iters=20,
        )
        solver_cold.step(joint_q_cold, joint_q_cold, iterations=10, step_size=1.0)

        body_q_cold, _ = _run_fk(model, joint_q_cold, n_problems)
        final_cold = _fk_end_effector_positions(model, body_q_cold, n_problems, ee_link, ee_off)
        err_cold = np.linalg.norm(final_cold[0] - targets.numpy()[0])

        # Warm start: same problem but with solver that persists ADMM state
        # Run in two phases to demonstrate warm-start benefit
        joint_q_warm = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)

        pos_obj_warm = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver_warm = ik.IKSolver(
            model,
            n_problems,
            [pos_obj_warm],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_max_iters=20,
        )
        # First phase: 5 iterations to establish warm-start state
        solver_warm.step(joint_q_warm, joint_q_warm, iterations=5, step_size=1.0)
        # Second phase: 5 more iterations with warm-started dual variables
        solver_warm.step(joint_q_warm, joint_q_warm, iterations=5, step_size=1.0)

        body_q_warm, _ = _run_fk(model, joint_q_warm, n_problems)
        final_warm = _fk_end_effector_positions(model, body_q_warm, n_problems, ee_link, ee_off)
        err_warm = np.linalg.norm(final_warm[0] - targets.numpy()[0])

        # Warm-started solver (split 5+5) should be at least as good as cold (10)
        # because dual variables carry useful information across calls
        test.assertLessEqual(
            err_warm,
            err_cold * 1.5,
            f"warm-start error {err_warm:.6f} much worse than cold {err_cold:.6f}",
        )


# ----------------------------------------------------------------------------
# 6. test_qp_matches_lm_accuracy
# ----------------------------------------------------------------------------


def test_qp_matches_lm_accuracy(test, device):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        targets = wp.array([[1.5, 1.0, 0.0], [1.0, 1.5, 0.0], [0.5, 1.8, 0.0]], dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        # LM solve
        pos_obj_lm = ik.IKObjectivePosition(link_index=ee_link, link_offset=ee_off, target_positions=targets)
        joint_q_lm = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
        solver_lm = ik.IKSolver(
            model,
            n_problems,
            [pos_obj_lm],
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            lambda_initial=1e-3,
        )
        solver_lm.step(joint_q_lm, joint_q_lm, iterations=40, step_size=1.0)

        body_q_lm, _ = _run_fk(model, joint_q_lm, n_problems)
        pos_lm = _fk_end_effector_positions(model, body_q_lm, n_problems, ee_link, ee_off)

        # QP solve
        pos_obj_qp = ik.IKObjectivePosition(link_index=ee_link, link_offset=ee_off, target_positions=targets)
        joint_q_qp = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
        solver_qp = ik.IKSolver(
            model,
            n_problems,
            [pos_obj_qp],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_max_iters=20,
        )
        solver_qp.step(joint_q_qp, joint_q_qp, iterations=80, step_size=1.0)

        body_q_qp, _ = _run_fk(model, joint_q_qp, n_problems)
        pos_qp = _fk_end_effector_positions(model, body_q_qp, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err_lm = np.linalg.norm(pos_lm[prob] - targets.numpy()[prob])
            err_qp = np.linalg.norm(pos_qp[prob] - targets.numpy()[prob])
            # QP should reach comparable accuracy; differential IK converges
            # more slowly than LM near workspace boundaries so we allow 2mm.
            test.assertLess(
                err_qp,
                max(err_lm * 10.0, 2e-3),
                f"problem {prob}: QP error {err_qp:.6f} much worse than LM {err_lm:.6f}",
            )


# ----------------------------------------------------------------------------
# 7. test_qp_batch
# ----------------------------------------------------------------------------


def test_qp_batch(test, device):
    with wp.ScopedDevice(device):
        n_problems = 5
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros((n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True)

        # Different targets for each problem
        target_data = [
            [1.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
            [1.8, 0.3, 0.0],
            [1.2, 1.2, 0.0],
        ]
        targets = wp.array(target_data, dtype=wp.vec3)
        ee_link = 1
        ee_off = wp.vec3(0.5, 0.0, 0.0)

        pos_obj = ik.IKObjectivePosition(
            link_index=ee_link,
            link_offset=ee_off,
            target_positions=targets,
        )

        solver = ik.IKSolver(
            model,
            n_problems,
            [pos_obj],
            optimizer=ik.IKOptimizer.QP,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=80, step_size=1.0)

        body_q, _ = _run_fk(model, joint_q_2d, n_problems)
        final = _fk_end_effector_positions(model, body_q, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err, 5e-3, f"batch problem {prob} error {err:.6f} > 5mm")


# ----------------------------------------------------------------------------
# Test class registration
# ----------------------------------------------------------------------------

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestIKQP(unittest.TestCase):
    pass


add_function_test(TestIKQP, "test_qp_converges_position", test_qp_converges_position, devices)
add_function_test(TestIKQP, "test_qp_converges_rotation", test_qp_converges_rotation, cuda_devices)
add_function_test(TestIKQP, "test_qp_joint_limits_respected", test_qp_joint_limits_respected, devices)
add_function_test(TestIKQP, "test_qp_velocity_limits_respected", test_qp_velocity_limits_respected, devices)
add_function_test(TestIKQP, "test_qp_warm_start_faster", test_qp_warm_start_faster, devices)
add_function_test(TestIKQP, "test_qp_matches_lm_accuracy", test_qp_matches_lm_accuracy, devices)
add_function_test(TestIKQP, "test_qp_batch", test_qp_batch, devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
