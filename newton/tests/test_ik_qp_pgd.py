# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the projected gradient descent QP inverse-kinematics optimizer."""

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


def _run_fk(model, joint_q_2d, n_problems):
    """Run FK and return (body_q, body_qd) arrays."""
    joint_qd_2d = wp.zeros((n_problems, model.joint_dof_count), dtype=wp.float32, device=model.device)
    body_q_2d = wp.zeros((n_problems, model.body_count), dtype=wp.transform, device=model.device)
    body_qd_2d = wp.zeros((n_problems, model.body_count), dtype=wp.spatial_vector, device=model.device)
    eval_fk_batched(model, joint_q_2d, joint_qd_2d, body_q_2d, body_qd_2d)
    return body_q_2d, body_qd_2d


# ----------------------------------------------------------------------------
# 1. test_qp_pgd_converges_position
# ----------------------------------------------------------------------------


def test_qp_pgd_converges_position(test, device):
    with wp.ScopedDevice(device):
        n_problems = 3
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros(
            (n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True
        )
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
            optimizer=ik.IKOptimizer.QP_PGD,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_pgd_max_iters=50,
        )

        # PGD is slower than ADMM so we give it more outer iterations
        solver.step(joint_q_2d, joint_q_2d, iterations=120, step_size=1.0)

        body_q, _ = _run_fk(model, joint_q_2d, n_problems)
        final = _fk_end_effector_positions(model, body_q, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err, 5e-3, f"problem {prob} position error {err:.6f} > 5mm")


# ----------------------------------------------------------------------------
# 2. test_qp_pgd_respects_joint_limits
# ----------------------------------------------------------------------------


def test_qp_pgd_respects_joint_limits(test, device):
    with wp.ScopedDevice(device):
        n_problems = 2
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros(
            (n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True
        )

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
            optimizer=ik.IKOptimizer.QP_PGD,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_pgd_max_iters=50,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=80, step_size=1.0)

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
# 3. test_qp_pgd_batch
# ----------------------------------------------------------------------------


def test_qp_pgd_batch(test, device):
    with wp.ScopedDevice(device):
        n_problems = 5
        model = _build_two_link_planar(device)

        joint_q_2d = wp.zeros(
            (n_problems, model.joint_coord_count), dtype=wp.float32, requires_grad=True
        )

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
            optimizer=ik.IKOptimizer.QP_PGD,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            qp_damping=1e-4,
            qp_pgd_max_iters=50,
        )

        solver.step(joint_q_2d, joint_q_2d, iterations=120, step_size=1.0)

        body_q, _ = _run_fk(model, joint_q_2d, n_problems)
        final = _fk_end_effector_positions(model, body_q, n_problems, ee_link, ee_off)

        for prob in range(n_problems):
            err = np.linalg.norm(final[prob] - targets.numpy()[prob])
            test.assertLess(err, 5e-3, f"batch problem {prob} error {err:.6f} > 5mm")


# ----------------------------------------------------------------------------
# Test class registration
# ----------------------------------------------------------------------------

devices = get_test_devices()


class TestIKQPPGD(unittest.TestCase):
    pass


add_function_test(TestIKQPPGD, "test_qp_pgd_converges_position", test_qp_pgd_converges_position, devices)
add_function_test(TestIKQPPGD, "test_qp_pgd_respects_joint_limits", test_qp_pgd_respects_joint_limits, devices)
add_function_test(TestIKQPPGD, "test_qp_pgd_batch", test_qp_pgd_batch, devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
