# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ADMM-based QP optimizer for differential inverse kinematics.

Solves velocity-space IK via box-constrained Quadratic Programming::

    min_{dq}  1/2 ||J dq - e||^2_W + 1/2 lambda ||dq||^2
    s.t.      lb <= dq <= ub

where ``lb`` and ``ub`` incorporate joint position limits and optional
velocity limits.  The ADMM inner loop is fused into a single tiled Warp
kernel with no CPU round-trips, and dual variables are warm-started across
outer IK iterations for rapid convergence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..model import Model
from .ik_common import IKJacobianType, compute_costs, eval_fk_batched, fk_accum
from .ik_objectives import IKObjective


@dataclass(slots=True)
class BatchCtx:
    """Per-step context shared between residual, Jacobian, and FK passes."""

    joint_q: wp.array2d[wp.float32]
    residuals: wp.array2d[wp.float32]
    fk_body_q: wp.array2d[wp.transform]
    problem_idx: wp.array[wp.int32]

    # AUTODIFF and MIXED
    fk_body_qd: wp.array2d[wp.spatial_vector] | None = None
    dq_dof: wp.array2d[wp.float32] | None = None
    joint_q_proposed: wp.array2d[wp.float32] | None = None
    joint_qd: wp.array2d[wp.float32] | None = None

    # ANALYTIC and MIXED
    jacobian_out: wp.array3d[wp.float32] | None = None
    motion_subspace: wp.array2d[wp.spatial_vector] | None = None
    fk_qd_zero: wp.array2d[wp.float32] | None = None
    fk_X_local: wp.array2d[wp.transform] | None = None


@wp.kernel
def _compute_box_bounds(
    joint_q: wp.array2d[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    n_dofs: int,
    has_vel_limit: int,
    vel_limit: wp.array[wp.float32],
    dt: float,
    lb_out: wp.array2d[wp.float32],
    ub_out: wp.array2d[wp.float32],
):
    """Compute per-DOF displacement bounds from joint and velocity limits.

    For each DOF *i*:
        ``lb[i] = max(q_lower[i] - q[i], -v_max[i] * dt)``
        ``ub[i] = min(q_upper[i] - q[i],  v_max[i] * dt)``

    Unbounded joints (infinite limits) pass through as-is.
    """
    row = wp.tid()
    for i in range(n_dofs):
        q_i = joint_q[row, i]
        lo = joint_limit_lower[i] - q_i
        hi = joint_limit_upper[i] - q_i

        if has_vel_limit == 1:
            v_max = vel_limit[i]
            vel_lb = -v_max * dt
            vel_ub = v_max * dt
            lo = wp.max(lo, vel_lb)
            hi = wp.min(hi, vel_ub)

        lb_out[row, i] = lo
        ub_out[row, i] = hi


class IKOptimizerQP:
    """ADMM-based QP optimizer for batched differential inverse kinematics.

    Solves for joint displacements ``dq`` that minimize tracking error
    subject to box constraints (joint position limits and optional velocity
    limits).  The full ADMM loop is fused into a single tiled Warp kernel
    so that all iterations execute on-device without CPU synchronization.
    Dual variables ``z`` and ``u`` are warm-started across outer IK
    iterations for fast convergence.

    Args:
        model: Shared articulation model.
        n_batch: Number of evaluation rows solved in parallel.
        objectives: Ordered IK objectives applied to every batch row.
        jacobian_mode: Jacobian backend to use.
        qp_max_iters: Maximum ADMM iterations per QP solve.
        qp_rho: Initial ADMM augmented-Lagrangian penalty parameter.
        qp_tol: ADMM convergence tolerance on the primal residual norm.
        damping: Regularization weight ``lambda`` for the ``||dq||^2``
            term.
        dt: Integration timestep [s] used for velocity-limit conversion.
        velocity_limit: Optional per-DOF velocity limits [rad/s or m/s],
            shape ``[joint_dof_count]``.
        problem_idx: Optional mapping from batch rows to base problem
            indices for per-problem objective data.
    """

    TILE_N_DOFS = None
    TILE_N_RESIDUALS = None
    _cache: ClassVar[dict[tuple[int, int, str], type]] = {}

    def __new__(
        cls,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        *a: Any,
        **kw: Any,
    ) -> IKOptimizerQP:
        n_dofs = model.joint_dof_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        arch = model.device.arch
        key = (n_dofs, n_residuals, arch)

        spec_cls = cls._cache.get(key)
        if spec_cls is None:
            spec_cls = cls._build_specialized(key)
            cls._cache[key] = spec_cls

        return super().__new__(spec_cls)

    def __init__(
        self,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        jacobian_mode: IKJacobianType = IKJacobianType.ANALYTIC,
        qp_max_iters: int = 20,
        qp_rho: float = 1.0,
        qp_tol: float = 1e-6,
        damping: float = 1e-4,
        dt: float = 0.01,
        velocity_limit: np.ndarray | None = None,
        *,
        problem_idx: wp.array[wp.int32] | None = None,
    ) -> None:
        self.model = model
        self.device = model.device
        self.n_batch = n_batch
        self.n_coords = model.joint_coord_count
        self.n_dofs = model.joint_dof_count
        self.n_residuals = sum(o.residual_dim() for o in objectives)

        self.objectives = objectives
        self.jacobian_mode = jacobian_mode
        self.has_analytic_objective = any(o.supports_analytic() for o in objectives)
        self.has_autodiff_objective = any(not o.supports_analytic() for o in objectives)

        self.qp_max_iters = qp_max_iters
        self.qp_rho = qp_rho
        self.qp_tol = qp_tol
        self.damping = damping
        self.dt = dt

        if self.TILE_N_DOFS is not None:
            assert self.n_dofs == self.TILE_N_DOFS
        if self.TILE_N_RESIDUALS is not None:
            assert self.n_residuals == self.TILE_N_RESIDUALS

        grad = jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED)
        self._alloc_solver_buffers(grad)
        self._alloc_admm_buffers(velocity_limit)
        self.problem_idx = problem_idx if problem_idx is not None else self.problem_idx_identity
        self.tape = wp.Tape() if grad else None

        self._build_residual_offsets()
        self._init_objectives()
        self._init_cuda_streams()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _alloc_solver_buffers(self, grad: bool) -> None:
        """Allocate FK, residual, Jacobian, and integration buffers."""
        device = self.device
        model = self.model

        self.qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.body_q = wp.zeros((self.n_batch, model.body_count), dtype=wp.transform, requires_grad=grad, device=device)
        self.body_qd = (
            wp.zeros((self.n_batch, model.body_count), dtype=wp.spatial_vector, device=device) if grad else None
        )

        self.residuals = wp.zeros((self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device)
        self.residuals_3d = wp.zeros((self.n_batch, self.n_residuals, 1), dtype=wp.float32, device=device)

        self.jacobian = wp.zeros((self.n_batch, self.n_residuals, self.n_dofs), dtype=wp.float32, device=device)
        self.dq_dof = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, requires_grad=grad, device=device)

        self.joint_q_proposed = wp.zeros(
            (self.n_batch, self.n_coords), dtype=wp.float32, requires_grad=grad, device=device
        )

        self.costs = wp.zeros(self.n_batch, dtype=wp.float32, device=device)

        self.problem_idx_identity = wp.array(np.arange(self.n_batch, dtype=np.int32), dtype=wp.int32, device=device)

        self.X_local = wp.zeros((self.n_batch, model.joint_count), dtype=wp.transform, device=device)
        self.joint_S_s = (
            wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=device)
            if self.jacobian_mode != IKJacobianType.AUTODIFF and self.has_analytic_objective
            else None
        )

    def _alloc_admm_buffers(self, velocity_limit: np.ndarray | None) -> None:
        """Allocate ADMM primal/dual variable and bound buffers."""
        device = self.device
        D = self.n_dofs
        B = self.n_batch

        self.admm_z = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_u = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_lb = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_ub = wp.zeros((B, D), dtype=wp.float32, device=device)

        self.has_vel_limit = velocity_limit is not None
        if velocity_limit is not None:
            self.vel_limit = wp.array(velocity_limit.astype(np.float32), dtype=wp.float32, device=device)
        else:
            self.vel_limit = wp.zeros(D, dtype=wp.float32, device=device)

    # ------------------------------------------------------------------
    # Shared infrastructure (mirrors LM optimizer)
    # ------------------------------------------------------------------

    def _build_residual_offsets(self) -> None:
        """Compute cumulative residual offsets for each objective."""
        offsets: list[int] = []
        offset = 0
        for obj in self.objectives:
            offsets.append(offset)
            offset += obj.residual_dim()
        self.residual_offsets = offsets

    def _init_objectives(self) -> None:
        """Allocate any per-objective buffers that must live on ``self.device``."""
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            obj.set_batch_layout(self.n_residuals, offset, self.n_batch)
            obj.bind_device(self.device)
            if self.jacobian_mode == IKJacobianType.MIXED:
                mode = IKJacobianType.ANALYTIC if obj.supports_analytic() else IKJacobianType.AUTODIFF
            else:
                mode = self.jacobian_mode
            obj.init_buffers(model=self.model, jacobian_mode=mode)

    def _init_cuda_streams(self) -> None:
        """Allocate per-objective Warp streams and sync events."""
        self.objective_streams: list[wp.Stream | None] = []
        self.sync_events: list[wp.Event | None] = []
        if self.device.is_cuda:
            for _ in range(len(self.objectives)):
                stream = wp.Stream(self.device)
                event = wp.Event(self.device)
                self.objective_streams.append(stream)
                self.sync_events.append(event)
        else:
            self.objective_streams = [None] * len(self.objectives)
            self.sync_events = [None] * len(self.objectives)

    def _parallel_for_objectives(self, fn: Callable[..., None], *extra: Any) -> None:
        """Run ``fn(obj, offset, *extra)`` across objectives on parallel CUDA streams."""
        if self.device.is_cuda:
            main = wp.get_stream(self.device)
            init_evt = main.record_event()
            for obj, offset, obj_stream, sync_event in zip(
                self.objectives, self.residual_offsets, self.objective_streams, self.sync_events, strict=False
            ):
                obj_stream.wait_event(init_evt)
                with wp.ScopedStream(obj_stream):
                    fn(obj, offset, *extra)
                obj_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
                fn(obj, offset, *extra)

    def _ctx_solver(
        self,
        joint_q: wp.array2d[wp.float32],
        *,
        residuals: wp.array2d[wp.float32] | None = None,
        jacobian: wp.array3d[wp.float32] | None = None,
    ) -> BatchCtx:
        """Build a :class:`BatchCtx` for the current joint configuration."""
        ctx = BatchCtx(
            joint_q=joint_q,
            residuals=residuals if residuals is not None else self.residuals,
            fk_body_q=self.body_q,
            problem_idx=self.problem_idx,
            fk_body_qd=getattr(self, "body_qd", None),
            dq_dof=self.dq_dof,
            joint_q_proposed=self.joint_q_proposed,
            joint_qd=self.qd_zero,
            jacobian_out=jacobian if jacobian is not None else self.jacobian,
            motion_subspace=getattr(self, "joint_S_s", None),
            fk_qd_zero=self.qd_zero,
            fk_X_local=self.X_local,
        )
        self._validate_ctx_for_mode(ctx)
        return ctx

    def _validate_ctx_for_mode(self, ctx: BatchCtx) -> None:
        """Assert that *ctx* has all arrays required by the active Jacobian mode."""
        missing: list[str] = []

        for name in ("joint_q", "residuals", "fk_body_q", "problem_idx"):
            if getattr(ctx, name) is None:
                missing.append(name)

        mode = self.jacobian_mode
        if mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            for name in ("fk_body_qd", "dq_dof", "joint_q_proposed", "joint_qd"):
                if getattr(ctx, name) is None:
                    missing.append(name)

        needs_analytic = mode == IKJacobianType.ANALYTIC or (
            mode == IKJacobianType.MIXED and self.has_analytic_objective
        )
        if needs_analytic:
            for name in ("jacobian_out", "motion_subspace", "fk_qd_zero"):
                if getattr(ctx, name) is None:
                    missing.append(name)
            if ctx.fk_X_local is None:
                missing.append("fk_X_local")

        if missing:
            raise RuntimeError(f"solver context missing: {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Residual and Jacobian computation
    # ------------------------------------------------------------------

    def _for_objectives_residuals(self, ctx: BatchCtx) -> None:
        """Evaluate all objective residuals into ``ctx.residuals``."""

        def _do(obj, offset, body_q_view, joint_q_view, model, output_residuals, problem_idx_array):
            obj.compute_residuals(
                body_q_view,
                joint_q_view,
                model,
                output_residuals,
                offset,
                problem_idx=problem_idx_array,
            )

        self._parallel_for_objectives(
            _do,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.residuals,
            ctx.problem_idx,
        )

    def _residuals_autodiff(self, ctx: BatchCtx) -> None:
        """Compute residuals using forward-kinematics autodiff path."""
        eval_fk_batched(self.model, ctx.joint_q, ctx.joint_qd, ctx.fk_body_q, ctx.fk_body_qd)
        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _residuals_analytic(self, ctx: BatchCtx) -> None:
        """Compute residuals using the two-pass analytic FK path."""
        self._fk_two_pass(self.model, ctx.joint_q, ctx.fk_body_q, ctx.fk_X_local, ctx.joint_q.shape[0])
        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _jacobian_at(self, ctx: BatchCtx) -> wp.array3d[wp.float32]:
        """Compute the Jacobian using the configured mode and return it."""
        mode = self.jacobian_mode

        if mode == IKJacobianType.AUTODIFF:
            self._jacobian_autodiff(ctx)
            return ctx.jacobian_out

        if mode == IKJacobianType.ANALYTIC:
            self._jacobian_analytic(ctx, accumulate=False)
            return ctx.jacobian_out

        # MIXED mode
        if self.has_autodiff_objective:
            self._jacobian_autodiff(ctx)
        else:
            ctx.jacobian_out.zero_()

        if self.has_analytic_objective:
            self._jacobian_analytic(ctx, accumulate=self.has_autodiff_objective)

        return ctx.jacobian_out

    def _jacobian_autodiff(self, ctx: BatchCtx) -> None:
        """Compute Jacobian columns for autodiff objectives via Warp tape."""
        if self.tape is None:
            raise RuntimeError("Autodiff Jacobian requested but tape is not initialized")

        ctx.jacobian_out.zero_()
        self.tape.reset()
        self.tape.gradients = {}
        ctx.dq_dof.zero_()

        with self.tape:
            self._integrate_dq(
                ctx.joint_q,
                dq_in=ctx.dq_dof,
                joint_q_out=ctx.joint_q_proposed,
                joint_qd_out=ctx.joint_qd,
            )

            res_ctx = BatchCtx(
                joint_q=ctx.joint_q_proposed,
                residuals=ctx.residuals,
                fk_body_q=ctx.fk_body_q,
                problem_idx=ctx.problem_idx,
                fk_body_qd=ctx.fk_body_qd,
                joint_qd=ctx.joint_qd,
            )
            self._residuals_autodiff(res_ctx)
            residuals_flat = ctx.residuals.flatten()

        self.tape.outputs = [residuals_flat]

        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if self.jacobian_mode == IKJacobianType.MIXED and obj.supports_analytic():
                continue
            obj.compute_jacobian_autodiff(self.tape, self.model, ctx.jacobian_out, offset, ctx.dq_dof)
            self.tape.zero()

    def _jacobian_analytic(self, ctx: BatchCtx, *, accumulate: bool) -> None:
        """Compute Jacobian columns for analytic objectives."""
        if not accumulate:
            ctx.jacobian_out.zero_()

        ctx.fk_qd_zero.zero_()
        self._compute_motion_subspace(
            body_q=ctx.fk_body_q,
            joint_S_s_out=ctx.motion_subspace,
            joint_qd_in=ctx.fk_qd_zero,
        )

        def _emit(obj, off, body_q_view, joint_q_view, model, jac_view, motion_subspace_view):
            if obj.supports_analytic():
                obj.compute_jacobian_analytic(body_q_view, joint_q_view, model, jac_view, motion_subspace_view, off)
            elif not accumulate:
                raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")

        self._parallel_for_objectives(
            _emit,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.jacobian_out,
            ctx.motion_subspace,
        )

    def _compute_residuals(
        self,
        joint_q: wp.array2d[wp.float32],
        output_residuals: wp.array2d[wp.float32] | None = None,
    ) -> wp.array2d[wp.float32]:
        """Evaluate residuals at *joint_q* using the active FK path."""
        buffer = output_residuals or self.residuals
        ctx = self._ctx_solver(joint_q, residuals=buffer)

        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx)
        else:
            self._residuals_analytic(ctx)

        return ctx.residuals

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        joint_q_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        iterations: int = 10,
        step_size: float = 1.0,
    ) -> None:
        """Run several differential-IK outer iterations via QP.

        Each outer iteration relinearizes (FK + Jacobian), builds a QP, and
        solves it with ADMM to get a displacement ``dq``.  ADMM dual
        variables are warm-started across iterations.

        Args:
            joint_q_in: Input joint coordinates [m or rad],
                shape ``[n_batch, joint_coord_count]``.
            joint_q_out: Output joint coordinates [m or rad],
                shape ``[n_batch, joint_coord_count]``. May alias
                *joint_q_in*.
            iterations: Number of outer IK iterations (re-linearizations).
            step_size: Scalar applied to each QP displacement before
                integration.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        joint_q = joint_q_out

        for _ in range(iterations):
            self._step(joint_q, step_size=step_size)

    def _step(self, joint_q: wp.array2d[wp.float32], step_size: float = 1.0) -> None:
        """One outer IK step: FK, Jacobian, QP (fused ADMM), integrate."""
        ctx = self._ctx_solver(joint_q)

        # 1. Compute residuals + Jacobian
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx)
        else:
            self._residuals_analytic(ctx)

        self._jacobian_at(ctx)

        # 2. Compute box-constraint bounds (position + velocity limits)
        wp.launch(
            _compute_box_bounds,
            dim=self.n_batch,
            inputs=[
                joint_q,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.n_dofs,
                1 if self.has_vel_limit else 0,
                self.vel_limit,
                self.dt,
            ],
            outputs=[self.admm_lb, self.admm_ub],
            device=self.device,
        )

        # 3. Reshape residuals for tiled kernel: (B, R) -> (B, R, 1)
        residuals_flat = ctx.residuals.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)

        # 4. Solve QP via fused ADMM kernel (warm-started from previous z, u)
        self.dq_dof.zero_()
        self._qp_solve_fused(
            ctx.jacobian_out,
            self.residuals_3d,
            self.admm_z,
            self.admm_u,
            self.admm_lb,
            self.admm_ub,
            self.dq_dof,
        )

        # 5. Integrate: q_new = q + step_size * z (projected feasible solution)
        wp.copy(self.dq_dof, self.admm_z)
        self._integrate_dq(
            joint_q,
            dq_in=self.dq_dof,
            joint_q_out=self.joint_q_proposed,
            joint_qd_out=self.qd_zero,
            step_size=step_size,
        )
        wp.copy(joint_q, self.joint_q_proposed)

    def reset(self) -> None:
        """Clear ADMM dual state before a new solve sequence."""
        self.admm_z.zero_()
        self.admm_u.zero_()

    def compute_costs(self, joint_q: wp.array2d[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate squared residual costs for a batch of joint configurations.

        Args:
            joint_q: Joint coordinates [m or rad] to evaluate,
                shape ``[n_batch, joint_coord_count]``.

        Returns:
            Per-row cost, shape ``[n_batch]``.
        """
        self._compute_residuals(joint_q)
        wp.launch(
            compute_costs,
            dim=self.n_batch,
            inputs=[self.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )
        return self.costs

    # ------------------------------------------------------------------
    # Fused ADMM tiled kernel (overridden by specialized subclass)
    # ------------------------------------------------------------------

    def _qp_solve_fused(self, jacobian, residuals, z, u, lb, ub, dq_out):
        raise NotImplementedError("Overridden by specialized subclass")

    @classmethod
    def _build_specialized(cls, key: tuple[int, int, str]) -> type[IKOptimizerQP]:
        """Build a specialized subclass with a fused tiled ADMM kernel.

        The kernel runs the full ADMM loop (solve, z-update, u-update,
        adaptive rho, convergence check) in a single Warp ``launch_tiled``
        with no CPU round-trips.  It is specialized per
        ``(n_dofs, n_residuals)`` so that tile dimensions are compile-time
        constants.
        """
        C, R, _ = key

        def _admm_fused_template(
            jacobians: wp.array3d[wp.float32],
            residuals: wp.array3d[wp.float32],
            z_io: wp.array2d[wp.float32],
            u_io: wp.array2d[wp.float32],
            lb: wp.array2d[wp.float32],
            ub: wp.array2d[wp.float32],
            rho_init: float,
            damping: float,
            tol: float,
            max_iters: int,
            dq_out: wp.array2d[wp.float32],
        ):
            row = wp.tid()

            RES = _Specialized.TILE_N_RESIDUALS
            DOF = _Specialized.TILE_N_DOFS

            # Load Jacobian and residual tiles (constant across ADMM iters)
            J = wp.tile_load(jacobians[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES, 1))

            # Precompute J^T and J^T J (constant across ADMM iters)
            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            # Precompute -J^T e (gradient of the quadratic, constant)
            neg_Jtr_2d = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, neg_Jtr_2d)
            neg_Jtr = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                neg_Jtr[i] = -neg_Jtr_2d[i, 0]

            # Warm-start z and u from previous outer iteration
            z_prev = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            u_prev = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                z_prev[i] = z_io[row, i]
                u_prev[i] = u_io[row, i]

            # Load box bounds
            lb_vec = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            ub_vec = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                lb_vec[i] = lb[row, i]
                ub_vec[i] = ub[row, i]

            rho = rho_init
            z_cur = z_prev
            u_cur = u_prev

            for _k in range(max_iters):
                # ----- dq update: solve (J^T J + (damping + rho) I) dq = -J^T e + rho (z - u) -----
                diag_val = damping + rho
                diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
                for i in range(DOF):
                    diag[i] = diag_val
                A = wp.tile_diag_add(JtJ, diag)

                rhs = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
                for i in range(DOF):
                    rhs[i] = neg_Jtr[i] + rho * (z_cur[i] - u_cur[i])

                L = wp.tile_cholesky(A)
                dq = wp.tile_cholesky_solve(L, rhs)

                # ----- z update: project (dq + u) onto [lb, ub] -----
                z_new = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
                for i in range(DOF):
                    val = dq[i] + u_cur[i]
                    val = wp.max(val, lb_vec[i])
                    val = wp.min(val, ub_vec[i])
                    z_new[i] = val

                # ----- u update: u += dq - z -----
                u_new = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
                for i in range(DOF):
                    u_new[i] = u_cur[i] + dq[i] - z_new[i]

                # ----- Adaptive rho (Boyd et al. 2011, Sec. 3.4.1) -----
                primal_norm_sq = float(0.0)
                dual_norm_sq = float(0.0)
                for i in range(DOF):
                    pri = dq[i] - z_new[i]
                    dua = rho * (z_new[i] - z_cur[i])
                    primal_norm_sq += pri * pri
                    dual_norm_sq += dua * dua

                primal_norm = wp.sqrt(primal_norm_sq)
                dual_norm = wp.sqrt(dual_norm_sq)

                mu = float(10.0)
                tau = float(2.0)
                if primal_norm > mu * dual_norm:
                    rho = rho * tau
                elif dual_norm > mu * primal_norm:
                    rho = rho / tau

                z_cur = z_new
                u_cur = u_new

                # ----- Convergence check -----
                if primal_norm < tol:
                    break

            # Write back warm-start state and solution
            wp.tile_store(dq_out[row], dq)
            for i in range(DOF):
                z_io[row, i] = z_cur[i]
                u_io[row, i] = u_cur[i]

        _admm_fused_template.__name__ = f"_qp_admm_fused_{C}_{R}"
        _admm_fused_template.__qualname__ = f"_qp_admm_fused_{C}_{R}"
        _qp_admm_fused_kernel = wp.kernel(enable_backward=False, module="unique")(_admm_fused_template)

        # Import FK integration kernels
        from ...solvers.featherstone.kernels import (  # noqa: PLC0415
            jcalc_integrate,
            jcalc_motion,
            jcalc_transform,
        )

        @wp.kernel
        def _integrate_dq_dof(
            joint_type: wp.array[wp.int32],
            joint_q_start: wp.array[wp.int32],
            joint_qd_start: wp.array[wp.int32],
            joint_dof_dim: wp.array2d[wp.int32],
            joint_q_curr: wp.array2d[wp.float32],
            joint_qd_curr: wp.array2d[wp.float32],
            dq_dof: wp.array2d[wp.float32],
            dt: float,
            joint_q_out: wp.array2d[wp.float32],
            joint_qd_out: wp.array2d[wp.float32],
        ):
            row, joint_idx = wp.tid()
            t = joint_type[joint_idx]
            coord_start = joint_q_start[joint_idx]
            dof_start = joint_qd_start[joint_idx]
            lin_axes = joint_dof_dim[joint_idx, 0]
            ang_axes = joint_dof_dim[joint_idx, 1]
            jcalc_integrate(
                t,
                joint_q_curr[row],
                joint_qd_curr[row],
                dq_dof[row],
                coord_start,
                dof_start,
                lin_axes,
                ang_axes,
                dt,
                joint_q_out[row],
                joint_qd_out[row],
            )

        @wp.kernel(module="unique")
        def _compute_motion_subspace_2d(
            joint_type: wp.array[wp.int32],
            joint_parent: wp.array[wp.int32],
            joint_qd_start: wp.array[wp.int32],
            joint_qd: wp.array2d[wp.float32],
            joint_axis: wp.array[wp.vec3],
            joint_dof_dim: wp.array2d[wp.int32],
            body_q: wp.array2d[wp.transform],
            joint_X_p: wp.array[wp.transform],
            joint_S_s: wp.array2d[wp.spatial_vector],
        ):
            row, joint_idx = wp.tid()
            type = joint_type[joint_idx]
            parent = joint_parent[joint_idx]
            qd_start = joint_qd_start[joint_idx]
            X_pj = joint_X_p[joint_idx]
            X_wpj = X_pj
            if parent >= 0:
                X_wpj = body_q[row, parent] * X_pj
            lin_axis_count = joint_dof_dim[joint_idx, 0]
            ang_axis_count = joint_dof_dim[joint_idx, 1]
            jcalc_motion(
                type,
                joint_axis,
                lin_axis_count,
                ang_axis_count,
                X_wpj,
                joint_qd[row],
                qd_start,
                joint_S_s[row],
            )

        @wp.kernel(module="unique")
        def _fk_local(
            joint_type: wp.array[wp.int32],
            joint_q: wp.array2d[wp.float32],
            joint_q_start: wp.array[wp.int32],
            joint_qd_start: wp.array[wp.int32],
            joint_axis: wp.array[wp.vec3],
            joint_dof_dim: wp.array2d[wp.int32],
            joint_X_p: wp.array[wp.transform],
            joint_X_c: wp.array[wp.transform],
            X_local_out: wp.array2d[wp.transform],
        ):
            row, local_joint_idx = wp.tid()
            t = joint_type[local_joint_idx]
            q_start = joint_q_start[local_joint_idx]
            axis_start = joint_qd_start[local_joint_idx]
            lin_axes = joint_dof_dim[local_joint_idx, 0]
            ang_axes = joint_dof_dim[local_joint_idx, 1]
            X_j = jcalc_transform(
                t,
                joint_axis,
                axis_start,
                lin_axes,
                ang_axes,
                joint_q[row],
                q_start,
            )
            X_rel = joint_X_p[local_joint_idx] * X_j * wp.transform_inverse(joint_X_c[local_joint_idx])
            X_local_out[row, local_joint_idx] = X_rel

        def _fk_two_pass(model, joint_q, body_q, X_local, n_batch):
            """Compute forward kinematics using the two-pass algorithm.

            Args:
                model: Articulation model.
                joint_q: Joint coordinates, shape ``[n_batch, joint_coord_count]``.
                body_q: Output body transforms, shape ``[n_batch, body_count]``.
                X_local: Workspace, shape ``[n_batch, joint_count]``.
                n_batch: Number of rows to process.
            """
            wp.launch(
                _fk_local,
                dim=[n_batch, model.joint_count],
                inputs=[
                    model.joint_type,
                    joint_q,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.joint_X_p,
                    model.joint_X_c,
                ],
                outputs=[X_local],
                device=model.device,
            )
            wp.launch(
                fk_accum,
                dim=[n_batch, model.joint_count],
                inputs=[model.joint_parent, X_local],
                outputs=[body_q],
                device=model.device,
            )

        class _Specialized(IKOptimizerQP):
            TILE_N_DOFS = wp.constant(C)
            TILE_N_RESIDUALS = wp.constant(R)
            TILE_THREADS = wp.constant(32)

            def _qp_solve_fused(self, jac, res, z, u, lb, ub, dq):
                wp.launch_tiled(
                    _qp_admm_fused_kernel,
                    dim=[self.n_batch],
                    inputs=[jac, res, z, u, lb, ub, self.qp_rho, self.damping, self.qp_tol, self.qp_max_iters, dq],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialized.__name__ = f"IKQP_{C}x{R}"
        _Specialized._integrate_dq_dof = staticmethod(_integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(_fk_two_pass)
        return _Specialized

    def _integrate_dq(
        self,
        joint_q: wp.array2d[wp.float32],
        *,
        dq_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        joint_qd_out: wp.array2d[wp.float32],
        step_size: float = 1.0,
    ) -> None:
        """Integrate ``dq_in`` into *joint_q* to produce *joint_q_out*."""
        batch = joint_q.shape[0]
        wp.launch(
            self._integrate_dq_dof,
            dim=[batch, self.model.joint_count],
            inputs=[
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                joint_q,
                dq_in,
                self.qd_zero,
                step_size,
            ],
            outputs=[joint_q_out, joint_qd_out],
            device=self.device,
        )
        joint_qd_out.zero_()

    def _compute_motion_subspace(
        self,
        *,
        body_q: wp.array2d[wp.transform],
        joint_S_s_out: wp.array2d[wp.spatial_vector],
        joint_qd_in: wp.array2d[wp.float32],
    ) -> None:
        """Compute per-DOF motion subspace vectors in world frame."""
        n_joints = self.model.joint_count
        batch = body_q.shape[0]
        wp.launch(
            self._compute_motion_subspace_2d,
            dim=[batch, n_joints],
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_qd_start,
                joint_qd_in,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                body_q,
                self.model.joint_X_p,
            ],
            outputs=[joint_S_s_out],
            device=self.device,
        )
