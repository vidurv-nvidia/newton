# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ADMM-based QP optimizer for differential inverse kinematics.

Solves velocity-space IK via Quadratic Programming::

    min_Δq  ½ ||J Δq - e||²_W + ½ λ ||Δq||²
    s.t.    lb ≤ Δq ≤ ub

where ``lb`` and ``ub`` incorporate joint position limits and optional
velocity limits.  Uses the Alternating Direction Method of Multipliers
(ADMM), implemented entirely as Warp kernels for GPU-native execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..model import Model
from .ik_common import IKJacobianType, compute_costs, eval_fk_batched, fk_accum
from .ik_objectives import IKObjective


@dataclass(slots=True)
class BatchCtx:
    joint_q: wp.array2d[wp.float32]
    residuals: wp.array2d[wp.float32]
    fk_body_q: wp.array2d[wp.transform]
    problem_idx: wp.array[wp.int32]

    fk_body_qd: wp.array2d[wp.spatial_vector] | None = None
    dq_dof: wp.array2d[wp.float32] | None = None
    joint_q_proposed: wp.array2d[wp.float32] | None = None
    joint_qd: wp.array2d[wp.float32] | None = None

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
    """Compute per-DOF displacement bounds from joint + velocity limits."""
    row = wp.tid()
    for i in range(n_dofs):
        q_i = joint_q[row, i]
        lo = joint_limit_lower[i]
        hi = joint_limit_upper[i]
        pos_lb = lo - q_i
        pos_ub = hi - q_i

        if has_vel_limit == 1:
            v_max = vel_limit[i]
            vel_lb = -v_max * dt
            vel_ub = v_max * dt
            pos_lb = wp.max(pos_lb, vel_lb)
            pos_ub = wp.min(pos_ub, vel_ub)

        lb_out[row, i] = pos_lb
        ub_out[row, i] = pos_ub


@wp.kernel
def _admm_z_update_box(
    dq: wp.array2d[wp.float32],
    u: wp.array2d[wp.float32],
    lb: wp.array2d[wp.float32],
    ub: wp.array2d[wp.float32],
    n_dofs: int,
    z_out: wp.array2d[wp.float32],
):
    """ADMM z-update: project (Δq + u) onto box constraints."""
    row = wp.tid()
    for i in range(n_dofs):
        val = dq[row, i] + u[row, i]
        val = wp.max(val, lb[row, i])
        val = wp.min(val, ub[row, i])
        z_out[row, i] = val


@wp.kernel
def _admm_u_update(
    dq: wp.array2d[wp.float32],
    z: wp.array2d[wp.float32],
    n_dofs: int,
    u: wp.array2d[wp.float32],
):
    """ADMM dual variable update: u += Δq - z."""
    row = wp.tid()
    for i in range(n_dofs):
        u[row, i] = u[row, i] + dq[row, i] - z[row, i]


@wp.kernel
def _admm_primal_residual(
    dq: wp.array2d[wp.float32],
    z: wp.array2d[wp.float32],
    n_dofs: int,
    residual_out: wp.array[wp.float32],
):
    """Compute ADMM primal residual: max_i |Δq_i - z_i|."""
    row = wp.tid()
    max_r = float(0.0)
    for i in range(n_dofs):
        r = wp.abs(dq[row, i] - z[row, i])
        max_r = wp.max(max_r, r)
    residual_out[row] = max_r


class IKOptimizerQP:
    """ADMM-based QP optimizer for batched differential inverse kinematics.

    Solves for joint displacements ``Δq`` that minimize tracking error
    subject to box constraints (joint position limits and optional velocity
    limits).  The ADMM algorithm runs entirely as Warp kernels, supporting
    both CPU and GPU execution.

    Args:
        model: Shared articulation model.
        n_batch: Number of evaluation rows solved in parallel.
        objectives: Ordered IK objectives applied to every batch row.
        jacobian_mode: Jacobian backend to use.
        qp_max_iters: Maximum ADMM iterations per QP solve.
        qp_rho: ADMM augmented Lagrangian penalty parameter.
        qp_tol: ADMM convergence tolerance (primal residual).
        damping: Regularization weight ``λ`` for ``||Δq||²`` term.
        dt: Integration timestep [s] used for velocity limit conversion.
        velocity_limit: Optional per-DOF velocity limits [rad/s or m/s].
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
        device = self.device
        model = self.model

        self.qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.body_q = wp.zeros(
            (self.n_batch, model.body_count), dtype=wp.transform, requires_grad=grad, device=device
        )
        self.body_qd = (
            wp.zeros((self.n_batch, model.body_count), dtype=wp.spatial_vector, device=device) if grad else None
        )

        self.residuals = wp.zeros(
            (self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device
        )
        self.residuals_3d = wp.zeros((self.n_batch, self.n_residuals, 1), dtype=wp.float32, device=device)

        self.jacobian = wp.zeros(
            (self.n_batch, self.n_residuals, self.n_dofs), dtype=wp.float32, device=device
        )
        self.dq_dof = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, requires_grad=grad, device=device)

        self.joint_q_proposed = wp.zeros(
            (self.n_batch, self.n_coords), dtype=wp.float32, requires_grad=grad, device=device
        )

        self.costs = wp.zeros(self.n_batch, dtype=wp.float32, device=device)

        self.problem_idx_identity = wp.array(
            np.arange(self.n_batch, dtype=np.int32), dtype=wp.int32, device=device
        )

        self.X_local = wp.zeros((self.n_batch, model.joint_count), dtype=wp.transform, device=device)
        self.joint_S_s = (
            wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=device)
            if self.jacobian_mode != IKJacobianType.AUTODIFF and self.has_analytic_objective
            else None
        )

    def _alloc_admm_buffers(self, velocity_limit: np.ndarray | None) -> None:
        device = self.device
        D = self.n_dofs
        B = self.n_batch

        self.admm_z = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_u = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_lb = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_ub = wp.zeros((B, D), dtype=wp.float32, device=device)
        self.admm_primal_res = wp.zeros(B, dtype=wp.float32, device=device)

        self.has_vel_limit = velocity_limit is not None
        if velocity_limit is not None:
            self.vel_limit = wp.array(
                velocity_limit.astype(np.float32), dtype=wp.float32, device=device
            )
        else:
            self.vel_limit = wp.zeros(D, dtype=wp.float32, device=device)

    # ------------------------------------------------------------------
    # Shared infrastructure (same as LM)
    # ------------------------------------------------------------------

    def _build_residual_offsets(self) -> None:
        offsets: list[int] = []
        offset = 0
        for obj in self.objectives:
            offsets.append(offset)
            offset += obj.residual_dim()
        self.residual_offsets = offsets

    def _init_objectives(self) -> None:
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            obj.set_batch_layout(self.n_residuals, offset, self.n_batch)
            obj.bind_device(self.device)
            if self.jacobian_mode == IKJacobianType.MIXED:
                mode = IKJacobianType.ANALYTIC if obj.supports_analytic() else IKJacobianType.AUTODIFF
            else:
                mode = self.jacobian_mode
            obj.init_buffers(model=self.model, jacobian_mode=mode)

    def _init_cuda_streams(self) -> None:
        self.objective_streams = []
        self.sync_events = []
        if self.device.is_cuda:
            for _ in range(len(self.objectives)):
                stream = wp.Stream(self.device)
                event = wp.Event(self.device)
                self.objective_streams.append(stream)
                self.sync_events.append(event)
        else:
            self.objective_streams = [None] * len(self.objectives)
            self.sync_events = [None] * len(self.objectives)

    def _parallel_for_objectives(self, fn, *extra):
        from collections.abc import Callable  # noqa: PLC0415

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

    def _ctx_solver(self, joint_q, *, residuals=None, jacobian=None):
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
        return ctx

    def _for_objectives_residuals(self, ctx):
        def _do(obj, offset, body_q_view, joint_q_view, model, output_residuals, problem_idx_array):
            obj.compute_residuals(
                body_q_view, joint_q_view, model, output_residuals, offset,
                problem_idx=problem_idx_array,
            )
        self._parallel_for_objectives(
            _do, ctx.fk_body_q, ctx.joint_q, self.model, ctx.residuals, ctx.problem_idx,
        )

    def _residuals_autodiff(self, ctx):
        eval_fk_batched(self.model, ctx.joint_q, ctx.joint_qd, ctx.fk_body_q, ctx.fk_body_qd)
        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _residuals_analytic(self, ctx):
        self._fk_two_pass(self.model, ctx.joint_q, ctx.fk_body_q, ctx.fk_X_local, ctx.joint_q.shape[0])
        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _jacobian_at(self, ctx):
        mode = self.jacobian_mode
        if mode == IKJacobianType.AUTODIFF:
            self._jacobian_autodiff(ctx)
            return ctx.jacobian_out
        if mode == IKJacobianType.ANALYTIC:
            self._jacobian_analytic(ctx, accumulate=False)
            return ctx.jacobian_out
        if self.has_autodiff_objective:
            self._jacobian_autodiff(ctx)
        else:
            ctx.jacobian_out.zero_()
        if self.has_analytic_objective:
            self._jacobian_analytic(ctx, accumulate=self.has_autodiff_objective)
        return ctx.jacobian_out

    def _jacobian_autodiff(self, ctx):
        if self.tape is None:
            raise RuntimeError("Autodiff Jacobian requested but tape is not initialized")
        ctx.jacobian_out.zero_()
        self.tape.reset()
        self.tape.gradients = {}
        ctx.dq_dof.zero_()
        with self.tape:
            self._integrate_dq(ctx.joint_q, dq_in=ctx.dq_dof,
                               joint_q_out=ctx.joint_q_proposed, joint_qd_out=ctx.joint_qd)
            res_ctx = BatchCtx(joint_q=ctx.joint_q_proposed, residuals=ctx.residuals,
                               fk_body_q=ctx.fk_body_q, problem_idx=ctx.problem_idx,
                               fk_body_qd=ctx.fk_body_qd, joint_qd=ctx.joint_qd)
            self._residuals_autodiff(res_ctx)
            residuals_flat = ctx.residuals.flatten()
        self.tape.outputs = [residuals_flat]
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if self.jacobian_mode == IKJacobianType.MIXED and obj.supports_analytic():
                continue
            obj.compute_jacobian_autodiff(self.tape, self.model, ctx.jacobian_out, offset, ctx.dq_dof)
            self.tape.zero()

    def _jacobian_analytic(self, ctx, *, accumulate):
        if not accumulate:
            ctx.jacobian_out.zero_()
        ctx.fk_qd_zero.zero_()
        self._compute_motion_subspace(
            body_q=ctx.fk_body_q, joint_S_s_out=ctx.motion_subspace, joint_qd_in=ctx.fk_qd_zero,
        )
        def _emit(obj, off, body_q_view, joint_q_view, model, jac_view, ms_view):
            if obj.supports_analytic():
                obj.compute_jacobian_analytic(body_q_view, joint_q_view, model, jac_view, ms_view, off)
            elif not accumulate:
                raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")
        self._parallel_for_objectives(_emit, ctx.fk_body_q, ctx.joint_q, self.model,
                                       ctx.jacobian_out, ctx.motion_subspace)

    def _compute_residuals(self, joint_q, output_residuals=None):
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
        iterations: int = 1,
        step_size: float = 1.0,
        tol: float | None = None,
        check_every: int = 5,
    ) -> int:
        """Run differential IK steps via QP.

        Each outer iteration relinearizes (FK + Jacobian), builds a QP, and
        solves it with ADMM to get a displacement ``Δq``.

        Args:
            joint_q_in: Input joint coordinates [m or rad],
                shape ``[n_batch, joint_coord_count]``.
            joint_q_out: Output joint coordinates [m or rad],
                shape ``[n_batch, joint_coord_count]``. May alias *joint_q_in*.
            iterations: Number of outer IK iterations (re-linearizations).
            step_size: Scalar applied to each QP displacement before
                integration.
            tol: Optional cost tolerance for early termination.
            check_every: How often to evaluate *tol*.

        Returns:
            Number of outer iterations actually executed.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        joint_q = joint_q_out
        iters_used = iterations

        for i in range(iterations):
            self._step(joint_q, step_size=step_size)
            if tol is not None and (i + 1) % check_every == 0:
                self.compute_costs(joint_q)
                if float(np.max(self.costs.numpy())) < tol:
                    iters_used = i + 1
                    break
        return iters_used

    def _step(self, joint_q: wp.array2d[wp.float32], step_size: float = 1.0) -> None:
        """One outer IK step: FK → Jacobian → QP (ADMM) → integrate."""

        ctx = self._ctx_solver(joint_q)

        # 1. Compute residuals + Jacobian
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx)
        else:
            self._residuals_analytic(ctx)

        self._jacobian_at(ctx)

        # 2. Compute constraint bounds
        lower = self.model.joint_limit_lower
        upper = self.model.joint_limit_upper
        wp.launch(
            _compute_box_bounds,
            dim=self.n_batch,
            inputs=[joint_q, lower, upper, self.n_dofs,
                    1 if self.has_vel_limit else 0, self.vel_limit, self.dt],
            outputs=[self.admm_lb, self.admm_ub],
            device=self.device,
        )

        # 3. Reshape residuals for tiled kernel
        residuals_flat = ctx.residuals.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)

        # 4. Solve QP via ADMM
        self.admm_z.zero_()
        self.admm_u.zero_()
        self.dq_dof.zero_()

        for k in range(self.qp_max_iters):
            # Δq-update: solve (H + ρI) Δq = -(J^T W e) + ρ(z - u)
            self._qp_solve_tiled(
                ctx.jacobian_out, self.residuals_3d,
                self.admm_z, self.admm_u,
                self.dq_dof,
            )

            # z-update: project onto box constraints
            wp.launch(
                _admm_z_update_box,
                dim=self.n_batch,
                inputs=[self.dq_dof, self.admm_u, self.admm_lb, self.admm_ub, self.n_dofs],
                outputs=[self.admm_z],
                device=self.device,
            )

            # u-update: dual variable
            wp.launch(
                _admm_u_update,
                dim=self.n_batch,
                inputs=[self.dq_dof, self.admm_z, self.n_dofs],
                outputs=[self.admm_u],
                device=self.device,
            )

            # Check convergence
            wp.launch(
                _admm_primal_residual,
                dim=self.n_batch,
                inputs=[self.dq_dof, self.admm_z, self.n_dofs],
                outputs=[self.admm_primal_res],
                device=self.device,
            )

            if (k + 1) % 5 == 0:
                max_res = float(np.max(self.admm_primal_res.numpy()))
                if max_res < self.qp_tol:
                    break

        # 5. Integrate: q_new = q + step_size * z (use z, the projected solution)
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
        """Clear ADMM state before a new solve."""
        self.admm_z.zero_()
        self.admm_u.zero_()
        self.admm_primal_res.zero_()

    def compute_costs(self, joint_q: wp.array2d[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate squared residual costs for a batch of joint configurations.

        Args:
            joint_q: Joint coordinates to evaluate,
                shape ``[n_batch, joint_coord_count]``.

        Returns:
            Costs for each batch row, shape ``[n_batch]``.
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
    # Tiled kernel (overridden by specialized subclass)
    # ------------------------------------------------------------------

    def _qp_solve_tiled(self, jacobian, residuals, z, u, dq_out):
        raise NotImplementedError("This method should be overridden by specialized solver")

    @classmethod
    def _build_specialized(cls, key: tuple[int, int, str]) -> type[IKOptimizerQP]:
        """Build a specialized subclass with tiled ADMM Δq-update kernel."""
        C, R, _ = key

        def _template(
            jacobians: wp.array3d[wp.float32],   # (n_batch, n_residuals, n_dofs)
            residuals: wp.array3d[wp.float32],   # (n_batch, n_residuals, 1)
            z: wp.array2d[wp.float32],           # (n_batch, n_dofs)
            u: wp.array2d[wp.float32],           # (n_batch, n_dofs)
            rho: float,
            damping: float,
            # outputs
            dq_out: wp.array2d[wp.float32],      # (n_batch, n_dofs)
        ):
            row = wp.tid()

            RES = _Specialized.TILE_N_RESIDUALS
            DOF = _Specialized.TILE_N_DOFS

            J = wp.tile_load(jacobians[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES, 1))

            # Build H = J^T J + (damping + rho) * I
            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            diag_val = damping + rho
            diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                diag[i] = diag_val
            A = wp.tile_diag_add(JtJ, diag)

            # Build rhs = -J^T e + rho * (z - u)
            Jtr_2d = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, Jtr_2d)

            rhs = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                zu = z[row, i] - u[row, i]
                rhs[i] = -Jtr_2d[i, 0] + rho * zu

            # Solve via Cholesky
            L = wp.tile_cholesky(A)
            delta = wp.tile_cholesky_solve(L, rhs)
            wp.tile_store(dq_out[row], delta)

        _template.__name__ = f"_qp_admm_solve_tiled_{C}_{R}"
        _template.__qualname__ = f"_qp_admm_solve_tiled_{C}_{R}"
        _qp_solve_kernel = wp.kernel(enable_backward=False, module="unique")(_template)

        # Import FK kernels (same as LM)
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
                t, joint_q_curr[row], joint_qd_curr[row], dq_dof[row],
                coord_start, dof_start, lin_axes, ang_axes, dt,
                joint_q_out[row], joint_qd_out[row],
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
                type, joint_axis, lin_axis_count, ang_axis_count,
                X_wpj, joint_qd[row], qd_start, joint_S_s[row],
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
                t, joint_axis, axis_start, lin_axes, ang_axes,
                joint_q[row], q_start,
            )
            X_rel = joint_X_p[local_joint_idx] * X_j * wp.transform_inverse(joint_X_c[local_joint_idx])
            X_local_out[row, local_joint_idx] = X_rel

        def _fk_two_pass(model, joint_q, body_q, X_local, n_batch):
            wp.launch(
                _fk_local,
                dim=[n_batch, model.joint_count],
                inputs=[model.joint_type, joint_q, model.joint_q_start, model.joint_qd_start,
                        model.joint_axis, model.joint_dof_dim, model.joint_X_p, model.joint_X_c],
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

            def _qp_solve_tiled(self, jac, res, z, u, dq):
                wp.launch_tiled(
                    _qp_solve_kernel,
                    dim=[self.n_batch],
                    inputs=[jac, res, z, u, self.qp_rho, self.damping, dq],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialized.__name__ = f"IKQP_{C}x{R}"
        _Specialized._integrate_dq_dof = staticmethod(_integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(_fk_two_pass)
        return _Specialized

    def _integrate_dq(self, joint_q, *, dq_in, joint_q_out, joint_qd_out, step_size=1.0):
        batch = joint_q.shape[0]
        wp.launch(
            self._integrate_dq_dof,
            dim=[batch, self.model.joint_count],
            inputs=[
                self.model.joint_type, self.model.joint_q_start,
                self.model.joint_qd_start, self.model.joint_dof_dim,
                joint_q, dq_in, self.qd_zero, step_size,
            ],
            outputs=[joint_q_out, joint_qd_out],
            device=self.device,
        )
        joint_qd_out.zero_()

    def _compute_motion_subspace(self, *, body_q, joint_S_s_out, joint_qd_in):
        n_joints = self.model.joint_count
        batch = body_q.shape[0]
        wp.launch(
            self._compute_motion_subspace_2d,
            dim=[batch, n_joints],
            inputs=[
                self.model.joint_type, self.model.joint_parent,
                self.model.joint_qd_start, joint_qd_in,
                self.model.joint_axis, self.model.joint_dof_dim,
                body_q, self.model.joint_X_p,
            ],
            outputs=[joint_S_s_out],
            device=self.device,
        )
