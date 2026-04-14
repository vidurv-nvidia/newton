# IKOptimizerQP: ADMM-Based Differential IK for Newton

## Goal

Add a QP-based differential IK optimizer to Newton that solves for joint displacements via Quadratic Programming using ADMM, implemented as pure Warp kernels. Plugs into the existing `IKSolver` infrastructure. Enables real-time velocity-based servo control on both CPU and GPU.

## QP Formulation

Each step solves:

```
min_Δq  ½ ||J Δq - e||²_W + ½ λ ||Δq||²
s.t.    q_lower - q ≤ Δq ≤ q_upper - q    (joint position limits)
        -v_max·dt ≤ Δq ≤ v_max·dt          (velocity limits, optional)
        G Δq ≤ h                            (general linear inequalities, optional)
```

Standard QP form: `min ½ x^T H x + c^T x  s.t. Ax ≤ b` where:
- `H = J^T W J + λI` (positive definite, n_dof × n_dof)
- `c = -J^T W e`
- Box constraints derived from joint limits + velocity limits
- General inequality constraints passed as G, h matrices

## ADMM Algorithm

Per batch element, per QP solve:

```
Initialize: Δq = 0, z = 0, u = 0
For k = 1..qp_max_iters:
    1. Δq-update: solve (H + ρI) Δq = -c + ρ(z - u)
    2. z-update:  z = clamp(Δq + u, lb, ub)  [box projection]
                  For general inequalities: z = project_onto_halfspaces(Δq + u, G, h)
    3. u-update:  u = u + Δq - z
    Converged if ||Δq - z||_inf < tol AND ||z - z_prev||_inf < tol
```

Step 1 is a symmetric positive-definite solve (Cholesky, tiled by DOF count). Step 2 is element-wise for box constraints. Step 3 is vector add. All parallel across batch elements.

## Files

| File | Action | Lines |
|------|--------|-------|
| `newton/_src/sim/ik/ik_qp_optimizer.py` | Create | ~450 |
| `newton/_src/sim/ik/ik_solver.py` | Modify | ~15 |
| `newton/_src/sim/ik/__init__.py` | Modify | ~2 |
| `newton/ik.py` | Modify | ~2 |
| `newton/tests/test_ik_qp.py` | Create | ~200 |

## IKOptimizerQP Class

### Constructor

```python
class IKOptimizerQP:
    def __init__(
        self,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        jacobian_mode: IKJacobianType = IKJacobianType.ANALYTIC,
        qp_max_iters: int = 20,
        qp_rho: float = 1.0,
        qp_tol: float = 1e-6,
        dt: float = 0.01,
        velocity_limit: np.ndarray | None = None,
        *,
        problem_idx: wp.array[wp.int32] | None = None,
    ):
```

### Required Interface Methods

```python
def step(self, joint_q_in, joint_q_out, iterations=1, step_size=1.0,
         tol=None, check_every=5) -> int:
    """Run `iterations` outer IK steps. Each outer step:
    1. Compute residuals + Jacobian at current q
    2. Build QP (H, c, constraint bounds)
    3. Solve QP via ADMM → get Δq
    4. Apply: q = q + step_size * Δq
    Returns iterations used."""

def reset(self) -> None:
    """Clear ADMM state (z, u dual variables)."""

def compute_costs(self, joint_q) -> wp.array[wp.float32]:
    """Evaluate sum-of-squared residuals. Same as LM/LBFGS."""
```

### Warp Kernels

All kernels are tiled by `(TILE_N_DOFS, TILE_N_RESIDUALS)` for compile-time optimization, following the same pattern as `IKOptimizerLM`.

1. **`_build_qp_matrices`** — Computes `H = J^T W J + (λ + ρ)I` and `g = -J^T W e + ρ(z - u)` from the Jacobian and residuals. One kernel launch, operates on `[n_batch]` elements.

2. **`_cholesky_factor`** — In-place Cholesky factorization of H (n_dof × n_dof per batch element). Tiled kernel.

3. **`_cholesky_solve`** — Forward/backward substitution to solve `L L^T x = b`. Tiled kernel.

4. **`_admm_z_update`** — For each DOF: `z[i] = clamp(Δq[i] + u[i], lb[i], ub[i])`. Element-wise, trivially parallel.

5. **`_admm_u_update`** — `u[i] += Δq[i] - z[i]`. Element-wise.

6. **`_admm_check_convergence`** — Computes `max(|Δq - z|)` and `max(|z - z_prev|)`, sets converged flag per batch element.

7. **`_compute_constraint_bounds`** — Merges joint position limits and velocity limits into unified `lb, ub` arrays: `lb[i] = max(q_lower[i] - q[i], -v_max[i] * dt)`, `ub[i] = min(q_upper[i] - q[i], v_max[i] * dt)`.

### Constraint Handling

**Box constraints** (joint position + velocity limits): Handled natively by ADMM z-update projection (clamping). Zero overhead.

**General linear inequalities** (`G Δq ≤ h`): Augmented into the ADMM formulation. The z-update becomes a projection onto the intersection of box constraints and halfspaces. For MVP, if `G` is provided, use iterative projection (project onto each halfspace sequentially). This is approximate but sufficient for a small number of constraints.

### Integration with IKSolver

Modify `ik_solver.py`:

1. Add `QP = "qp"` to `IKOptimizer` enum
2. In `IKSolver.__init__`: add `elif optimizer is IKOptimizer.QP:` branch constructing `IKOptimizerQP`
3. In `IKSolver.step()`: add dispatch branch (identical to LM — passes `iterations`, `step_size`, `tol`, `check_every`)
4. New IKSolver constructor params: `qp_max_iters`, `qp_rho`, `qp_tol`, `qp_dt`, `qp_velocity_limit`

### Specialized Kernel Pattern

Following `IKOptimizerLM`'s pattern: use `__new__()` with a class-level `_cache` to generate specialized kernel subclasses per `(n_dofs, n_residuals, arch)` tuple. This allows Warp to compile optimized tiled kernels at the exact DOF/residual dimensions.

### Reused Infrastructure

Copied/reused from `IKOptimizerLM`:
- `_jacobian_at()`, `_jacobian_analytic()`, `_jacobian_autodiff()` — Jacobian computation
- `_compute_residuals()`, `_residuals_analytic()`, `_residuals_autodiff()` — residual evaluation
- `_ctx_solver()`, `BatchCtx` — batch context management
- `_integrate_dq()` — joint displacement integration
- `compute_costs()` — cost evaluation
- FK pass via `fk_accum` / `eval_fk_batched`

### Public API Exposure

Add to `newton/ik.py`:
```python
from ._src.sim.ik import IKOptimizerQP
```

Add to `newton/_src/sim/ik/__init__.py`:
```python
from .ik_qp_optimizer import IKOptimizerQP
```

### Tests

`newton/tests/test_ik_qp.py` using `unittest` (per AGENTS.md):

1. **test_qp_converges_position** — QP solver reaches target position within tolerance
2. **test_qp_converges_rotation** — QP solver reaches target orientation within tolerance
3. **test_qp_respects_joint_limits** — Output q stays within joint limits
4. **test_qp_velocity_mode** — With iterations=1, produces smooth displacement
5. **test_qp_matches_lm_accuracy** — For unconstrained case, QP and LM reach similar accuracy
6. **test_qp_batch** — Works with n_problems > 1

### Expected Performance

For Panda (7-DOF), per QP solve:
- Build H, c: ~1 Jacobian evaluation (same as 1 LM iteration)
- ADMM 20 iterations: 20 × (7×7 Cholesky solve + 7-element clamp + 7-element add) ≈ trivial
- Total per tick: ~1 Jacobian eval + negligible ADMM overhead ≈ similar to Newton 1-iter (~0.2ms)
- With better convergence (velocity-space formulation): expected sub-mm tracking at 0.2ms — competitive with mink

### Non-Goals

- Replacing LM/LBFGS for position-mode IK (QP is for velocity/servo mode)
- Sparse QP for large DOF systems (dense Cholesky is fine for ≤ 30 DOF)
- Warm-starting ADMM across ticks (could be added later)
- Collision avoidance implementation (just the G, h interface)
