"""
ilqr.py – iLQR (iterative Linear Quadratic Regulator) for MuJoCo systems.

Algorithm
---------
Implements the DDP / iLQR algorithm from:
  - Tassa et al., IROS 2012: "Synthesis and stabilization of complex
    behaviours through online trajectory optimisation"
  - Li & Todorov, ICINCO 2004: "Iterative LQR design for nonlinear
    biological movement systems"

Notation
--------
  T    : planning horizon (steps)
  nq   : MuJoCo position dimension
  nv   : MuJoCo velocity dimension
  nu   : MuJoCo control dimension
  nx   : full state dimension = nq + nv
  ndx  : tangent-space state = 2 * nv  (handles floating-base quaternion)

State derivative space
----------------------
MuJoCo uses a unit quaternion for floating-base orientation. The iLQR
backward/forward passes operate in the 36-D tangent space (ndx = 2*nv):

  dx = [dpos (nv),  dvel (nv)]

where dpos is the velocity-space positional difference (mj_differentiatePos)
and dvel is the ordinary velocity difference.

Perturbations are applied via mj_integratePos (position part) and additive
offset (velocity part).

Dynamics Jacobians
------------------
Uses MuJoCo's mjd_transitionFD which computes

  A[t] = ∂x_{t+1}/∂x_t   (ndx × ndx)
  B[t] = ∂x_{t+1}/∂u_t   (ndx × nu)

in the tangent space, handling quaternion integration internally.

Cost interface
--------------
The caller supplies a CostFunction subclass with analytical gradients /
Hessians.  This avoids the O(ndx²) finite-difference Hessian, which for
GO2 (ndx=36) would be 1296 MuJoCo calls per time step.

Backward pass
-------------
Standard Riccati recursion from t=T back to t=0:

  Q_x  = l_x  + A^T V_x
  Q_u  = l_u  + B^T V_x
  Q_xx = l_xx + A^T V_xx A
  Q_uu = l_uu + B^T V_xx B  +  μ I   (regularisation)
  Q_ux = l_ux + B^T V_xx A

  k[t]  = −Q_uu^{−1} Q_u       (feedforward)
  K[t]  = −Q_uu^{−1} Q_ux      (feedback)

  V_x  = Q_x  + K^T Q_uu k + K^T Q_u + Q_ux^T k
  V_xx = Q_xx + K^T Q_uu K + K^T Q_ux + Q_ux^T K

Forward line search
-------------------
Given step size α ∈ (0, 1]:
  x_0 = x0
  u_t = clip( U[t] + α k[t] + K[t] (x_t − X[t]) )
  x_{t+1} = f(x_t, u_t)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ILQRConfig:
    """Hyper-parameters for the iLQR solver."""
    horizon:       int   = 100      # planning horizon T
    mu_init:       float = 1.0      # initial Tikhonov regularisation
    mu_min:        float = 1e-6     # minimum regularisation
    mu_max:        float = 1e10     # maximum regularisation
    delta_0:       float = 2.0      # regularisation scaling factor δ₀
    alpha_min:     float = 1e-8     # minimum line-search step size
    tol:           float = 1e-4     # convergence: relative cost improvement
    max_iter:      int   = 50       # maximum outer iterations
    fd_eps:        float = 1e-6     # finite-difference epsilon (for dyn. Jac.)
    ctrl_limit:    float = 33.5     # symmetric torque clamp [Nm]
    verbose:       bool  = True     # print iteration info


# ---------------------------------------------------------------------------
# Cost function interface
# ---------------------------------------------------------------------------

class CostFunction(ABC):
    """
    Abstract cost function for iLQR.

    Subclass this and implement all five methods.  The Gauss-Newton
    structure (analytical Jacobians → PSD Hessians) is strongly preferred
    over finite-difference Hessians for performance.
    """

    @abstractmethod
    def running_cost(self, x: np.ndarray, u: np.ndarray, t: int) -> float:
        """Scalar running cost l(x, u, t)."""

    @abstractmethod
    def running_cost_derivatives(
        self, x: np.ndarray, u: np.ndarray, t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lx, lu, lxx, luu, lux) at (x, u, t).
          lx  : (ndx,)
          lu  : (nu,)
          lxx : (ndx, ndx)  – must be PSD
          luu : (nu,  nu)   – must be PSD
          lux : (nu,  ndx)
        """

    @abstractmethod
    def terminal_cost(self, x: np.ndarray) -> float:
        """Scalar terminal cost lf(x)."""

    @abstractmethod
    def terminal_cost_derivatives(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (lx, lxx) at x.
          lx  : (ndx,)
          lxx : (ndx, ndx)  – must be PSD
        """


# ---------------------------------------------------------------------------
# Core iLQR solver
# ---------------------------------------------------------------------------

class ILQR:
    """
    iLQR solver for a MuJoCo-based dynamical system.

    Parameters
    ----------
    model  : mujoco.MjModel  – the compiled model
    cost   : CostFunction    – user-supplied cost with analytical derivatives
    config : ILQRConfig      – solver hyper-parameters (optional)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: CostFunction,
        config: Optional[ILQRConfig] = None,
    ) -> None:
        self.model  = model
        self.cost   = cost
        self.cfg    = config or ILQRConfig()

        # --- dimensions ---
        self.nq  = model.nq                       # position DOF     (19 for GO2)
        self.nv  = model.nv                       # velocity DOF     (18 for GO2)
        self.na  = model.na                       # actuator states  (0 for GO2)
        self.nu  = model.nu                       # control DOF      (12 for GO2)
        self.nx  = self.nq + self.nv              # full state       (37)
        self.ndx = 2 * self.nv + self.na          # tangent state    (36 for GO2)
        self.T   = self.cfg.horizon

        # --- MuJoCo data objects (separate instances for rollout vs. FD) ---
        self.data    = mujoco.MjData(model)
        self.data_fd = mujoco.MjData(model)

        self._allocate()

    # ------------------------------------------------------------------
    # Memory allocation
    # ------------------------------------------------------------------

    def _allocate(self) -> None:
        T, ndx, nu, nx = self.T, self.ndx, self.nu, self.nx

        # Trajectory
        self.X = np.zeros((T + 1, nx))
        self.U = np.zeros((T, nu))

        # Dynamics Jacobians (tangent space)
        self.A = np.zeros((T, ndx, ndx))   # ∂f/∂x
        self.B = np.zeros((T, ndx, nu))    # ∂f/∂u

        # Cost first/second derivatives
        self.lx  = np.zeros((T + 1, ndx))
        self.lu  = np.zeros((T, nu))
        self.lxx = np.zeros((T + 1, ndx, ndx))
        self.luu = np.zeros((T, nu, nu))
        self.lux = np.zeros((T, nu, ndx))

        # Value function (backward pass)
        self.Vx  = np.zeros(ndx)
        self.Vxx = np.zeros((ndx, ndx))

        # Policy gains
        self.k = np.zeros((T, nu))        # feedforward
        self.K = np.zeros((T, nu, ndx))   # feedback

        # Expected cost improvement dV = dV1 α + dV2 α²
        self.dV1 = 0.0
        self.dV2 = 0.0

        # Regularisation state
        self.mu    = self.cfg.mu_init
        self.delta = self.cfg.delta_0

    # ------------------------------------------------------------------
    # MuJoCo helpers
    # ------------------------------------------------------------------

    def _set_state(self, d: mujoco.MjData, x: np.ndarray) -> None:
        d.qpos[:] = x[:self.nq]
        d.qvel[:] = x[self.nq:]
        mujoco.mj_forward(self.model, d)

    def _get_state(self, d: mujoco.MjData) -> np.ndarray:
        return np.concatenate([d.qpos.copy(), d.qvel.copy()])

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One MuJoCo step: x_{t+1} = f(x_t, u_t)."""
        self._set_state(self.data, x)
        self.data.ctrl[:] = np.clip(u, -self.cfg.ctrl_limit, self.cfg.ctrl_limit)
        mujoco.mj_step(self.model, self.data)
        return self._get_state(self.data)

    def state_diff(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Tangent-space difference  dx = x1 ⊖ x2.
        Returns [dpos (nv), dvel (nv)] where dpos uses mj_differentiatePos
        to correctly handle the floating-base quaternion.
        """
        dpos = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, dpos, 1.0,
                                   x2[:self.nq], x1[:self.nq])
        dvel = x1[self.nq:] - x2[self.nq:]
        return np.concatenate([dpos, dvel])

    def state_add(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Add tangent vector  x_new = x ⊕ dx.
        Uses mj_integratePos for the position part (quaternion-safe).
        """
        q_new = x[:self.nq].copy()
        mujoco.mj_integratePos(self.model, q_new, dx[:self.nv], 1.0)
        v_new = x[self.nq:] + dx[self.nv:]
        return np.concatenate([q_new, v_new])

    # ------------------------------------------------------------------
    # Forward rollout
    # ------------------------------------------------------------------

    def forward_rollout(
        self, x0: np.ndarray, U: np.ndarray
    ) -> float:
        """
        Roll out x_{t+1} = f(x_t, U[t]) and accumulate total cost.
        Stores trajectory in self.X.
        """
        self.X[0] = x0.copy()
        total = 0.0
        for t in range(self.T):
            self.X[t + 1] = self._step(self.X[t], U[t])
            total += self.cost.running_cost(self.X[t], U[t], t)
        total += self.cost.terminal_cost(self.X[self.T])
        return total

    def _rollout_new(
        self, x0: np.ndarray, U_new: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Rollout with a candidate control sequence; does NOT overwrite self.X.
        Returns (X_new, total_cost).
        """
        X_new = np.empty((self.T + 1, self.nx))
        X_new[0] = x0.copy()
        total = 0.0
        for t in range(self.T):
            X_new[t + 1] = self._step(X_new[t], U_new[t])
            total += self.cost.running_cost(X_new[t], U_new[t], t)
        total += self.cost.terminal_cost(X_new[self.T])
        return X_new, total

    # ------------------------------------------------------------------
    # Dynamics linearisation  (A, B via mjd_transitionFD)
    # ------------------------------------------------------------------

    def compute_dynamics_jacobians(self) -> None:
        """
        Compute A[t], B[t] for t=0..T-1 using MuJoCo's mjd_transitionFD.

        mjd_transitionFD sets up the state and ctrl from d, calls a
        forward integration, and returns centred-FD Jacobians in the
        2*nv tangent space – quaternion integration is handled internally.
        """
        eps = self.cfg.fd_eps
        ndx = self.ndx
        nu  = self.nu

        # mjd_transitionFD expects 2-D arrays of shape (ndx, ndx) / (ndx, nu)
        A = np.zeros((ndx, ndx))
        B = np.zeros((ndx, nu))

        for t in range(self.T):
            self._set_state(self.data_fd, self.X[t])
            self.data_fd.ctrl[:] = np.clip(
                self.U[t], -self.cfg.ctrl_limit, self.cfg.ctrl_limit
            )

            mujoco.mjd_transitionFD(
                self.model, self.data_fd,
                eps, True,          # centred finite differences
                A, B,
                None, None,         # skip sensor Jacobians C, D
            )

            self.A[t] = A.copy()
            self.B[t] = B.copy()

    # ------------------------------------------------------------------
    # Cost derivatives  (analytical, from CostFunction interface)
    # ------------------------------------------------------------------

    def compute_cost_derivatives(self) -> None:
        """
        Fill lx, lu, lxx, luu, lux from the user-supplied CostFunction.
        Running steps t=0..T-1, terminal step t=T.
        """
        for t in range(self.T):
            lx, lu, lxx, luu, lux = self.cost.running_cost_derivatives(
                self.X[t], self.U[t], t
            )
            self.lx[t]  = lx
            self.lu[t]  = lu
            self.lxx[t] = lxx
            self.luu[t] = luu
            self.lux[t] = lux

        lx_T, lxx_T = self.cost.terminal_cost_derivatives(self.X[self.T])
        self.lx[self.T]  = lx_T
        self.lxx[self.T] = lxx_T

    # ------------------------------------------------------------------
    # Backward pass  (Riccati recursion)
    # ------------------------------------------------------------------

    def backward_pass(self) -> bool:
        """
        Riccati recursion from t=T to t=0.

        Returns True on success, False if Q_uu is not positive definite
        (caller should increase μ and retry).
        """
        # Initialise value function at terminal step
        self.Vx  = self.lx[self.T].copy()
        self.Vxx = self.lxx[self.T].copy()
        self.dV1 = 0.0
        self.dV2 = 0.0

        for t in reversed(range(self.T)):
            A   = self.A[t]      # (ndx, ndx)
            B   = self.B[t]      # (ndx, nu)
            lx  = self.lx[t]     # (ndx,)
            lu  = self.lu[t]     # (nu,)
            lxx = self.lxx[t]    # (ndx, ndx)
            luu = self.luu[t]    # (nu, nu)
            lux = self.lux[t]    # (nu, ndx)

            # Q-function approximation
            #   Q_xx = l_xx + A^T V_xx A
            #   Q_uu = l_uu + B^T V_xx B  +  μ I
            #   Q_ux = l_ux + B^T V_xx A
            #   Q_x  = l_x  + A^T V_x
            #   Q_u  = l_u  + B^T V_x

            VxxA = self.Vxx @ A           # (ndx, ndx)
            VxxB = self.Vxx @ B           # (ndx, nu)

            Q_xx = lxx + A.T @ VxxA
            Q_uu = luu + B.T @ VxxB + self.mu * np.eye(self.nu)
            Q_ux = lux + B.T @ VxxA
            Q_x  = lx  + A.T @ self.Vx
            Q_u  = lu  + B.T @ self.Vx

            # Symmetrise Q_uu for numerical stability
            Q_uu = 0.5 * (Q_uu + Q_uu.T)

            # Cholesky factorisation of Q_uu
            try:
                L = np.linalg.cholesky(Q_uu)
            except np.linalg.LinAlgError:
                return False  # not PD → increase regularisation

            # Gains:  k = -Q_uu^{-1} Q_u,   K = -Q_uu^{-1} Q_ux
            k_t = -np.linalg.solve(Q_uu, Q_u)    # (nu,)
            K_t = -np.linalg.solve(Q_uu, Q_ux)   # (nu, ndx)

            self.k[t] = k_t
            self.K[t] = K_t

            # Expected improvement (for Armijo condition check)
            self.dV1 += Q_u @ k_t
            self.dV2 += 0.5 * k_t @ (Q_uu @ k_t)

            # Value function update
            #   V_x  = Q_x  + K^T (Q_uu k + Q_u) + Q_ux^T k
            #   V_xx = Q_xx + K^T Q_uu K + K^T Q_ux + Q_ux^T K
            Quu_k  = Q_uu @ k_t                   # (nu,)
            self.Vx  = Q_x + K_t.T @ (Quu_k + Q_u) + Q_ux.T @ k_t
            self.Vxx = Q_xx + K_t.T @ Q_uu @ K_t \
                             + K_t.T @ Q_ux        \
                             + Q_ux.T @ K_t
            self.Vxx = 0.5 * (self.Vxx + self.Vxx.T)  # keep symmetric

        return True

    # ------------------------------------------------------------------
    # Forward pass with Armijo line search
    # ------------------------------------------------------------------

    def forward_pass(
        self, x0: np.ndarray, U_ref: np.ndarray, cost_ref: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        """
        Line search over α ∈ {1, 0.5, 0.25, …, α_min}.

        At each candidate α:
          u_t = clip( U_ref[t] + α k[t] + K[t] (x_t − X[t]) )

        Accepts the first α satisfying the Armijo sufficient-decrease
        condition:

          ΔJ_actual ≥ 0.1 * ΔJ_expected(α)

        Returns (U_new, X_new, new_cost, alpha) on success,
                (None, None, cost_ref, 0) on failure.
        """
        alpha = 1.0
        while alpha >= self.cfg.alpha_min:
            U_new = np.empty_like(U_ref)
            X_new = np.empty((self.T + 1, self.nx))
            X_new[0] = x0.copy()
            total = 0.0

            for t in range(self.T):
                # State deviation in tangent space
                dx = self.state_diff(X_new[t], self.X[t])   # (ndx,)

                # Compute candidate control
                u_new = U_ref[t] + alpha * self.k[t] + self.K[t] @ dx
                u_new = np.clip(u_new, -self.cfg.ctrl_limit, self.cfg.ctrl_limit)
                U_new[t] = u_new

                X_new[t + 1] = self._step(X_new[t], u_new)
                total += self.cost.running_cost(X_new[t], u_new, t)

            total += self.cost.terminal_cost(X_new[self.T])

            # Armijo sufficient-decrease condition
            dJ_expected = alpha * self.dV1 + alpha ** 2 * self.dV2
            dJ_actual   = cost_ref - total

            if dJ_expected < 0:
                # Expected improvement is positive (dV1,dV2 < 0 convention)
                # Flip sign for comparison
                pass

            if dJ_actual >= 0.1 * abs(dJ_expected):
                return U_new, X_new, total, alpha

            alpha *= 0.5

        return None, None, cost_ref, 0.0

    # ------------------------------------------------------------------
    # Regularisation schedule
    # ------------------------------------------------------------------

    def _increase_regularisation(self) -> None:
        self.delta = max(self.delta * self.cfg.delta_0, self.cfg.delta_0)
        self.mu    = max(self.mu  * self.delta,         self.cfg.mu_min)
        if self.mu > self.cfg.mu_max:
            self.mu = self.cfg.mu_max

    def _decrease_regularisation(self) -> None:
        self.delta = min(self.delta / self.cfg.delta_0, 1.0 / self.cfg.delta_0)
        self.mu    = max(self.mu * self.delta, self.cfg.mu_min)

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        U_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run iLQR optimisation from initial state x0.

        Parameters
        ----------
        x0     : initial state (nx,)
        U_init : initial control sequence (T, nu) – zeros if None

        Returns
        -------
        U_opt  : optimised control sequence (T, nu)
        X_opt  : optimised state trajectory  (T+1, nx)
        cost   : final total cost
        """
        if U_init is not None:
            self.U = U_init.copy()
        else:
            self.U = np.zeros((self.T, self.nu))

        # Reset regularisation
        self.mu    = self.cfg.mu_init
        self.delta = self.cfg.delta_0

        # Initial forward rollout
        cost = self.forward_rollout(x0, self.U)

        if self.cfg.verbose:
            print(f"{'Iter':>5}  {'Cost':>14}  {'dCost':>12}  "
                  f"{'alpha':>8}  {'mu':>10}")
            print("-" * 58)
            print(f"{'init':>5}  {cost:>14.6f}  {'–':>12}  {'–':>8}  "
                  f"{self.mu:>10.2e}")

        t0 = time.perf_counter()

        for it in range(self.cfg.max_iter):
            # ---- dynamics & cost derivatives ----
            self.compute_dynamics_jacobians()
            self.compute_cost_derivatives()

            # ---- backward pass (with regularisation retry) ----
            bp_ok = False
            for _ in range(10):            # up to 10 regularisation increases
                if self.backward_pass():
                    bp_ok = True
                    break
                self._increase_regularisation()
                if self.mu >= self.cfg.mu_max:
                    break

            if not bp_ok:
                if self.cfg.verbose:
                    print(f"  Backward pass failed at iter {it}, stopping.")
                break

            # ---- forward pass / line search ----
            U_new, X_new, new_cost, alpha = self.forward_pass(
                x0, self.U, cost
            )

            if U_new is None:           # line search failed
                self._increase_regularisation()
                if self.cfg.verbose:
                    print(f"{it+1:>5}  line-search failed  mu={self.mu:.2e}")
                continue

            # ---- accept step ----
            dcost = cost - new_cost
            cost  = new_cost
            self.U  = U_new
            self.X  = X_new             # type: ignore[assignment]

            self._decrease_regularisation()

            if self.cfg.verbose:
                print(f"{it+1:>5}  {cost:>14.6f}  {dcost:>12.6f}  "
                      f"{alpha:>8.4f}  {self.mu:>10.2e}")

            # ---- convergence check ----
            if abs(dcost) < self.cfg.tol * (1.0 + abs(cost)):
                if self.cfg.verbose:
                    print(f"  Converged at iter {it+1}.")
                break

        elapsed = time.perf_counter() - t0
        if self.cfg.verbose:
            print(f"  Total wall time: {elapsed:.3f} s")

        return self.U.copy(), self.X.copy(), cost
