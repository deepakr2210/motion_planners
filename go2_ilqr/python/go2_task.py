"""
go2_task.py – GO2-specific cost functions for iLQR.

State layout (MuJoCo Menagerie go2.xml, floating-base):
  qpos[0:3]   base position  (x, y, z)
  qpos[3:7]   base quaternion (w, x, y, z)
  qpos[7:19]  joint angles   (FL_hip, FL_thigh, FL_calf,
                               FR_hip, FR_thigh, FR_calf,
                               RL_hip, RL_thigh, RL_calf,
                               RR_hip, RR_thigh, RR_calf)

  qvel[0:3]   base linear velocity  (world frame)
  qvel[3:6]   base angular velocity (body frame)
  qvel[6:18]  joint velocities

Tangent-space layout (ndx = 36 = 2 * nv):
  dx[0:3]    base position perturbation
  dx[3:6]    base orientation perturbation (rotation-vector in body frame)
  dx[6:18]   joint angle perturbation
  dx[18:21]  base linear-velocity perturbation
  dx[21:24]  base angular-velocity perturbation
  dx[24:36]  joint-velocity perturbation

Tasks implemented
-----------------
1. StandingTask   – hold a fixed standing pose
2. VelocityTask   – track desired forward/yaw velocity while standing

Both use the Gauss-Newton structure:
  l(x, u) = 0.5 Σ_i  w_i ||r_i(x, u)||²

so that Hessians are J^T W J (always PSD) and gradients are J^T W r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import mujoco
import numpy as np

from ilqr import CostFunction


# ---------------------------------------------------------------------------
# Default joint positions for GO2 standing
# ---------------------------------------------------------------------------

# Menagerie joint order: FL, FR, RL, RR  (matches actuator order in go2.xml)
GO2_DEFAULT_QPOS = np.array([
    0.0,  0.9, -1.8,   # FL: hip, thigh, calf
    0.0,  0.9, -1.8,   # FR
    0.0,  0.9, -1.8,   # RL
    0.0,  0.9, -1.8,   # RR
])

GO2_STANDING_HEIGHT = 0.27   # metres (trunk CoM above ground)


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product  q1 * q2  (both in (w, x, y, z) order)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    Orientation error as a 3-D rotation vector.

    err = 2 * vec( q_ref^{-1} * q )

    For small errors this is approximately the axis-angle representation
    of the relative rotation and maps directly to the tangent-space
    orientation components (dx[3:6]).
    """
    q_err = quat_mul(quat_inv(q_ref), q)
    # Ensure shortest path (flip if w < 0)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:]    # (3,)  – vector part


# ---------------------------------------------------------------------------
# Helper: build residual Jacobians for the tangent state
# ---------------------------------------------------------------------------

def _jac_height(nv: int, ndx: int) -> np.ndarray:
    """Jacobian of base-height residual w.r.t. tangent state.  Shape (1, ndx)."""
    J = np.zeros((1, ndx))
    J[0, 2] = 1.0      # dx[2] = Δz of base position
    return J


def _jac_orientation(nv: int, ndx: int) -> np.ndarray:
    """
    Jacobian of 3-D orientation-error residual w.r.t. tangent state.
    Small-angle approximation: dr_ori/d(dx[3:6]) ≈ I.
    Shape (3, ndx).
    """
    J = np.zeros((3, ndx))
    J[0:3, 3:6] = np.eye(3)
    return J


def _jac_linvel(nv: int, ndx: int) -> np.ndarray:
    """Jacobian of base linear-velocity residual (3,). Shape (3, ndx)."""
    J = np.zeros((3, ndx))
    J[0:3, nv:nv+3] = np.eye(3)
    return J


def _jac_angvel(nv: int, ndx: int) -> np.ndarray:
    """Jacobian of base angular-velocity residual (3,). Shape (3, ndx)."""
    J = np.zeros((3, ndx))
    J[0:3, nv+3:nv+6] = np.eye(3)
    return J


def _jac_joint_pos(nv: int, ndx: int) -> np.ndarray:
    """Jacobian of joint-angle residual (12,). Shape (12, ndx)."""
    J = np.zeros((12, ndx))
    J[0:12, 6:18] = np.eye(12)
    return J


def _jac_joint_vel(nv: int, ndx: int) -> np.ndarray:
    """Jacobian of joint-velocity residual (12,). Shape (12, ndx)."""
    J = np.zeros((12, ndx))
    J[0:12, nv+6:nv+18] = np.eye(12)
    return J


def _jac_ctrl(nu: int) -> np.ndarray:
    """Jacobian of control residual (nu,). Shape (nu, nu)."""
    return np.eye(nu)


# ---------------------------------------------------------------------------
# StandingTask
# ---------------------------------------------------------------------------

@dataclass
class StandingTaskConfig:
    """Cost weights for the standing task."""
    # Quadratic weights on each residual block
    w_height:     float = 200.0   # base height
    w_orient:     float = 100.0   # base orientation
    w_linvel:     float = 10.0    # base linear velocity
    w_angvel:     float = 10.0    # base angular velocity
    w_joint_pos:  float = 5.0     # joint angles vs. default
    w_joint_vel:  float = 1.0     # joint velocities
    w_ctrl:       float = 0.001   # control effort

    # Terminal cost multiplier (applied to all state terms)
    w_terminal:   float = 10.0

    # Reference
    target_height: float = GO2_STANDING_HEIGHT
    target_joint_pos: np.ndarray = field(
        default_factory=lambda: GO2_DEFAULT_QPOS.copy()
    )


class StandingTask(CostFunction):
    """
    iLQR cost for keeping GO2 in a stable standing pose.

    Uses Gauss-Newton approximation: l = 0.5 r^T W r
    so that lxx = J_x^T W J_x (always PSD).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: StandingTaskConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg   = cfg or StandingTaskConfig()

        nq, nv, nu, na = model.nq, model.nv, model.nu, model.na
        self.nq, self.nv, self.nu = nq, nv, nu
        self.ndx = 2 * nv + na

        # Pre-compute fixed Jacobians (all constant for quadratic costs)
        ndx = self.ndx
        self._Jh  = _jac_height(nv, ndx)       # (1,  ndx)
        self._Jo  = _jac_orientation(nv, ndx)  # (3,  ndx)
        self._Jv  = _jac_linvel(nv, ndx)       # (3,  ndx)
        self._Jw  = _jac_angvel(nv, ndx)       # (3,  ndx)
        self._Jqp = _jac_joint_pos(nv, ndx)    # (12, ndx)
        self._Jqv = _jac_joint_vel(nv, ndx)    # (12, ndx)
        self._Ju  = _jac_ctrl(nu)              # (12, 12)

        # Pre-compute Hessian contributions (independent of x, u)
        c = self.cfg
        self._lxx_running  = (
            c.w_height    * self._Jh.T  @ self._Jh  +
            c.w_orient    * self._Jo.T  @ self._Jo  +
            c.w_linvel    * self._Jv.T  @ self._Jv  +
            c.w_angvel    * self._Jw.T  @ self._Jw  +
            c.w_joint_pos * self._Jqp.T @ self._Jqp +
            c.w_joint_vel * self._Jqv.T @ self._Jqv
        )  # (36, 36)
        self._luu_running = c.w_ctrl * self._Ju.T @ self._Ju  # (12, 12)
        self._lxx_terminal = c.w_terminal * self._lxx_running

        # Reference quaternion = identity (upright)
        self._q_ref = np.array([1.0, 0.0, 0.0, 0.0])

    # --- helpers ---

    def _extract(self, x: np.ndarray):
        """Split state into named components."""
        qpos = x[:self.nq]
        qvel = x[self.nq:]
        base_pos  = qpos[0:3]
        base_quat = qpos[3:7]
        joint_pos = qpos[7:19]
        base_linvel = qvel[0:3]
        base_angvel = qvel[3:6]
        joint_vel   = qvel[6:18]
        return (base_pos, base_quat, joint_pos,
                base_linvel, base_angvel, joint_vel)

    def _residuals(self, x: np.ndarray, u: np.ndarray):
        """
        Compute residual blocks and aggregate gradient w.r.t. tangent state.

        Returns
        -------
        lx  : (ndx,)
        cost: float
        """
        c = self.cfg
        base_pos, base_quat, joint_pos, base_linvel, base_angvel, joint_vel \
            = self._extract(x)

        r_h  = np.array([base_pos[2] - c.target_height])          # (1,)
        r_o  = quat_error(base_quat, self._q_ref)                  # (3,)
        r_v  = base_linvel                                          # (3,)
        r_w  = base_angvel                                          # (3,)
        r_qp = joint_pos - c.target_joint_pos                      # (12,)
        r_qv = joint_vel                                            # (12,)
        r_u  = u                                                    # (12,)

        cost = 0.5 * (
            c.w_height    * r_h  @ r_h  +
            c.w_orient    * r_o  @ r_o  +
            c.w_linvel    * r_v  @ r_v  +
            c.w_angvel    * r_w  @ r_w  +
            c.w_joint_pos * r_qp @ r_qp +
            c.w_joint_vel * r_qv @ r_qv +
            c.w_ctrl      * r_u  @ r_u
        )

        # Gradient: lx = Σ J_i^T w_i r_i
        lx = (
            c.w_height    * self._Jh.T  @ r_h  +
            c.w_orient    * self._Jo.T  @ r_o  +
            c.w_linvel    * self._Jv.T  @ r_v  +
            c.w_angvel    * self._Jw.T  @ r_w  +
            c.w_joint_pos * self._Jqp.T @ r_qp +
            c.w_joint_vel * self._Jqv.T @ r_qv
        )  # (36,)
        lu = c.w_ctrl * self._Ju.T @ r_u   # (12,)

        return cost, lx, lu

    def _terminal_residuals(self, x: np.ndarray):
        c = self.cfg
        base_pos, base_quat, joint_pos, base_linvel, base_angvel, joint_vel \
            = self._extract(x)

        r_h  = np.array([base_pos[2] - c.target_height])
        r_o  = quat_error(base_quat, self._q_ref)
        r_v  = base_linvel
        r_w  = base_angvel
        r_qp = joint_pos - c.target_joint_pos
        r_qv = joint_vel

        cost = 0.5 * c.w_terminal * (
            c.w_height    * r_h  @ r_h  +
            c.w_orient    * r_o  @ r_o  +
            c.w_linvel    * r_v  @ r_v  +
            c.w_angvel    * r_w  @ r_w  +
            c.w_joint_pos * r_qp @ r_qp +
            c.w_joint_vel * r_qv @ r_qv
        )

        lx = c.w_terminal * (
            c.w_height    * self._Jh.T  @ r_h  +
            c.w_orient    * self._Jo.T  @ r_o  +
            c.w_linvel    * self._Jv.T  @ r_v  +
            c.w_angvel    * self._Jw.T  @ r_w  +
            c.w_joint_pos * self._Jqp.T @ r_qp +
            c.w_joint_vel * self._Jqv.T @ r_qv
        )
        return cost, lx

    # --- CostFunction interface ---

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: int) -> float:
        cost, _, _ = self._residuals(x, u)
        return cost

    def running_cost_derivatives(
        self, x: np.ndarray, u: np.ndarray, t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, lx, lu = self._residuals(x, u)
        # Hessians are constant (pre-computed)
        return lx, lu, self._lxx_running, self._luu_running, np.zeros((self.nu, self.ndx))

    def terminal_cost(self, x: np.ndarray) -> float:
        cost, _ = self._terminal_residuals(x)
        return cost

    def terminal_cost_derivatives(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, lx = self._terminal_residuals(x)
        return lx, self._lxx_terminal


# ---------------------------------------------------------------------------
# VelocityTask  (forward / lateral / yaw velocity tracking + standing)
# ---------------------------------------------------------------------------

@dataclass
class VelocityTaskConfig:
    """Cost weights for velocity-tracking task."""
    # Target base velocities (world / body frame)
    target_vx:  float = 0.5    # forward  [m/s]
    target_vy:  float = 0.0    # lateral  [m/s]
    target_yaw: float = 0.0    # yaw rate [rad/s]

    target_height:    float = GO2_STANDING_HEIGHT
    target_joint_pos: np.ndarray = field(
        default_factory=lambda: GO2_DEFAULT_QPOS.copy()
    )

    # Weights
    w_height:    float = 150.0
    w_orient:    float = 50.0
    w_vel_xy:    float = 100.0   # vx / vy tracking
    w_yaw:       float = 50.0    # yaw-rate tracking
    w_vz:        float = 20.0    # penalise vertical velocity
    w_angvel_xy: float = 5.0     # roll/pitch rate
    w_joint_pos: float = 2.0
    w_joint_vel: float = 0.5
    w_ctrl:      float = 0.001
    w_terminal:  float = 5.0


class VelocityTask(CostFunction):
    """
    iLQR cost for velocity-tracking locomotion.

    Encourages the GO2 to move at (target_vx, target_vy) with target yaw rate,
    while maintaining nominal height, orientation, and joint configuration.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: VelocityTaskConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg   = cfg or VelocityTaskConfig()

        nq, nv, nu, na = model.nq, model.nv, model.nu, model.na
        self.nq, self.nv, self.nu = nq, nv, nu
        self.ndx = 2 * nv + na
        ndx = self.ndx

        # Reference quaternion = identity
        self._q_ref = np.array([1.0, 0.0, 0.0, 0.0])

        # Pre-compute fixed Jacobians
        self._Jh  = _jac_height(nv, ndx)
        self._Jo  = _jac_orientation(nv, ndx)
        self._Jqp = _jac_joint_pos(nv, ndx)
        self._Jqv = _jac_joint_vel(nv, ndx)
        self._Ju  = _jac_ctrl(nu)

        # Velocity Jacobians (individual components)
        # vx: tangent index nv+0, vy: nv+1, vz: nv+2
        # wx: nv+3, wy: nv+4, wz: nv+5
        self._Jvx = np.zeros((1, ndx)); self._Jvx[0, nv+0] = 1.0
        self._Jvy = np.zeros((1, ndx)); self._Jvy[0, nv+1] = 1.0
        self._Jvz = np.zeros((1, ndx)); self._Jvz[0, nv+2] = 1.0
        self._Jwz = np.zeros((1, ndx)); self._Jwz[0, nv+5] = 1.0
        # roll/pitch rates
        self._Jwxy = np.zeros((2, ndx))
        self._Jwxy[0, nv+3] = 1.0
        self._Jwxy[1, nv+4] = 1.0

        self._precompute_hessians()

    def _precompute_hessians(self) -> None:
        c = self.cfg
        self._lxx_running = (
            c.w_height    * self._Jh.T   @ self._Jh   +
            c.w_orient    * self._Jo.T   @ self._Jo   +
            c.w_vel_xy    * self._Jvx.T  @ self._Jvx  +
            c.w_vel_xy    * self._Jvy.T  @ self._Jvy  +
            c.w_vz        * self._Jvz.T  @ self._Jvz  +
            c.w_yaw       * self._Jwz.T  @ self._Jwz  +
            c.w_angvel_xy * self._Jwxy.T @ self._Jwxy +
            c.w_joint_pos * self._Jqp.T  @ self._Jqp  +
            c.w_joint_vel * self._Jqv.T  @ self._Jqv
        )
        self._luu_running  = c.w_ctrl * self._Ju.T @ self._Ju
        self._lxx_terminal = c.w_terminal * self._lxx_running

    def _extract(self, x):
        qpos = x[:self.nq]; qvel = x[self.nq:]
        return (qpos[0:3], qpos[3:7], qpos[7:19],
                qvel[0:3], qvel[3:6], qvel[6:18])

    def _residuals(self, x, u):
        c = self.cfg
        base_pos, base_quat, joint_pos, base_linvel, base_angvel, joint_vel \
            = self._extract(x)

        r_h   = np.array([base_pos[2] - c.target_height])
        r_o   = quat_error(base_quat, self._q_ref)
        r_vx  = np.array([base_linvel[0] - c.target_vx])
        r_vy  = np.array([base_linvel[1] - c.target_vy])
        r_vz  = np.array([base_linvel[2]])
        r_wz  = np.array([base_angvel[2] - c.target_yaw])
        r_wxy = base_angvel[0:2]
        r_qp  = joint_pos - c.target_joint_pos
        r_qv  = joint_vel
        r_u   = u

        cost = 0.5 * (
            c.w_height    * r_h  @ r_h  +
            c.w_orient    * r_o  @ r_o  +
            c.w_vel_xy    * r_vx @ r_vx +
            c.w_vel_xy    * r_vy @ r_vy +
            c.w_vz        * r_vz @ r_vz +
            c.w_yaw       * r_wz @ r_wz +
            c.w_angvel_xy * r_wxy @ r_wxy +
            c.w_joint_pos * r_qp @ r_qp +
            c.w_joint_vel * r_qv @ r_qv +
            c.w_ctrl      * r_u @ r_u
        )

        lx = (
            c.w_height    * self._Jh.T   @ r_h   +
            c.w_orient    * self._Jo.T   @ r_o   +
            c.w_vel_xy    * self._Jvx.T  @ r_vx  +
            c.w_vel_xy    * self._Jvy.T  @ r_vy  +
            c.w_vz        * self._Jvz.T  @ r_vz  +
            c.w_yaw       * self._Jwz.T  @ r_wz  +
            c.w_angvel_xy * self._Jwxy.T @ r_wxy +
            c.w_joint_pos * self._Jqp.T  @ r_qp  +
            c.w_joint_vel * self._Jqv.T  @ r_qv
        )
        lu = c.w_ctrl * self._Ju.T @ r_u
        return cost, lx, lu

    def _terminal_residuals(self, x):
        c = self.cfg
        base_pos, base_quat, joint_pos, base_linvel, base_angvel, joint_vel \
            = self._extract(x)

        r_h   = np.array([base_pos[2] - c.target_height])
        r_o   = quat_error(base_quat, self._q_ref)
        r_vx  = np.array([base_linvel[0] - c.target_vx])
        r_vy  = np.array([base_linvel[1] - c.target_vy])
        r_vz  = np.array([base_linvel[2]])
        r_wz  = np.array([base_angvel[2] - c.target_yaw])
        r_wxy = base_angvel[0:2]
        r_qp  = joint_pos - c.target_joint_pos
        r_qv  = joint_vel

        cost = 0.5 * c.w_terminal * (
            c.w_height    * r_h  @ r_h  +
            c.w_orient    * r_o  @ r_o  +
            c.w_vel_xy    * r_vx @ r_vx +
            c.w_vel_xy    * r_vy @ r_vy +
            c.w_vz        * r_vz @ r_vz +
            c.w_yaw       * r_wz @ r_wz +
            c.w_angvel_xy * r_wxy @ r_wxy +
            c.w_joint_pos * r_qp @ r_qp +
            c.w_joint_vel * r_qv @ r_qv
        )

        lx = c.w_terminal * (
            c.w_height    * self._Jh.T   @ r_h   +
            c.w_orient    * self._Jo.T   @ r_o   +
            c.w_vel_xy    * self._Jvx.T  @ r_vx  +
            c.w_vel_xy    * self._Jvy.T  @ r_vy  +
            c.w_vz        * self._Jvz.T  @ r_vz  +
            c.w_yaw       * self._Jwz.T  @ r_wz  +
            c.w_angvel_xy * self._Jwxy.T @ r_wxy +
            c.w_joint_pos * self._Jqp.T  @ r_qp  +
            c.w_joint_vel * self._Jqv.T  @ r_qv
        )
        return cost, lx

    def running_cost(self, x, u, t):
        cost, _, _ = self._residuals(x, u)
        return cost

    def running_cost_derivatives(self, x, u, t):
        _, lx, lu = self._residuals(x, u)
        return lx, lu, self._lxx_running, self._luu_running, np.zeros((self.nu, self.ndx))

    def terminal_cost(self, x):
        cost, _ = self._terminal_residuals(x)
        return cost

    def terminal_cost_derivatives(self, x):
        _, lx = self._terminal_residuals(x)
        return lx, self._lxx_terminal
