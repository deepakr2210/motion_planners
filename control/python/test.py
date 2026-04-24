"""
QP Differential IK Control (Simplified)
========================================
Resolved-rate IK solved as a QP using MuJoCo Jacobians.

Cost
----
    min   (1/2) ‖ J q̇ + α e ‖²_W                     ← primary task (EE tracking)
    q̇    + (λ² / 2) ‖ q̇ ‖²                            ← damping (numerical stability)
          + (β  / 2) ‖ q̇ + k_post (q − q_ref) ‖²      ← soft posture task (Type 1)

Hard constraints (enforced exactly)
-----------------------------------
    velocity limits :  dq_min ≤ q̇ ≤ dq_max
    position limits :  (q_min − q)/dt ≤ q̇ ≤ (q_max − q)/dt

Both reduce to per-joint box bounds on q̇, so they are intersected into a
single set of lower/upper bounds and passed to OSQP with A = I.

Standard QP form solved:
    min  (1/2) q̇ᵀ P q̇ + cᵀ q̇
    s.t. lb ≤ q̇ ≤ ub
"""
from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp


class QPDiffIKControl:

    def __init__(
        self,
        model:    mujoco.MjModel,
        data:     mujoco.MjData,
        ee_site:  str,
        dt:       float,
        W:        Optional[np.ndarray] = None,
        lam:      float = 0.01,
        alpha:    float = 1.0,
        # posture task
        q_ref:    Optional[np.ndarray] = None,
        beta:     float = 0.0,
        k_post:   float = 1.0,
        # limits
        dq_max:   Optional[np.ndarray] = None,
        use_jnt_range: bool = True,
    ):
        """
        Parameters
        ----------
        model, data    : MuJoCo model/data pair
        ee_site        : name of the end-effector site
        dt             : integration timestep [s]
        W              : (6,6) task-space weight; identity if None
        lam            : damping coefficient  λ
        alpha          : task-error gain      α       (drives J q̇ → −α e)
        q_ref          : (nv,) reference posture; disables posture task if None
        beta           : posture task weight  β       (0 disables it)
        k_post         : posture gain         k_post
        dq_max         : (nv,) symmetric joint-velocity bound; None disables
        use_jnt_range  : if True, read joint-position limits from model.jnt_range
        """
        self.model = model
        self.data  = data
        self.nv    = model.nv
        self.dt    = dt
        self.lam   = lam
        self.alpha = alpha

        # ---- end-effector site
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_id < 0:
            raise ValueError(f"Site '{ee_site}' not found in model")

        # ---- task-space weight
        self.W = np.eye(6) if W is None else np.asarray(W, float)
        if self.W.shape != (6, 6):
            raise ValueError("W must be (6, 6)")

        # ---- posture task (Type 1 soft)
        self.beta   = float(beta)
        self.k_post = float(k_post)
        self.q_ref  = None if q_ref is None else np.asarray(q_ref, float)
        if self.beta > 0 and self.q_ref is None:
            raise ValueError("q_ref must be provided when beta > 0")

        # ---- hard limits
        self.dq_max = None if dq_max is None else np.asarray(dq_max, float)
        if use_jnt_range:
            self.q_min = model.jnt_range[:, 0].copy()
            self.q_max = model.jnt_range[:, 1].copy()
            # unlimited joints in MuJoCo have jnt_range = [0, 0]; mask those off
            self.has_pos_limit = model.jnt_limited.astype(bool)
        else:
            self.q_min = None
            self.q_max = None
            self.has_pos_limit = np.zeros(self.nv, bool)

    # ---------------------------------------------------------- kinematics --

    def _site_jacobian(self, q_current: np.ndarray) -> np.ndarray:
        """Full 6xnv Jacobian of the EE site at q_current."""
        self.data.qpos[:self.nv] = q_current
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        return np.vstack([jacp, jacr])

    # ---------------------------------------------------------- QP assembly --

    def _build_qp(
        self,
        J:         np.ndarray,
        e:         np.ndarray,
        q_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (P, c, lb, ub) for the QP
            min (1/2) q̇ᵀ P q̇ + cᵀ q̇   s.t.   lb ≤ q̇ ≤ ub
        """
        nv = self.nv

        # ---- cost: primary task + damping
        P = J.T @ self.W @ J + (self.lam ** 2) * np.eye(nv)
        c = self.alpha * (J.T @ self.W @ e)

        # ---- cost: soft posture task (Type 1)
        #   (β/2) ‖q̇ + k (q − q_ref)‖²   ⇒  P += β I,  c += β k (q − q_ref)
        if self.beta > 0.0:
            e_post = q_current - self.q_ref
            P += self.beta * np.eye(nv)
            c += self.beta * self.k_post * e_post

        # ---- hard limits: start with ±∞ box, then tighten
        lb = np.full(nv, -np.inf)
        ub = np.full(nv,  np.inf)

        # velocity limits
        if self.dq_max is not None:
            lb = np.maximum(lb, -self.dq_max)
            ub = np.minimum(ub,  self.dq_max)

        # position limits → per-joint velocity bounds via forward Euler
        if np.any(self.has_pos_limit):
            lb_pos = (self.q_min - q_current) / self.dt
            ub_pos = (self.q_max - q_current) / self.dt
            mask = self.has_pos_limit
            lb[mask] = np.maximum(lb[mask], lb_pos[mask])
            ub[mask] = np.minimum(ub[mask], ub_pos[mask])

        return P, c, lb, ub

    # --------------------------------------------------------------- solve --

    def _solve(
        self,
        P: np.ndarray, c: np.ndarray,
        lb: np.ndarray, ub: np.ndarray,
    ) -> np.ndarray:
        """OSQP with box constraints on q̇ (A = I)."""
        n = P.shape[0]
        prob = osqp.OSQP()
        prob.setup(
            P=sp.csc_matrix(P),
            q=c,
            A=sp.csc_matrix(np.eye(n)),
            l=lb,
            u=ub,
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=4000,
            polish=True,
        )
        res = prob.solve()
        if res.info.status not in ("solved", "solved_inaccurate"):
            raise RuntimeError(f"OSQP failed: {res.info.status}")
        return res.x

    # ----------------------------------------------------------------- main --

    def execute(
        self,
        q_current: np.ndarray,
        e:         np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve one IK step.

        Parameters
        ----------
        q_current : (nv,) current joint positions
        e         : (6,)  task-space error  [Δpos (3); Δrot (3)]
                    Controller drives  J q̇  →  −α e.

        Returns
        -------
        q_next : (nv,) integrated joint positions  (q + q̇·dt)
        q_dot  : (nv,) optimal joint velocities
        """
        q_current = np.asarray(q_current, float)

        J = self._site_jacobian(q_current)
        P, c, lb, ub = self._build_qp(J, e, q_current)
        q_dot = self._solve(P, c, lb, ub)
        q_next = q_current + q_dot * self.dt
        return q_next, q_dot