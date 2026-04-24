"""
QP Differential IK Control
===========================
Solves the weighted, damped differential-IK as a quadratic program:

    min  (1/2) || J q̇ + α e ||²_W  +  (λ²/2) || q̇ ||²₂
    q̇
    s.t. hard constraints : lb_h ≤ A_h q̇ ≤ ub_h
         soft constraints : lb_s ≤ A_s q̇ ≤ ub_s  (relaxed via slack, penalised)

Solvers: OSQP (default) or scipy SLSQP.

Soft-constraint formulation
---------------------------
Slack variables s_i (one per soft-constraint row) are appended to the
decision vector  z = [q̇; s].  The soft constraint rows become

    lb_s ≤ A_s q̇ − s ≤ ub_s      (always feasible)

and the cost gains a term  (w/2) ‖s‖²  so the solver trades off constraint
violation against the primary task cost.

Usage
-----
    model = mujoco.MjModel.from_xml_path("robot.xml")
    data  = mujoco.MjData(model)
    ctrl  = QPDiffIKControl(model, data, ee_site="hand", dt=0.01,
                             W=np.diag([1,1,1, 0.1,0.1,0.1]))

    # hard joint-velocity bound
    ctrl.add_hard_velocity_limits(-v_max * np.ones(nv), v_max * np.ones(nv))

    # hard joint-position bound  (applied each call using q_current)
    ctrl.add_hard_position_limits(model.jnt_range[:,0], model.jnt_range[:,1])

    # soft: e.g. keep a secondary joint near a preferred posture
    ctrl.add_soft_constraint(np.eye(nv), q_pref - 0.05, q_pref + 0.05, weight=5.0)

    q_next, q_dot = ctrl.execute(q_current, task_error)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

try:
    import osqp
    _OSQP_OK = True
except ImportError:
    _OSQP_OK = False


class QPDiffIKControl:

    def __init__(
        self,
        model:   mujoco.MjModel,
        data:    mujoco.MjData,
        ee_site: str,
        dt:      float,
        W:       Optional[np.ndarray] = None,
        lam:     float = 0.01,
        alpha:   float = 1.0,
        solver:  str = "osqp",
    ):
        """
        Parameters
        ----------
        model, data : MuJoCo model/data pair
        ee_site     : name of the end-effector site in the XML
        dt          : integration timestep
        W           : (6,6) task-space weight matrix; identity if None
        lam         : damping coefficient λ
        alpha       : task-error gain α
        solver      : "osqp" (default) or "scipy"
        """
        self.model = model
        self.data  = data
        self.nv    = model.nv
        self.dt    = dt
        self.lam   = lam
        self.alpha = alpha

        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_id < 0:
            raise ValueError(f"Site '{ee_site}' not found in model")

        if W is None:
            self.W = np.eye(6)
        else:
            self.W = np.asarray(W, dtype=float)
            if self.W.shape != (6, 6):
                raise ValueError("W must be (6, 6)")

        self.solver = "scipy" if (solver == "osqp" and not _OSQP_OK) else solver

        # constraint registries
        self._hard:       List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._soft:       List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._pos_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -------------------------------------------------------- API ---------------------

    def add_hard_constraint(
        self,
        A:  np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> None:
        """Register  lb ≤ A q̇ ≤ ub  as a hard constraint."""
        A = np.asarray(A, float)
        if A.shape[1] > self.nv:
            raise ValueError(f"A has {A.shape[1]} columns but model.nv={self.nv}")
        self._hard.append((A, np.asarray(lb, float), np.asarray(ub, float)))

    def add_hard_velocity_limits(
        self,
        dq_min: np.ndarray,
        dq_max: np.ndarray,
    ) -> None:
        """Enforce per-joint velocity bounds (hard)."""
        dq_min = np.asarray(dq_min, float)
        dq_max = np.asarray(dq_max, float)
        self.add_hard_constraint(np.eye(len(dq_min)), dq_min, dq_max)

    def add_hard_position_limits(
        self,
        q_min: np.ndarray,
        q_max: np.ndarray,
    ) -> None:
        """
        Enforce joint-position bounds (hard).  Converted per-call to velocity
        limits:  (q_min − q) / dt  ≤  q̇  ≤  (q_max − q) / dt.
        """
        self._pos_limits = (np.asarray(q_min, float), np.asarray(q_max, float))

    def add_soft_constraint(
        self,
        A:      np.ndarray,
        lb:     np.ndarray,
        ub:     np.ndarray,
        weight: float = 1.0,
    ) -> None:
        """
        Register  lb ≤ A q̇ ≤ ub  as a soft constraint.
        Violation is penalised by  (weight/2) ‖s‖²  where s is the slack.
        """
        A = np.asarray(A, float)
        if A.shape[1] > self.nv:
            raise ValueError(f"A has {A.shape[1]} columns but model.nv={self.nv}")
        self._soft.append((A, np.asarray(lb, float), np.asarray(ub, float), float(weight)))

    # ------------------------------------------------------- kinematics ------------------

    def _update_kinematics(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _site_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        return np.vstack([jacp, jacr])  # (6, nv)

    # ---------------------------------------------------- QP assembly --------------------

    def _build_qp(
        self,
        J:         np.ndarray,
        e:         np.ndarray,
        q_current: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the augmented QP matrices.

        Decision vector:  z = [ q̇ (nv) | s_1 ... s_k (slack) ]

        Returns
        -------
        P       : (n_z, n_z)  positive-definite cost Hessian
        c       : (n_z,)      linear cost vector
        A_con   : (n_c, n_z)  constraint matrix
        lb_con  : (n_c,)
        ub_con  : (n_c,)
        """
        nv = self.nv
        n_slack = sum(int(A.shape[0]) for A, *_ in self._soft)
        n_vars  = nv + n_slack

        # ---- cost --------------------------------------------------------
        P = np.zeros((n_vars, n_vars))
        # primary task + damping block
        P[:nv, :nv] = J.T @ self.W @ J + self.lam ** 2 * np.eye(nv)
        # slack penalty blocks (one per soft-constraint group)
        off = nv
        for (A_s, _, _, w) in self._soft:
            ns = A_s.shape[0]
            P[off:off + ns, off:off + ns] = w * np.eye(ns)
            off += ns

        c = np.zeros(n_vars)
        c[:nv] = self.alpha * (J.T @ self.W @ e)

        # ---- constraints -------------------------------------------------
        A_rows, lb_rows, ub_rows = [], [], []

        def _push(row_block, lb_block, ub_block):
            A_rows.append(row_block)
            lb_rows.append(lb_block)
            ub_rows.append(ub_block)

        # --- hard constraints
        for (A_h, lb_h, ub_h) in self._hard:
            nr = A_h.shape[0]
            row = np.zeros((nr, n_vars))
            row[:, :A_h.shape[1]] = A_h
            _push(row, lb_h, ub_h)

        # --- position-limit hard constraint (per call)
        if self._pos_limits is not None:
            q_min, q_max = self._pos_limits
            nq = len(q_min)
            qc = q_current[:nq]
            row = np.zeros((nq, n_vars))
            row[:, :nq] = np.eye(nq)
            _push(row, (q_min - qc) / self.dt, (q_max - qc) / self.dt)

        # --- soft constraints:  A_s q̇ − s ∈ [lb_s, ub_s]
        off = nv
        for (A_s, lb_s, ub_s, _) in self._soft:
            ns = A_s.shape[0]
            row = np.zeros((ns, n_vars))
            row[:, :A_s.shape[1]] = A_s
            row[:, off:off + ns]  = -np.eye(ns)
            _push(row, lb_s, ub_s)
            off += ns

        if A_rows:
            A_con  = np.vstack(A_rows)
            lb_con = np.concatenate(lb_rows)
            ub_con = np.concatenate(ub_rows)
        else:
            A_con  = np.zeros((0, n_vars))
            lb_con = np.zeros(0)
            ub_con = np.zeros(0)

        return P, c, A_con, lb_con, ub_con

    # ---------------------------------------------------- solvers ------------

    def _solve_osqp(
        self,
        P: np.ndarray, c: np.ndarray,
        A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    ) -> np.ndarray:
        n = P.shape[0]
        P_sp = sp.csc_matrix(P)
        A_sp = sp.csc_matrix(A) if A.shape[0] > 0 else sp.csc_matrix((0, n))

        prob = osqp.OSQP()
        prob.setup(
            P_sp, c, A_sp, lb, ub,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=8000,
            polish=True,
        )
        res = prob.solve()
        status = res.info.status
        if status not in ("solved", "solved_inaccurate"):
            raise RuntimeError(f"OSQP: {status}")
        return res.x

    def _solve_scipy(
        self,
        P: np.ndarray, c: np.ndarray,
        A: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    ) -> np.ndarray:
        x0 = np.zeros(P.shape[0])

        constraints = []
        if A.shape[0] > 0:
            # lb ≤ Ax ≤ ub  →  two "ineq" (≥ 0) constraints
            constraints = [
                {"type": "ineq", "fun": lambda x: A @ x - lb, "jac": lambda x: A},
                {"type": "ineq", "fun": lambda x: ub - A @ x, "jac": lambda x: -A},
            ]

        res = minimize(
            lambda x: (0.5 * x @ P @ x + c @ x, P @ x + c),
            x0,
            jac=True,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"scipy SLSQP: {res.message}")
        return res.x

    # ----------------------------------------------------------------- main --

    def execute(
        self,
        q_current: np.ndarray,
        e:         np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the QP and integrate one step.

        Parameters
        ----------
        q_current : (ndof,) current joint positions
        e         : (6,)   task-space error  [Δpos (3), Δrot (3)]
                    The controller drives  J q̇  →  −α e.

        Returns
        -------
        q_next : (ndof,) integrated joint positions
        q_dot  : (ndof,) optimal joint velocities
        """
        ndof = q_current.shape[0]
        self.data.qpos[:ndof] = q_current
        self._update_kinematics()

        J = self._site_jacobian()                                     # (6, nv)
        P, c, A_con, lb_con, ub_con = self._build_qp(J, e, q_current)

        if self.solver == "osqp":
            z = self._solve_osqp(P, c, A_con, lb_con, ub_con)
        else:
            z = self._solve_scipy(P, c, A_con, lb_con, ub_con)

        q_dot  = z[:ndof]
        q_next = q_current + q_dot * self.dt
        return q_next, q_dot
