from __future__ import annotations

import mujoco
import numpy as np


class DiffIKControl:
    """
        model  = mujoco.MjModel.from_xml_path("robot.xml")
        data   = mujoco.MjData(model)
        ctrl   = DiffIKControl(model, data, ee_site="hand", dt=0.01)
        q_new  = ctrl.execute(q_current, v_twist)
    """

    def __init__(
        self,
        model:    mujoco.MjModel,
        data:     mujoco.MjData,
        ee_site:  str,
        dt:       float,
        lam:      float = 0.1,
    ):
        self.model  = model
        self.data   = data
        self.nv     = model.nv
        self.dt     = dt
        self.lam    = lam
        self.ee_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_id < 0:
            raise ValueError(f"Site '{ee_site}' not found in model")

    @staticmethod
    def vee(S: np.ndarray) -> np.ndarray:
        """Convert a 3x3 skew-symmetric matrix to a 3-vector."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def _update_kinematics(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _site_jacobian(self, site_id: int) -> np.ndarray:
        """Return the (6, nv) stacked Jacobian [Jp; Jr] for a site."""
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return np.vstack([jacp, jacr])

    def _damped_pinv(self, J: np.ndarray) -> np.ndarray:
        """Damped least-squares pseudo-inverse: J^T (J J^T + λ²I)^{-1}."""
        n = J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + self.lam**2 * np.eye(n))

    def execute(self, q_current: np.ndarray, v_in: np.ndarray) -> np.ndarray:
        """
        Compute next joint position for a desired EE twist.
        """
        ndof = q_current.shape[0]
        self.data.qpos[:ndof] = q_current
        self._update_kinematics()

        J          = self._site_jacobian(self.ee_id)
        q_dot      = self._damped_pinv(J) @ v_in   # (nv,)
        q = q_current + q_dot[:ndof] * self.dt
        return q, q_dot
