"""
Python collision checker for sphere obstacles.

Two query interfaces
--------------------
  is_point_in_collision(p)          3-D workspace point vs. all spheres   O(n)
  is_config_in_collision(q)         robot joint config vs. all spheres
    approximate=True  (default)     FK → check body-frame origins + bounding radii  fast
    approximate=False               full MuJoCo mj_forward + contact count           exact

Example
-------
  import mujoco
  from obstacles import CollisionChecker

  m = mujoco.MjModel.from_xml_path("assets/mujoco_menagerie/franka_emika_panda/scene_with_obstacles.xml")
  d = mujoco.MjData(m)

  cc = CollisionChecker.from_yaml("obstacles/data.yaml", model=m, data=d)

  print(cc.is_point_in_collision([0.3, 0.2, 0.5]))
  print(cc.is_config_in_collision([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Approximate per-link bounding radii (Franka Panda, body IDs 1-8) ─────────
# Conservative — intentionally slightly larger than the true geometry.
_LINK_BOUNDING_RADII = np.array([0.10, 0.09, 0.09, 0.09, 0.07, 0.07, 0.06, 0.06])


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray   # shape (3,)
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))

    def contains(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point - self.center)) < self.radius


class CollisionChecker:
    """
    Geometric collision checker for a scene populated with sphere obstacles.

    Parameters
    ----------
    spheres  : list of Sphere
    model    : mujoco.MjModel  (required for is_config_in_collision)
    data     : mujoco.MjData   (required for is_config_in_collision)
    ndof     : number of controlled joints (default 7 for Panda)
    """

    def __init__(
        self,
        spheres: list[Sphere],
        model=None,
        data=None,
        ndof: int = 7,
    ):
        self._spheres = spheres
        self._model   = model
        self._data    = data
        self._ndof    = ndof

        # Pre-compute centres and radii as arrays for vectorised checks
        if spheres:
            self._centers = np.stack([s.center for s in spheres])  # (N, 3)
            self._radii   = np.array([s.radius for s in spheres])  # (N,)
        else:
            self._centers = np.empty((0, 3))
            self._radii   = np.empty((0,))

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path, model=None, data=None, ndof: int = 7) -> "CollisionChecker":
        """Load obstacles from the YAML file written by obstacles/generator.py."""
        with open(path) as f:
            doc = yaml.safe_load(f)
        spheres = [
            Sphere(center=np.array(s["center"]), radius=float(s["radius"]))
            for s in doc["obstacles"]
        ]
        return cls(spheres, model=model, data=data, ndof=ndof)

    @classmethod
    def from_list(cls, centers: list, radii: list | float,
                  model=None, data=None, ndof: int = 7) -> "CollisionChecker":
        """Convenience constructor from plain lists."""
        if isinstance(radii, (int, float)):
            radii = [radii] * len(centers)
        spheres = [Sphere(center=np.array(c), radius=float(r))
                   for c, r in zip(centers, radii)]
        return cls(spheres, model=model, data=data, ndof=ndof)

    # ── Workspace point query ─────────────────────────────────────────────────

    def is_point_in_collision(self, point: Sequence[float]) -> bool:
        """
        Return True if the 3-D workspace point is inside any obstacle sphere.
        No MuJoCo required.
        """
        if len(self._spheres) == 0:
            return False
        p = np.asarray(point, dtype=float)
        dists = np.linalg.norm(self._centers - p, axis=1)
        return bool(np.any(dists < self._radii))

    def point_min_clearance(self, point: Sequence[float]) -> float:
        """
        Signed distance from *point* to the nearest obstacle surface.
        Negative means the point is inside an obstacle.
        Returns +inf if there are no obstacles.
        """
        if len(self._spheres) == 0:
            return float("inf")
        p = np.asarray(point, dtype=float)
        dists = np.linalg.norm(self._centers - p, axis=1) - self._radii
        return float(np.min(dists))

    # ── Joint-space config query ──────────────────────────────────────────────

    def is_config_in_collision(
        self,
        q: Sequence[float],
        approximate: bool = True,
    ) -> bool:
        """
        Return True if the robot at joint configuration *q* is in collision.

        Parameters
        ----------
        q            : joint angles [rad], length = ndof
        approximate  : True  → FK + per-link bounding spheres (fast, conservative)
                       False → full mj_forward + MuJoCo contact count (exact, slower)

        Raises
        ------
        RuntimeError if model/data were not provided at construction.
        """
        self._require_mujoco()
        if approximate:
            return self._check_approx(np.asarray(q, dtype=float))
        else:
            return self._check_exact(np.asarray(q, dtype=float))

    def config_min_clearance(self, q: Sequence[float]) -> float:
        """
        Approximate minimum clearance between any robot link and any obstacle [m].
        Negative means a link bounding sphere overlaps an obstacle.
        """
        self._require_mujoco()
        import mujoco
        q = np.asarray(q, dtype=float)
        m, d = self._model, self._data

        saved_qpos = d.qpos.copy()
        saved_qvel = d.qvel.copy()

        d.qpos[:self._ndof] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)

        min_clearance = float("inf")
        n_bodies = min(m.nbody, len(_LINK_BOUNDING_RADII) + 1)
        for b in range(1, n_bodies):
            bp = d.xpos[b]
            lr = _LINK_BOUNDING_RADII[b - 1]
            dists = np.linalg.norm(self._centers - bp, axis=1) - self._radii - lr
            if len(dists):
                min_clearance = min(min_clearance, float(np.min(dists)))

        d.qpos[:] = saved_qpos
        d.qvel[:] = saved_qvel
        mujoco.mj_kinematics(m, d)

        return min_clearance

    # ── Batch queries (for use in planners / RRT / trajopt) ──────────────────

    def are_configs_in_collision(
        self, configs: np.ndarray, approximate: bool = True
    ) -> np.ndarray:
        """
        Vectorised config check over a (K, ndof) array.
        Returns a boolean array of shape (K,).
        """
        return np.array(
            [self.is_config_in_collision(q, approximate=approximate)
             for q in configs],
            dtype=bool,
        )

    def filter_free_configs(
        self, configs: np.ndarray, approximate: bool = True
    ) -> np.ndarray:
        """Return only the collision-free rows from a (K, ndof) config array."""
        mask = ~self.are_configs_in_collision(configs, approximate=approximate)
        return configs[mask]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def spheres(self) -> list[Sphere]:
        return self._spheres

    @property
    def num_obstacles(self) -> int:
        return len(self._spheres)

    def __repr__(self) -> str:
        return f"CollisionChecker(n_obstacles={self.num_obstacles})"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_mujoco(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                "CollisionChecker: model and data must be provided for "
                "joint-space collision queries."
            )

    def _check_approx(self, q: np.ndarray) -> bool:
        """FK + bounding-sphere geometric check (fast)."""
        import mujoco
        m, d = self._model, self._data

        saved_qpos = d.qpos.copy()
        saved_qvel = d.qvel.copy()

        d.qpos[:self._ndof] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)

        hit = False
        n_bodies = min(m.nbody, len(_LINK_BOUNDING_RADII) + 1)
        for b in range(1, n_bodies):
            bp = d.xpos[b]
            lr = _LINK_BOUNDING_RADII[b - 1]
            dists = np.linalg.norm(self._centers - bp, axis=1)
            if np.any(dists < self._radii + lr):
                hit = True
                break

        d.qpos[:] = saved_qpos
        d.qvel[:] = saved_qvel
        mujoco.mj_kinematics(m, d)

        return hit

    def _check_exact(self, q: np.ndarray) -> bool:
        """Full MuJoCo forward pass + contact count (exact)."""
        import mujoco
        m, d = self._model, self._data

        saved_qpos = d.qpos.copy()
        saved_qvel = d.qvel.copy()

        d.qpos[:self._ndof] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        hit = d.ncon > 0

        d.qpos[:] = saved_qpos
        d.qvel[:] = saved_qvel
        mujoco.mj_forward(m, d)

        return hit
