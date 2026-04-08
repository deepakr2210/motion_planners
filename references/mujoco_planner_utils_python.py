"""
mujoco_planner_utils_python.py

Reference sheet: MuJoCo Python API utilities for custom motion planners.

This file IS importable — every snippet is a working example or helper
function you can copy into your planner.  Requires:
    pip install mujoco   (tested with mujoco >= 3.0)

Layout
------
  0.  Lifecycle
  1.  State management
  2.  Forward / inverse dynamics
  3.  Kinematics stages
  4.  Jacobians
  5.  Inertia / dynamics
  6.  Collision detection
  7.  Ray casting
  8.  Object velocity / acceleration
  9.  Applying external forces
  10. Position integration & differentiation
  11. Finite-difference derivatives
  12. Signed distance functions
  13. Quaternion utilities
  14. Pose utilities
  15. Vector / matrix utilities
  16. Linear algebra / decompositions
  17. Box-constrained QP solver
  18. Object lookup
  19. Energy
  20. Key mjData / mjModel fields quick reference

Conventions
-----------
  All arrays are numpy.ndarray with dtype=np.float64.
  Quaternions are [w, x, y, z].
  nq = m.nq   (position DOFs, may include quaternion components)
  nv = m.nv   (velocity / generalised-force DOFs)
  nu = m.nu   (control inputs)
  na = m.na   (actuator activations)
  nb = m.nbody
  nG = m.ngeom
  nS = m.nsite
"""

from __future__ import annotations
import numpy as np
import mujoco


# ============================================================
# 0. LIFECYCLE
# ============================================================

def load_model(xml_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a model from an MJCF / URDF file and allocate data."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    return m, d

def load_model_string(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a model from an MJCF XML string."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    return m, d

def copy_data(m: mujoco.MjModel, d: mujoco.MjData) -> mujoco.MjData:
    """Deep-copy mjData (essential for rollout branching in tree search)."""
    d_copy = mujoco.MjData(m)
    mujoco.mj_copyData(d_copy, m, d)
    return d_copy

def reset(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Reset data to model defaults (qpos = qpos0, everything else = 0)."""
    mujoco.mj_resetData(m, d)

def reset_to_keyframe(m: mujoco.MjModel, d: mujoco.MjData, key: int) -> None:
    """Reset to a named keyframe stored in the model (0-based index)."""
    mujoco.mj_resetDataKeyframe(m, d, key)


# ============================================================
# 1. STATE MANAGEMENT
# ============================================================
# mjtState flags (combine with |):
#   mujoco.mjtState.mjSTATE_QPOS   = 1 << 1
#   mujoco.mjtState.mjSTATE_QVEL   = 1 << 2
#   mujoco.mjtState.mjSTATE_ACT    = 1 << 3
#   mujoco.mjtState.mjSTATE_CTRL   = 1 << 6
#   mujoco.mjtState.mjSTATE_PHYSICS = qpos | qvel | act | history
#   mujoco.mjtState.mjSTATE_FULLPHYSICS = time | physics | plugin

def get_state(m: mujoco.MjModel, d: mujoco.MjData, sig: int) -> np.ndarray:
    """Serialize selected state components into a flat array."""
    n = mujoco.mj_stateSize(m, sig)
    state = np.zeros(n)
    mujoco.mj_getState(m, d, state, sig)
    return state

def set_state(m: mujoco.MjModel, d: mujoco.MjData,
              state: np.ndarray, sig: int) -> None:
    """Deserialize a flat state array back into mjData."""
    mujoco.mj_setState(m, d, state, sig)

def copy_physics_state(m: mujoco.MjModel,
                       src: mujoco.MjData,
                       dst: mujoco.MjData) -> None:
    """Copy qpos+qvel+act between two MjData objects."""
    sig = (mujoco.mjtState.mjSTATE_QPOS |
           mujoco.mjtState.mjSTATE_QVEL |
           mujoco.mjtState.mjSTATE_ACT)
    mujoco.mj_copyState(m, src, dst, sig)


# ============================================================
# 2. FORWARD / INVERSE DYNAMICS – full pipeline
# ============================================================

def forward(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Full forward dynamics (no time integration).
    Populates ALL position/velocity/acceleration arrays in d.
    After this call, d.xpos, d.xquat, d.site_xpos, d.qM,
    d.qfrc_bias, d.qacc, d.contact[...] are all valid.
    """
    mujoco.mj_forward(m, d)

def step(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Advance simulation by d->timestep (= m.opt.timestep).
    Calls mj_forward then integrates qpos/qvel.
    """
    mujoco.mj_step(m, d)

def step_with_ctrl(m: mujoco.MjModel, d: mujoco.MjData,
                   ctrl: np.ndarray) -> None:
    """Step simulation with explicit control (non-destructive)."""
    np.copyto(d.ctrl, ctrl)
    mujoco.mj_step(m, d)

def rollout(m: mujoco.MjModel, d0: mujoco.MjData,
            ctrl_seq: np.ndarray,   # (T, nu)
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out a control sequence from the current state.
    Returns (qpos_traj[T+1, nq], qvel_traj[T+1, nv]).
    Does NOT modify d0.
    """
    d = copy_data(m, d0)
    T = ctrl_seq.shape[0]
    qpos = np.zeros((T + 1, m.nq))
    qvel = np.zeros((T + 1, m.nv))
    qpos[0] = d.qpos.copy()
    qvel[0] = d.qvel.copy()
    for t in range(T):
        np.copyto(d.ctrl, ctrl_seq[t])
        mujoco.mj_step(m, d)
        qpos[t + 1] = d.qpos.copy()
        qvel[t + 1] = d.qvel.copy()
    return qpos, qvel

def inverse_dynamics(m: mujoco.MjModel, d: mujoco.MjData,
                     qacc: np.ndarray | None = None) -> np.ndarray:
    """
    Compute inverse dynamics.
    If qacc is provided it is written to d.qacc before the call.
    Returns d.qfrc_inverse [nv] — generalised torques needed for qacc.
    Requires: mj_kinematics + mj_fwdVelocity already called (or call
    mj_forwardSkip with mjSTAGE_VEL first).
    """
    if qacc is not None:
        np.copyto(d.qacc, qacc)
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse.copy()


# ============================================================
# 3. KINEMATICS STAGES
# ============================================================

def run_kinematics(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Run kinematics pass only (cheapest useful call).
    Populates: xpos, xquat, xmat, xipos, geom_xpos, site_xpos, cdof, cinert.
    Required before any Jacobian call.
    """
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)

def run_kinematics_and_velocity(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Kinematics + velocity stage.
    Populates everything in run_kinematics PLUS:
      cvel, cdof_dot, qfrc_bias (Coriolis + gravity).
    Required before velocity Jacobians, RNE, mj_objectVelocity.
    """
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)

def get_body_pose(d: mujoco.MjData, body_id: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (pos[3], quat[4]) of a body in world frame.
    (After mj_kinematics has been called.)
    """
    pos  = d.xpos[body_id].copy()
    quat = d.xquat[body_id].copy()
    return pos, quat

def get_site_pose(d: mujoco.MjData, site_id: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (pos[3], mat[9]) of a site in world frame.
    """
    pos = d.site_xpos[site_id].copy()
    mat = d.site_xmat[site_id].copy().reshape(3, 3)
    return pos, mat

def local_to_global(d: mujoco.MjData,
                    pos_local: np.ndarray, quat_local: np.ndarray,
                    body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a local body-frame pose to world frame.
    Returns (xpos[3], xmat[9]).
    """
    xpos = np.zeros(3)
    xmat = np.zeros(9)
    mujoco.mj_local2Global(d, xpos, xmat, pos_local, quat_local,
                            body_id, 0)
    return xpos, xmat


# ============================================================
# 4. JACOBIANS
# ============================================================
# All Jacobians: jacp [3, nv] (translational), jacr [3, nv] (rotational).
# Set either to None to skip computing that part.
# Must call run_kinematics() first.

def jacobian_body(m: mujoco.MjModel, d: mujoco.MjData,
                  body_id: int,
                  translation: bool = True,
                  rotation: bool = True
                  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Jacobian at body frame origin.
      Returns (jacp[3, nv], jacr[3, nv]).  Pass translation/rotation=False
      to skip and receive None for that component.
    """
    jacp = np.zeros((3, m.nv)) if translation else None
    jacr = np.zeros((3, m.nv)) if rotation    else None
    mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
    return jacp, jacr

def jacobian_body_com(m: mujoco.MjModel, d: mujoco.MjData,
                      body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian at body centre of mass. Returns (jacp[3,nv], jacr[3,nv])."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBodyCom(m, d, jacp, jacr, body_id)
    return jacp, jacr

def jacobian_site(m: mujoco.MjModel, d: mujoco.MjData,
                  site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian at site position. Returns (jacp[3,nv], jacr[3,nv])."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
    return jacp, jacr

def jacobian_geom(m: mujoco.MjModel, d: mujoco.MjData,
                  geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian at geom centre. Returns (jacp[3,nv], jacr[3,nv])."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacGeom(m, d, jacp, jacr, geom_id)
    return jacp, jacr

def jacobian_point(m: mujoco.MjModel, d: mujoco.MjData,
                   point: np.ndarray, body_id: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Jacobian of an arbitrary world-space point attached to body.
    Returns (jacp[3,nv], jacr[3,nv]).
    """
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, point, body_id)
    return jacp, jacr

def jacobian_subtree_com(m: mujoco.MjModel, d: mujoco.MjData,
                         body_id: int) -> np.ndarray:
    """
    Translational Jacobian of subtree CoM (rotation unavailable).
    Requires mj_subtreeVel() to have been called.
    Returns jacp[3, nv].
    """
    mujoco.mj_subtreeVel(m, d)
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jacp, body_id)
    return jacp

def jacobian_dot(m: mujoco.MjModel, d: mujoco.MjData,
                 point: np.ndarray, body_id: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Time derivative of Jacobian (Jdot) — needed for qacc from task acc.
    a_task = J * qacc + Jdot * qvel.
    Returns (jacp_dot[3,nv], jacr_dot[3,nv]).
    """
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacDot(m, d, jacp, jacr, point, body_id)
    return jacp, jacr

def full_jacobian_site(m: mujoco.MjModel, d: mujoco.MjData,
                       site_id: int) -> np.ndarray:
    """
    Convenience: stacked [jacp; jacr] → shape (6, nv).
    Row 0-2: translational, Row 3-5: rotational.
    """
    jacp, jacr = jacobian_site(m, d, site_id)
    return np.vstack([jacp, jacr])


# ============================================================
# 5. INERTIA / DYNAMICS
# ============================================================

def get_mass_matrix(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """
    Return the full (dense) nv×nv joint-space inertia matrix M(q).
    Prerequisites: mj_kinematics + mj_comPos (or mj_forward).
    Note: mj_fullM + mj_crb are called internally.
    """
    M_sparse = np.zeros(m.nM)
    mujoco.mj_fullM(m, M_sparse, d.qM)   # expand sparse → flat
    # mj_fullM writes into a flat array of length nv*nv
    M_dense = np.zeros(m.nv * m.nv)
    mujoco.mj_fullM(m, M_dense, d.qM)
    return M_dense.reshape(m.nv, m.nv)

def solve_M(m: mujoco.MjModel, d: mujoco.MjData,
            y: np.ndarray) -> np.ndarray:
    """
    Solve M * x = y using the pre-computed L'DL factorisation.
    y can be shape (nv,) or (nv, k) for k simultaneous solves.
    Prerequisites: mj_factorM must have been called (done by mj_forward).
    """
    y = np.atleast_2d(y.T).T   # ensure 2D column layout
    if y.ndim == 1:
        y2d = y[:, np.newaxis]
    else:
        y2d = y
    x = np.zeros_like(y2d)
    mujoco.mj_solveM(m, d, x, y2d, y2d.shape[1])
    return x.squeeze()

def mul_M(m: mujoco.MjModel, d: mujoco.MjData,
          vec: np.ndarray) -> np.ndarray:
    """Efficient sparse multiplication: res = M * vec. Shape (nv,)."""
    res = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, res, vec)
    return res

def get_bias(d: mujoco.MjData) -> np.ndarray:
    """
    Return C(q,v) = Coriolis + gravity = d.qfrc_bias [nv].
    Valid after mj_fwdVelocity (or mj_forward).
    """
    return d.qfrc_bias.copy()

def rne(m: mujoco.MjModel, d: mujoco.MjData,
        qacc: np.ndarray | None = None) -> np.ndarray:
    """
    Recursive Newton-Euler.
    If qacc is given → result = M*qacc + C(q,v).
    If qacc is None  → result = C(q,v)  (bias only).
    Returns result[nv].
    """
    result = np.zeros(m.nv)
    if qacc is not None:
        np.copyto(d.qacc, qacc)
        mujoco.mj_rne(m, d, 1, result)
    else:
        mujoco.mj_rne(m, d, 0, result)
    return result


# ============================================================
# 6. COLLISION DETECTION
# ============================================================

def run_collision(m: mujoco.MjModel, d: mujoco.MjData) -> list[dict]:
    """
    Run broadphase + narrowphase collision pipeline.
    Returns a list of contact dicts with keys:
      dist    — signed distance (negative = penetration)
      pos     — contact point in world frame [3]
      normal  — contact normal (from geom0 to geom1) [3]
      geom    — [geom_id_0, geom_id_1]
      dim     — constraint dimension (1/3/4/6)
      friction — [tangent1, tangent2, spin, roll1, roll2]
    """
    mujoco.mj_collision(m, d)
    contacts = []
    for i in range(d.ncon):
        c = d.contact[i]
        contacts.append({
            'dist':     c.dist,
            'pos':      c.pos.copy(),
            'normal':   c.frame[:3].copy(),   # frame rows: [normal, tan1, tan2]
            'geom':     list(c.geom),
            'dim':      c.dim,
            'friction': c.friction.copy(),
        })
    return contacts

def geom_distance(m: mujoco.MjModel, d: mujoco.MjData,
                  geom1: int, geom2: int,
                  distmax: float = 1.0
                  ) -> tuple[float, np.ndarray]:
    """
    Signed distance between two geoms (no full pipeline needed).
    Returns (dist, fromto[6]) where fromto = [p_on_geom1, p_on_geom2].
    dist < 0 means penetration.
    """
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(m, d, geom1, geom2, distmax, fromto)
    return dist, fromto

def contact_force(m: mujoco.MjModel, d: mujoco.MjData,
                  contact_id: int) -> np.ndarray:
    """
    6D force/torque in contact frame for a detected contact.
    contact_id: index into d.contact[].
    Returns result[6] = [fx, fy, fz, tx, ty, tz].
    """
    result = np.zeros(6)
    mujoco.mj_contactForce(m, d, contact_id, result)
    return result

def is_in_collision(m: mujoco.MjModel, d: mujoco.MjData) -> bool:
    """True if any contact has penetration (dist < 0) after mj_collision."""
    mujoco.mj_collision(m, d)
    for i in range(d.ncon):
        if d.contact[i].dist < 0:
            return True
    return False

def min_clearance(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """
    Minimum signed distance across all detected contacts.
    More negative = deeper penetration.
    Returns +inf if no contacts.
    """
    mujoco.mj_collision(m, d)
    if d.ncon == 0:
        return float('inf')
    return min(d.contact[i].dist for i in range(d.ncon))


# ============================================================
# 7. RAY CASTING
# ============================================================

def ray_cast(m: mujoco.MjModel, d: mujoco.MjData,
             origin: np.ndarray, direction: np.ndarray,
             body_exclude: int = -1,
             include_static: bool = True
             ) -> tuple[float, int, np.ndarray]:
    """
    Cast a ray and return distance, hit geom id, and surface normal.
    Returns (dist, geom_id, normal[3]).  dist = -1 if no hit.
    direction need not be unit-length.
    """
    geomid  = np.array([-1], dtype=np.int32)
    normal  = np.zeros(3)
    dist = mujoco.mj_ray(m, d, origin, direction,
                          None,              # geomgroup filter (None = all)
                          int(include_static),
                          body_exclude,
                          geomid, normal)
    return dist, int(geomid[0]), normal

def multi_ray_cast(m: mujoco.MjModel, d: mujoco.MjData,
                   origin: np.ndarray,
                   directions: np.ndarray,   # (N, 3)
                   cutoff: float = 1e6,
                   body_exclude: int = -1
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cast N rays from a single origin point.
    Returns (dist[N], geomid[N], normal[N, 3]).
    """
    n = directions.shape[0]
    vec_flat = directions.astype(np.float64).ravel()
    geomid  = np.full(n, -1, dtype=np.int32)
    dist    = np.full(n, -1.0)
    normal  = np.zeros((n, 3))
    mujoco.mj_multiRay(m, d, origin, vec_flat,
                        None, 1, body_exclude,
                        geomid, dist, normal.ravel(), n, cutoff)
    return dist, geomid, normal


# ============================================================
# 8. OBJECT VELOCITY / ACCELERATION
# ============================================================

def body_velocity(m: mujoco.MjModel, d: mujoco.MjData,
                  body_id: int,
                  local_frame: bool = False) -> np.ndarray:
    """
    6D velocity [angular(3); linear(3)] in body-centred frame.
    local_frame=True → local orientation, False → world orientation.
    """
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY,
                              body_id, res, int(local_frame))
    return res

def site_velocity(m: mujoco.MjModel, d: mujoco.MjData,
                  site_id: int,
                  local_frame: bool = False) -> np.ndarray:
    """6D velocity at a site. Returns [angular(3); linear(3)]."""
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE,
                              site_id, res, int(local_frame))
    return res

def body_acceleration(m: mujoco.MjModel, d: mujoco.MjData,
                      body_id: int,
                      local_frame: bool = False) -> np.ndarray:
    """6D acceleration [angular(3); linear(3)] of a body."""
    res = np.zeros(6)
    mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_BODY,
                                  body_id, res, int(local_frame))
    return res


# ============================================================
# 9. APPLYING EXTERNAL FORCES
# ============================================================

def apply_cartesian_force(m: mujoco.MjModel, d: mujoco.MjData,
                          force: np.ndarray | None,
                          torque: np.ndarray | None,
                          point: np.ndarray,
                          body_id: int,
                          target: np.ndarray | None = None) -> np.ndarray:
    """
    Apply Cartesian force/torque at a world-space point on body.
    Accumulates the equivalent generalised force into 'target'
    (defaults to d.qfrc_applied).
    Returns the modified qfrc_target[nv].
    """
    if target is None:
        target = d.qfrc_applied
    mujoco.mj_applyFT(m, d, force, torque, point, body_id, target)
    return target

def set_cartesian_wrench(d: mujoco.MjData,
                         body_id: int,
                         wrench: np.ndarray) -> None:
    """
    Set d.xfrc_applied[body_id] = wrench[6] = [fx, fy, fz, tx, ty, tz].
    This is applied directly as Cartesian force/torque each step.
    """
    d.xfrc_applied[body_id] = wrench


# ============================================================
# 10. POSITION INTEGRATION & DIFFERENTIATION
# ============================================================

def differentiate_pos(m: mujoco.MjModel,
                      qpos1: np.ndarray, qpos2: np.ndarray,
                      dt: float) -> np.ndarray:
    """
    Compute qvel = (qpos2 - qpos1) / dt in the tangent space.
    Handles quaternion joints correctly.
    Returns qvel[nv].
    """
    qvel = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, qvel, dt, qpos1, qpos2)
    return qvel

def integrate_pos(m: mujoco.MjModel,
                  qpos: np.ndarray, qvel: np.ndarray,
                  dt: float) -> np.ndarray:
    """
    Integrate: qpos_new = qpos ⊕ qvel*dt (quaternion-aware).
    Returns modified qpos (in-place); pass a copy to avoid side-effects.
    """
    qpos_new = qpos.copy()
    mujoco.mj_integratePos(m, qpos_new, qvel, dt)
    return qpos_new

def normalize_quat(m: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """
    Re-normalise all quaternion components in a qpos vector.
    Returns qpos (modified in-place).
    """
    mujoco.mj_normalizeQuat(m, qpos)
    return qpos


# ============================================================
# 11. FINITE-DIFFERENCE DERIVATIVES
# ============================================================

def transition_matrices(m: mujoco.MjModel, d: mujoco.MjData,
                        eps: float = 1e-6,
                        centered: bool = True,
                        compute_sensor: bool = False
                        ) -> dict[str, np.ndarray]:
    """
    Finite-differenced state-space matrices (control theory notation):
      dx_next = A*dx + B*du
      dsensor = C*dx + D*du
    State x = [qpos(nq); qvel(nv); act(na)], but linearised in
    tangent space of size (2*nv + na).

    Returns dict with keys 'A', 'B', and optionally 'C', 'D'.
    """
    nx = 2 * m.nv + m.na
    nu = m.nu
    ns = m.nsensordata

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    C = np.zeros((ns, nx)) if compute_sensor else None
    D = np.zeros((ns, nu)) if compute_sensor else None

    mujoco.mjd_transitionFD(m, d, eps, int(centered), A, B, C, D)
    result = {'A': A, 'B': B}
    if compute_sensor:
        result['C'] = C
        result['D'] = D
    return result

def inverse_dynamics_jacobians(m: mujoco.MjModel, d: mujoco.MjData,
                                eps: float = 1e-6
                                ) -> dict[str, np.ndarray]:
    """
    Finite-differenced Jacobians of inverse dynamics.
    Returns dict: DfDq, DfDv, DfDa  each [nv, nv].
    (Transposed from control-theory convention — columns are inputs.)
    """
    nv = m.nv
    DfDq = np.zeros((nv, nv))
    DfDv = np.zeros((nv, nv))
    DfDa = np.zeros((nv, nv))
    mujoco.mjd_inverseFD(m, d, eps, 1,
                          DfDq, DfDv, DfDa,
                          None, None, None, None)
    return {'DfDq': DfDq, 'DfDv': DfDv, 'DfDa': DfDa}


# ============================================================
# 12. SIGNED DISTANCE FUNCTIONS (SDF)
# ============================================================

def sdf_value(m: mujoco.MjModel, d: mujoco.MjData,
              geom_id: int, point: np.ndarray) -> float:
    """
    Evaluate the SDF for a geom at a world-space point.
    Returns signed distance (negative = inside).
    Note: geom must be backed by an SDF plugin.
    """
    s = mujoco.MjSDF()
    s.geom = geom_id
    return mujoco.mjc_distance(m, d, s, point)

def sdf_gradient(m: mujoco.MjModel, d: mujoco.MjData,
                 geom_id: int, point: np.ndarray) -> np.ndarray:
    """
    Gradient of the SDF at a world-space point (points outward).
    Returns gradient[3].
    """
    s = mujoco.MjSDF()
    s.geom = geom_id
    gradient = np.zeros(3)
    mujoco.mjc_gradient(m, d, s, gradient, point)
    return gradient


# ============================================================
# 13. QUATERNION UTILITIES
# ============================================================
# Convention: [w, x, y, z]

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product: res = q1 * q2."""
    res = np.zeros(4)
    mujoco.mju_mulQuat(res, q1, q2)
    return res

def quat_inv(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate / inverse."""
    res = np.zeros(4)
    mujoco.mju_negQuat(res, q)
    return res

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q: res = q * v * q^{-1}."""
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(res, v, q)
    return res

def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to quaternion."""
    res = np.zeros(4)
    mujoco.mju_axisAngle2Quat(res, axis, angle)
    return res

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, q)
    return res.reshape(3, 3)

def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion."""
    res = np.zeros(4)
    mujoco.mju_mat2Quat(res, R.ravel())
    return res

def euler_to_quat(euler: np.ndarray, seq: str = 'xyz') -> np.ndarray:
    """
    Euler angles (radians) to quaternion.
    seq: 3-char string from 'xyzXYZ'; lower=intrinsic, upper=extrinsic.
    """
    res = np.zeros(4)
    mujoco.mju_euler2Quat(res, euler, seq)
    return res

def quat_error(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """
    Orientation error as 3D angular velocity:
    qb * q(res) = qa  →  res is axis-angle to rotate from qb to qa.
    Useful for orientation tracking errors in control / IK.
    """
    res = np.zeros(3)
    mujoco.mju_subQuat(res, qa, qb)
    return res

def quat_integrate(q: np.ndarray, omega: np.ndarray,
                   dt: float) -> np.ndarray:
    """Integrate quaternion: q_new = q + omega * dt."""
    q_new = q.copy()
    mujoco.mju_quatIntegrate(q_new, omega, dt)
    mujoco.mju_normalize4(q_new)
    return q_new

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    t ∈ [0, 1]; t=0 returns q0, t=1 returns q1.
    """
    err = quat_error(q1, q0)           # axis-angle from q0 to q1
    return quat_integrate(q0, err, t)  # scale err by t


# ============================================================
# 14. POSE UTILITIES
# ============================================================

def pose_mul(pos1: np.ndarray, quat1: np.ndarray,
             pos2: np.ndarray, quat2: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Compose two poses: T = T1 * T2.  Returns (pos, quat)."""
    posres  = np.zeros(3)
    quatres = np.zeros(4)
    mujoco.mju_mulPose(posres, quatres, pos1, quat1, pos2, quat2)
    return posres, quatres

def pose_inv(pos: np.ndarray, quat: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Invert a pose: T^{-1}.  Returns (pos, quat)."""
    posres  = np.zeros(3)
    quatres = np.zeros(4)
    mujoco.mju_negPose(posres, quatres, pos, quat)
    return posres, quatres

def pose_transform_vec(pos: np.ndarray, quat: np.ndarray,
                       vec: np.ndarray) -> np.ndarray:
    """Transform vector by pose: res = pos + quat * vec."""
    res = np.zeros(3)
    mujoco.mju_trnVecPose(res, pos, quat, vec)
    return res


# ============================================================
# 15. VECTOR / MATRIX UTILITIES
# ============================================================

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors."""
    res = np.zeros(3)
    mujoco.mju_cross(res, a, b)
    return res

def transform_spatial(vec: np.ndarray, newpos: np.ndarray,
                      oldpos: np.ndarray,
                      rot: np.ndarray | None = None,
                      is_force: bool = False) -> np.ndarray:
    """
    Coordinate transform of a 6D motion or force vector.
    rot: 3×3 rotation matrix (None = identity).
    is_force: True for force/torque, False for velocity/acceleration.
    Returns res[6].
    """
    res = np.zeros(6)
    rot_flat = rot.ravel() if rot is not None else None
    mujoco.mju_transformSpatial(res, vec, int(is_force),
                                 newpos, oldpos, rot_flat)
    return res

def dense_to_sparse(mat: np.ndarray):
    """Convert dense nv×nv matrix to CSR sparse format."""
    nr, nc = mat.shape
    nnz = int(np.sum(mat != 0))
    res    = np.zeros(nnz)
    rownnz = np.zeros(nr, dtype=np.int32)
    rowadr = np.zeros(nr, dtype=np.int32)
    colind = np.zeros(nnz, dtype=np.int32)
    mujoco.mju_dense2sparse(res, mat.ravel(), nr, nc,
                             rownnz, rowadr, colind, nnz)
    return res, rownnz, rowadr, colind

def sparse_to_dense(res_sparse, rownnz, rowadr, colind,
                    nr: int, nc: int) -> np.ndarray:
    """Convert CSR sparse matrix to dense nr×nc array."""
    dense = np.zeros(nr * nc)
    mujoco.mju_sparse2dense(dense, res_sparse, nr, nc,
                             rownnz, rowadr, colind)
    return dense.reshape(nr, nc)


# ============================================================
# 16. LINEAR ALGEBRA – DECOMPOSITIONS
# ============================================================

def cholesky_factor(mat: np.ndarray,
                    min_diag: float = 0.0) -> tuple[np.ndarray, int]:
    """
    Cholesky factorisation of symmetric positive-definite matrix.
    Modifies mat in-place (lower triangle becomes L).
    Returns (L, rank).
    """
    L = mat.copy()
    rank = mujoco.mju_cholFactor(L.ravel(), L.shape[0], min_diag)
    return L, rank

def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L*L') * x = b given Cholesky factor L."""
    res = np.zeros_like(b)
    mujoco.mju_cholSolve(res, L.ravel(), b, b.shape[0])
    return res

def eig3(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a 3×3 symmetric matrix.
    Returns (eigval[3], eigvec[3×3], quat[4]) where
    mat = eigvec @ diag(eigval) @ eigvec'.
    """
    eigval = np.zeros(3)
    eigvec = np.zeros(9)
    quat   = np.zeros(4)
    mujoco.mju_eig3(eigval, eigvec, quat, mat.ravel())
    return eigval, eigvec.reshape(3, 3), quat


# ============================================================
# 17. BOX-CONSTRAINED QP SOLVER
# ============================================================

def box_qp(H: np.ndarray, g: np.ndarray,
           lower: np.ndarray | None = None,
           upper: np.ndarray | None = None,
           x0: np.ndarray | None = None
           ) -> np.ndarray | None:
    """
    Solve: min 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper.
    H must be symmetric positive-definite (nv×nv).
    Returns solution x[n] or None if solver failed.

    Use cases:
      - Constrained IK
      - Control Barrier Function (CBF) QP
      - Trajectory optimisation Newton step with joint limits
    """
    n = g.shape[0]
    res   = x0.copy() if x0 is not None else np.zeros(n)
    R     = np.zeros(n * (n + 7))
    index = np.zeros(n, dtype=np.int32)
    rank  = mujoco.mju_boxQP(res, R, index, H.ravel(), g, n, lower, upper)
    if rank < 0:
        return None
    return res


# ============================================================
# 18. OBJECT LOOKUP
# ============================================================

def body_id(m: mujoco.MjModel, name: str) -> int:
    """Return body id by name (-1 if not found)."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

def geom_id(m: mujoco.MjModel, name: str) -> int:
    """Return geom id by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

def site_id(m: mujoco.MjModel, name: str) -> int:
    """Return site id by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

def joint_id(m: mujoco.MjModel, name: str) -> int:
    """Return joint id by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

def actuator_id(m: mujoco.MjModel, name: str) -> int:
    """Return actuator id by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def sensor_id(m: mujoco.MjModel, name: str) -> int:
    """Return sensor id by name."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)


# ============================================================
# 19. ENERGY
# ============================================================

def compute_energy(m: mujoco.MjModel, d: mujoco.MjData
                   ) -> tuple[float, float]:
    """
    Compute potential and kinetic energy.
    Returns (E_pot, E_kin).
    Requires mj_fwdPosition and mj_fwdVelocity to have run.
    """
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    return float(d.energy[0]), float(d.energy[1])


# ============================================================
# 20. KEY mjData / mjModel FIELDS QUICK REFERENCE
# ============================================================

_MJDATA_FIELDS = """
STATE
  d.qpos  [nq]         joint positions (may include quaternion components)
  d.qvel  [nv]         joint velocities (tangent-space)
  d.qacc  [nv]         joint accelerations
  d.ctrl  [nu]         control inputs
  d.act   [na]         actuator activations

KINEMATICS  (populated after mj_kinematics + mj_comPos / mj_forward)
  d.xpos       [nb, 3]   body frame origins in world
  d.xquat      [nb, 4]   body frame quaternions [w,x,y,z]
  d.xmat       [nb, 9]   body frame rotation matrices (row-major)
  d.xipos      [nb, 3]   body CoM positions
  d.geom_xpos  [nG, 3]   geom positions
  d.geom_xmat  [nG, 9]   geom orientations
  d.site_xpos  [nS, 3]   site positions
  d.site_xmat  [nS, 9]   site orientations
  d.xanchor    [nJ, 3]   joint anchor positions in world
  d.xaxis      [nJ, 3]   joint axes in world
  d.subtree_com[nb, 3]   subtree CoM positions

INERTIA  (populated after mj_crb / mj_makeM / mj_forward)
  d.qM    [nM]        sparse inertia matrix (use mj_fullM to expand)
  d.qLD   [nM]        L'DL factorisation of M

DYNAMICS
  d.qfrc_bias      [nv]   C(q,v) = Coriolis + gravity
  d.qfrc_actuator  [nv]   actuator generalised forces
  d.qfrc_passive   [nv]   spring + damper + gravity-comp
  d.qfrc_applied   [nv]   user-set external generalised forces
  d.xfrc_applied   [nb,6] user-set Cartesian wrench per body
  d.qfrc_inverse   [nv]   result of mj_inverse

CONTACTS
  d.ncon               number of active contacts
  d.contact[i].dist    signed distance
  d.contact[i].pos     contact point (world)
  d.contact[i].frame   3×3 frame; row 0 = normal
  d.contact[i].geom    [geom_id_0, geom_id_1]
  d.contact[i].dim     constraint dimension (1/3/4/6)

CONSTRAINTS
  d.nefc         total constraint rows
  d.efc_J        constraint Jacobian (sparse, nefc×nv)
  d.efc_force    constraint forces [nefc]

ENERGY
  d.energy[0]    potential energy
  d.energy[1]    kinetic energy

SENSORS
  d.sensordata   [nsensordata]   all sensor outputs

MODEL SIZES  (m.*)
  m.nq, m.nv, m.na, m.nu   DOF counts
  m.nbody, m.ngeom, m.nsite, m.njnt
  m.nM   non-zeros in sparse M
  m.nsensordata  total sensor data length
  m.opt.timestep  simulation timestep
  m.opt.gravity   [3] gravity vector
"""

def print_mjdata_reference():
    """Print a quick-reference card for mjData fields."""
    print(_MJDATA_FIELDS)
