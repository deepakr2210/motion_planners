# MuJoCo as a Rigid-Body Oracle for Custom Planners

This directory contains two self-contained reference files that expose
every useful MuJoCo utility for building motion planners (RRT/RRT*,
iLQR, MPC, trajectory optimisation, IK, CBF-QP, etc.):

| File | Language | Purpose |
|---|---|---|
| [mujoco_planner_utils_python.py](mujoco_planner_utils_python.py) | Python | Working helper functions wrapping the Python bindings |
| [mujoco_planner_utils_cpp.h](mujoco_planner_utils_cpp.h) | C++ | Annotated header with every relevant C API signature |

---

## Architecture: MuJoCo as an Oracle

Your planner never needs to implement physics. Instead it calls MuJoCo
as a black-box oracle that answers queries like:

```
Given configuration q (and optionally velocity v, control u):
  → Where are all bodies / sites / geoms?   (kinematics)
  → What are the Jacobians?                 (differential kinematics)
  → What is the mass matrix / bias force?   (dynamics)
  → Are any bodies in collision?            (collision detection)
  → What is the signed distance to X?       (proximity)
  → Where does this ray hit?                (sensor / occupancy)
  → What joint torques produce acceleration a?  (inverse dynamics)
  → What is the next state under control u?     (forward simulation)
  → What are ∂f/∂q, ∂f/∂v (linearised dynamics)?  (derivatives)
```

The two core data structures:

```
mjModel* m   — compiled model, READ-ONLY after load
               stores all geometry, inertia, joint limits, …
mjData*  d   — simulation state workspace, READ-WRITE
               stores qpos, qvel, qacc, contacts, Jacobians, …
```

In Python: `mujoco.MjModel`, `mujoco.MjData`.

---

## Minimal Setup

### Python
```python
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("robot.xml")
d = mujoco.MjData(m)

# Set joint configuration
d.qpos[:] = q

# Run forward kinematics only (cheaper than full forward dynamics)
mujoco.mj_kinematics(m, d)
mujoco.mj_comPos(m, d)          # required before Jacobian calls

# Get end-effector site position
ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
print(d.site_xpos[ee_id])       # shape (3,)
```

### C++
```cpp
#include <mujoco/mujoco.h>

char error[1000];
mjModel* m = mj_loadXML("robot.xml", nullptr, error, sizeof(error));
mjData*  d = mj_makeData(m);

// Set configuration
mju_copy(d->qpos, q, m->nq);

// Forward kinematics
mj_kinematics(m, d);
mj_comPos(m, d);

// End-effector position
int ee_id = mj_name2id(m, mjOBJ_SITE, "ee_site");
mjtNum* ee_pos = d->site_xpos + 3 * ee_id;   // pointer into (nsite x 3) array
```

---

## Category Guide

### 1. Kinematics

The cheapest oracle call. Only needs `qpos`; gives body/geom/site poses.

**Call order for position-only queries:**
```
mj_kinematics(m, d)   →  xpos, xquat, xmat, geom_xpos, site_xpos
mj_comPos(m, d)        →  cdof, cinert, subtree_com   ← needed for Jacobians
```

**Call order for velocity queries:**
```
(above) + mj_comVel(m, d)  →  cvel, cdof_dot, qfrc_bias
```

**Key outputs in mjData:**

| Field | Shape | Content |
|---|---|---|
| `d.xpos` | (nbody, 3) | Body frame origins in world |
| `d.xquat` | (nbody, 4) | Body quaternions [w,x,y,z] |
| `d.xmat` | (nbody, 9) | Body rotation matrices (row-major) |
| `d.xipos` | (nbody, 3) | Body CoM positions |
| `d.geom_xpos` | (ngeom, 3) | Geom positions |
| `d.site_xpos` | (nsite, 3) | Site positions |
| `d.site_xmat` | (nsite, 9) | Site orientations |
| `d.xanchor` | (njnt, 3) | Joint anchor positions |
| `d.xaxis` | (njnt, 3) | Joint axes |

---

### 2. Jacobians

All Jacobians are **3 × nv**. `jacp` = translational, `jacr` = rotational.
Either can be `NULL` / `None` to skip. Requires `mj_kinematics + mj_comPos`.

| Function | Point |
|---|---|
| `mj_jac(m, d, jacp, jacr, point, body)` | Arbitrary world-space point on body |
| `mj_jacBody` | Body frame origin |
| `mj_jacBodyCom` | Body CoM |
| `mj_jacSite` | Site position |
| `mj_jacGeom` | Geom centre |
| `mj_jacSubtreeCom` | Subtree CoM (translation only) |
| `mj_jacDot` | Time derivative Jdot (for full task-space control) |

**Full 6-DOF Jacobian (Python):**
```python
jacp = np.zeros((3, m.nv))
jacr = np.zeros((3, m.nv))
mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
J = np.vstack([jacp, jacr])     # (6, nv)
```

**Pseudo-inverse IK step:**
```python
dx = target_pos - d.site_xpos[ee_id]           # task-space error
J, _ = jacobian_site(m, d, ee_id)              # (3, nv)
dq = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(3), dx)
d.qpos += dq
```

---

### 3. Dynamics

**Mass matrix M(q):**
```python
# After mj_forward or mj_crb + mj_makeM + mj_factorM
M_dense = np.zeros(m.nv * m.nv)
mujoco.mj_fullM(m, M_dense, d.qM)
M = M_dense.reshape(m.nv, m.nv)

# Efficient M*v (sparse, never form the dense matrix):
res = np.zeros(m.nv)
mujoco.mj_mulM(m, d, res, v)

# Solve M*x = y (uses L'DL, O(nv^2)):
mujoco.mj_solveM(m, d, x, y, 1)
```

**Bias / Coriolis + gravity C(q,v):**
```python
mujoco.mj_forward(m, d)
bias = d.qfrc_bias.copy()   # nv
```

**Inverse dynamics** (given desired `qacc`, returns required torques):
```python
d.qacc[:] = desired_qacc
mujoco.mj_inverse(m, d)
tau = d.qfrc_inverse.copy()
```

**Key dynamics fields:**

| Field | Shape | Content |
|---|---|---|
| `d.qM` | (nM,) | Sparse inertia matrix |
| `d.qLD` | (nM,) | L'DL factorisation |
| `d.qfrc_bias` | (nv,) | C(q,v): Coriolis + gravity |
| `d.qfrc_actuator` | (nv,) | Actuator generalised forces |
| `d.qfrc_passive` | (nv,) | Spring + damper + grav-comp |
| `d.qacc` | (nv,) | Joint accelerations |
| `d.qfrc_inverse` | (nv,) | Inverse dynamics result |

---

### 4. Collision Detection

**Full pipeline:**
```python
mujoco.mj_collision(m, d)       # fills d.contact[0..d.ncon-1]
for i in range(d.ncon):
    c = d.contact[i]
    print(c.dist,          # signed distance (neg = penetration)
          c.pos,           # contact point [3]
          c.frame[:3],     # surface normal [3]  (geom[0] → geom[1])
          c.geom)          # [geom_id_0, geom_id_1]
```

**Targeted pairwise query** (no full pipeline):
```python
fromto = np.zeros(6)
dist = mujoco.mj_geomDistance(m, d, geom1_id, geom2_id,
                               distmax=0.5, fromto=fromto)
# fromto[0:3] = nearest point on geom1
# fromto[3:6] = nearest point on geom2
```

**mjContact struct fields:**

| Field | Type | Description |
|---|---|---|
| `dist` | float | Signed distance; negative = penetration |
| `pos[3]` | float[3] | Midpoint between geom surfaces |
| `frame[9]` | float[9] | Contact frame; `frame[0:3]` = outward normal |
| `geom[2]` | int[2] | Geom ids |
| `dim` | int | Constraint dimension (1/3/4/6) |
| `friction[5]` | float[5] | tan1, tan2, spin, roll1, roll2 |
| `efc_address` | int | Row in `efc_J` (-1 if excluded) |

---

### 5. Ray Casting

```python
origin    = np.array([0, 0, 1.0])
direction = np.array([0, 0, -1.0])
geomid    = np.array([-1], dtype=np.int32)
normal    = np.zeros(3)

dist = mujoco.mj_ray(m, d, origin, direction,
                      None,    # geomgroup filter
                      1,       # include static geoms
                      -1,      # body_exclude
                      geomid, normal)
# dist = -1 if no hit; otherwise distance along ray
```

**Batch rays from a single point** (e.g., depth sensor / occupancy):
```python
mujoco.mj_multiRay(m, d, origin, vec_flat,
                    None, 1, -1,
                    geomid, dist, normal_flat, nray, cutoff)
```

---

### 6. Finite-Difference Derivatives

These are the **most powerful** functions for gradient-based planners.

**State-transition matrices (for LQR / iLQR / MPC):**
```python
# State x = [qpos(nq), qvel(nv), act(na)], linearised as size 2*nv+na
# d(x_next) = A*dx + B*du
A = np.zeros((2*m.nv + m.na, 2*m.nv + m.na))
B = np.zeros((2*m.nv + m.na, m.nu))
mujoco.mjd_transitionFD(m, d, eps=1e-6, flg_centered=1,
                          A=A, B=B, C=None, D=None)
```

**Inverse dynamics Jacobians** (for trajectory optimisation):
```python
DfDq = np.zeros((m.nv, m.nv))
DfDv = np.zeros((m.nv, m.nv))
DfDa = np.zeros((m.nv, m.nv))
mujoco.mjd_inverseFD(m, d, 1e-6, 1,
                      DfDq, DfDv, DfDa,
                      None, None, None, None)
```

---

### 7. Quaternion Utilities

All use the `[w, x, y, z]` convention.

| Function | Purpose |
|---|---|
| `mju_mulQuat(res, q1, q2)` | q1 * q2 |
| `mju_negQuat(res, q)` | Conjugate / inverse |
| `mju_rotVecQuat(res, v, q)` | Rotate vector |
| `mju_quat2Mat(res, q)` | → 3×3 matrix |
| `mju_mat2Quat(q, mat)` | ← 3×3 matrix |
| `mju_subQuat(res, qa, qb)` | Orientation error (axis-angle) |
| `mju_quatIntegrate(q, omega, dt)` | Integrate angular velocity |
| `mju_euler2Quat(q, euler, seq)` | Euler → quaternion |
| `mju_axisAngle2Quat(q, axis, angle)` | Axis-angle → quaternion |

**Tracking error for controllers:**
```python
# error = axis-angle to rotate from q_current to q_target
err = np.zeros(3)
mujoco.mju_subQuat(err, q_target, q_current)
```

---

### 8. Box-Constrained QP

```
min  0.5 * x' * H * x + x' * g
s.t. lower <= x <= upper
```

MuJoCo ships an active-set QP solver for this. Useful for:
- Constrained IK (joint limits)
- Control Barrier Function (CBF) QPs
- Trajectory optimisation step with bounds

```python
n     = m.nv
res   = np.zeros(n)           # warm-start
R     = np.zeros(n * (n+7))   # scratch space
index = np.zeros(n, dtype=np.int32)

rank = mujoco.mju_boxQP(res, R, index,
                         H.ravel(), g, n,
                         lower, upper)
# rank < 0  → failed; otherwise res contains the solution
```

---

### 9. Rollout / State Saving Pattern

For sampling-based planners (RRT, MPPI, CEM), never copy the full
mjModel — only copy mjData:

```python
# Save a node
d_saved = mujoco.MjData(m)
mujoco.mj_copyData(d_saved, m, d)     # full copy

# Restore
mujoco.mj_copyData(d, m, d_saved)

# Minimal copy (physics state only, faster)
sig = (mujoco.mjtState.mjSTATE_QPOS |
       mujoco.mjtState.mjSTATE_QVEL |
       mujoco.mjtState.mjSTATE_ACT)
mujoco.mj_copyState(m, src, dst, sig)
```

---

### 10. Selective Forward Pipeline (Performance)

Avoid rerunning expensive stages when only part of the state changed:

| What changed | Minimum call |
|---|---|
| qpos only | `mj_kinematics + mj_comPos` |
| qpos + qvel | add `mj_comVel` |
| qpos + qvel + ctrl | `mj_forwardSkip(m, d, mjSTAGE_NONE, 1)` |
| qvel only (qpos unchanged) | `mj_forwardSkip(m, d, mjSTAGE_POS, 1)` |
| Nothing (e.g., re-check sensor) | `mj_forwardSkip(m, d, mjSTAGE_VEL, 1)` |

In Python:
```python
import mujoco
mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_POS, 1)
```

---

### 11. External Forces

```python
# Method 1: generalised force (nv-space)
d.qfrc_applied[joint_id] += torque

# Method 2: Cartesian force/torque on a body (set every step)
d.xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]

# Method 3: applyFT (maps Cartesian force at world point → qfrc_applied)
mujoco.mj_applyFT(m, d, force, torque, point, body_id, d.qfrc_applied)
```

---

## Common Planner Recipes

### Numerical IK (Jacobian Pseudo-Inverse)
```python
def ik_step(m, d, site_id, target_pos, dt=0.01, alpha=0.5):
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, site_id)
    err  = target_pos - d.site_xpos[site_id]
    dq   = alpha * jacp.T @ np.linalg.solve(jacp @ jacp.T + 1e-4*np.eye(3), err)
    mujoco.mj_integratePos(m, d.qpos, dq, dt)
    mujoco.mj_normalizeQuat(m, d.qpos)
```

### iLQR / DDP Backward Pass (get A, B)
```python
A = np.zeros((2*m.nv, 2*m.nv))
B = np.zeros((2*m.nv, m.nu))
mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
# Use A, B in your Riccati recursion
```

### Collision-Free RRT Validity Check
```python
def is_valid(m, d, q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return all(d.contact[i].dist >= 0 for i in range(d.ncon))
```

### MPPI Rollout Batch
```python
def mppi_rollout(m, d0, ctrl_seq, noise_std=0.1, K=100):
    costs = np.zeros(K)
    for k in range(K):
        d = mujoco.MjData(m)
        mujoco.mj_copyData(d, m, d0)
        noise = np.random.randn(*ctrl_seq.shape) * noise_std
        for t, u in enumerate(ctrl_seq + noise):
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            costs[k] += your_cost_fn(d)
    return costs
```

---

## Notes on Data Types

- All `mjtNum` arrays are `double` (64-bit) by default.
- In Python, MuJoCo arrays appear as `numpy.ndarray` views — **do not
  reshape them** unless you know what you are doing; use `.copy()` if
  you need to store them.
- Quaternion convention is **[w, x, y, z]** throughout.
- Jacobians are stored **row-major**: `jacp[row, col]` where
  `row ∈ {0,1,2}` (x,y,z) and `col ∈ {0,...,nv-1}`.
- The `contact.frame` 3×3 matrix is **row-major**: rows are
  `[normal, tangent1, tangent2]`.
- In C++, `d->xpos` is a flat `mjtNum*` of length `nbody * 3`;
  access body `i` as `d->xpos + 3*i`.
