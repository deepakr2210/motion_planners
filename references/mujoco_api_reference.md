# MuJoCo API Reference for Motion Planning

Sources:
- `include/mujoco/mujoco.h` (google-deepmind/mujoco, main branch)
- `include/mujoco/mjdata.h`
- `doc/computation/index.rst`

---

## Table of Contents

1. [Mathematical Framework](#1-mathematical-framework)
2. [Key Data Structures](#2-key-data-structures)
3. [Simulation Pipeline](#3-simulation-pipeline)
4. [Forward Kinematics](#4-forward-kinematics)
5. [Jacobian Computation](#5-jacobian-computation)
6. [Dynamics Functions](#6-dynamics-functions)
7. [Mass Matrix Operations](#7-mass-matrix-operations)
8. [Collision Detection](#8-collision-detection)
9. [Contact Forces](#9-contact-forces)
10. [Ray Casting](#10-ray-casting)
11. [Object Velocity and Acceleration](#11-object-velocity-and-acceleration)
12. [Constraint Pipeline](#12-constraint-pipeline)
13. [Finite-Difference Derivatives](#13-finite-difference-derivatives)
14. [Quaternion Utilities](#14-quaternion-utilities)
15. [Vector Math Utilities](#15-vector-math-utilities)
16. [Matrix Math Utilities](#16-matrix-math-utilities)
17. [Sparse / Cholesky Math](#17-sparse--cholesky-math)
18. [State Management](#18-state-management)
19. [Name Lookup](#19-name-lookup)
20. [Energy and Sensors](#20-energy-and-sensors)
21. [Miscellaneous Utilities](#21-miscellaneous-utilities)

---

## 1. Mathematical Framework

### Equations of Motion (continuous time)

```
M·v̇ + c = τ + J^T·f
```

| Symbol | Size | Description | MuJoCo field |
|--------|------|-------------|--------------|
| nq | scalar | number of position coordinates | `mjModel.nq` |
| nv | scalar | number of degrees of freedom | `mjModel.nv` |
| nc | scalar | number of active constraints | `mjData.nefc` |
| q | nq | joint position | `mjData.qpos` |
| v | nv | joint velocity | `mjData.qvel` |
| τ | nv | applied force: passive + actuator + external | `mjData.qfrc_passive + qfrc_actuator + qfrc_applied` |
| c(q,v) | nv | bias force: Coriolis + centrifugal + gravity | `mjData.qfrc_bias` |
| M(q) | nv×nv | joint-space inertia matrix | `mjData.qM` (sparse) |
| J(q) | nc×nv | constraint Jacobian | `mjData.efc_J` |
| r(q) | nc | constraint residual | `mjData.efc_pos` |
| f(q,v,τ) | nc | constraint force | `mjData.efc_force` |

### Forward and Inverse Dynamics

```
Forward:  v̇ = M⁻¹(τ + J^T·f - c)
Inverse:  τ = M·v̇ + c - J^T·f
```

### Bias Force

The bias force `c` includes Coriolis, centrifugal, and gravitational forces. Computed using
the **Recursive Newton-Euler (RNE)** algorithm with acceleration set to zero.

### Mass Matrix

`M` is computed using the **Composite Rigid-Body (CRB)** algorithm. Stored in sparse format.
Its **L^T·D·L** factorization is also stored for efficient `M⁻¹·x` via back-substitution.

### Constraint Dual Problem

The constraint force `f` is the solution to a convex optimization:

```
f = argmin (1/2)·λ^T·(A+R)·λ + λ^T·(a_u - a_r)
    subject to λ ∈ Ω
```

Where:
- `A = J·M⁻¹·J^T` — inverse inertia in constraint space
- `a_u = J·M⁻¹·(τ-c) + J̇·v` — unconstrained acceleration in constraint space
- `R` — diagonal regularizer (makes constraints soft)
- `Ω` — constraint set (unconstrained for equality; box for friction loss; cone for contacts)

### Friction Cones

**Elliptic cone** (condim > 1):
```
K = { f ∈ R^n : f₁ ≥ 0, f₁² ≥ Σᵢ₌₂ⁿ fᵢ²/μᵢ₋₁² }
```

**Pyramidal cone** (condim > 1):
```
K = { f ∈ R^(2(n-1)) : f ≥ 0 }
```

### Contact Types by condim

| condim | Elliptic scalar constraints | Pyramidal scalar constraints | Description |
|--------|----------------------------|------------------------------|-------------|
| 1 | 1 | 1 | Frictionless (normal only) |
| 3 | 3 | 4 | Normal + tangential friction |
| 4 | 4 | 6 | + torsional friction (soft finger) |
| 6 | 6 | 10 | + rolling friction |

---

## 2. Key Data Structures

### mjContact struct

```c
struct mjContact_ {
  mjtNum dist;                    // distance between nearest points; neg: penetration
  mjtNum pos[3];                  // position of contact point: midpoint between geoms
  mjtNum frame[9];                // normal is in [0-2], points from geom[0] to geom[1]
  mjtNum includemargin;           // include if dist < includemargin = margin - gap
  mjtNum friction[5];             // tangent1, 2, spin, roll1, 2
  mjtNum solref[mjNREF];          // constraint solver reference, normal direction
  mjtNum solreffriction[mjNREF];  // constraint solver reference, friction directions
  mjtNum solimp[mjNIMP];          // constraint solver impedance
  mjtNum mu;                      // friction of regularized cone, set by mj_makeConstraint
  mjtNum H[36];                   // cone Hessian, set by mj_constraintUpdate
  int    dim;                     // contact space dimensionality: 1, 3, 4 or 6
  int    geom[2];                 // geom ids; -1 for flex
  int    flex[2];                 // flex ids; -1 for geom
  int    elem[2];                 // element ids; -1 for geom or flex vertex
  int    vert[2];                 // vertex ids; -1 for geom or flex element
  int    exclude;                 // 0: include, 1: in gap, 2: fused, 3: no dofs, 4: passive
  int    efc_address;             // address in efc; -1: not included
};
```

**Contact frame convention:**
- X-axis: contact normal (outward from geom[0] toward geom[1])
- Y, Z-axes: tangent plane
- `dist > 0`: separated; `dist = 0`: touching; `dist < 0`: penetrating

### mjData — motion-planning-relevant fields

```c
// STATE
mjtNum* qpos;            // joint position          (nq x 1)
mjtNum* qvel;            // joint velocity          (nv x 1)
mjtNum* qacc;            // joint acceleration      (nv x 1)
mjtNum* act;             // actuator activation     (na x 1)

// APPLIED FORCES (user-settable)
mjtNum* ctrl;            // actuator control        (nu x 1)
mjtNum* qfrc_applied;    // generalized force       (nv x 1)
mjtNum* xfrc_applied;    // Cartesian force/torque  (nbody x 6)

// KINEMATIC RESULTS (computed by mj_kinematics / mj_fwdPosition)
mjtNum* xpos;            // body frame position         (nbody x 3)
mjtNum* xquat;           // body frame orientation      (nbody x 4)
mjtNum* xmat;            // body frame orientation mat  (nbody x 9)
mjtNum* xipos;           // body CoM position           (nbody x 3)
mjtNum* ximat;           // body inertia frame mat      (nbody x 9)
mjtNum* geom_xpos;       // geom position               (ngeom x 3)
mjtNum* geom_xmat;       // geom orientation            (ngeom x 9)
mjtNum* site_xpos;       // site position               (nsite x 3)
mjtNum* site_xmat;       // site orientation            (nsite x 9)
mjtNum* subtree_com;     // subtree center of mass      (nbody x 3)

// DYNAMICS RESULTS
mjtNum* qfrc_bias;       // bias force c(q,v)           (nv x 1)
mjtNum* qfrc_passive;    // total passive force         (nv x 1)
mjtNum* qfrc_actuator;   // actuator force              (nv x 1)
mjtNum* qfrc_smooth;     // net unconstrained force     (nv x 1)
mjtNum* qfrc_constraint; // constraint force in jt space (nv x 1)

// INERTIA
mjtNum* qM;              // inertia (sparse)            (nM x 1)
mjtNum* qLD;             // L'*D*L factorization        (nC x 1)

// CONSTRAINT DATA
mjContact* contact;      // array of detected contacts  (ncon x 1)
mjtNum* efc_J;           // constraint Jacobian         (nJ x 1)
mjtNum* efc_pos;         // constraint position residual (nefc x 1)
mjtNum* efc_force;       // constraint force            (nefc x 1)
```

---

## 3. Simulation Pipeline

These functions drive the full simulation loop. For motion planning, you typically call
`mj_forward()` (position + velocity + acceleration) or its sub-steps.

```c
// Advance simulation by one step (calls mj_step1 then mj_step2 internally).
MJAPI void mj_step(const mjModel* m, mjData* d);

// Step 1: everything up to and including constraint forces.
MJAPI void mj_step1(const mjModel* m, mjData* d);

// Step 2: integrate state, advance time.
MJAPI void mj_step2(const mjModel* m, mjData* d);

// Run full forward dynamics pipeline without advancing time.
// Computes: kinematics, com, tendon, transmission, actuation,
//           crb, factorM, velocity, passive, actuation, acceleration, constraint.
MJAPI void mj_forward(const mjModel* m, mjData* d);

// Run full inverse dynamics pipeline.
// Computes: kinematics, com, tendon, transmission, actuation,
//           crb, factorM, velocity, passive, actuation, invConstraint.
MJAPI void mj_inverse(const mjModel* m, mjData* d);

// mj_forward with skip: skipstage skips up to a given pipeline stage,
// skipsensor=1 skips sensor computation.
MJAPI void mj_forwardSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor);
MJAPI void mj_inverseSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor);
```

### Individual pipeline stages

```c
// Position stage: runs kinematics, comPos, tendon, transmission, camlight,
//                 collision, makeConstraint.
MJAPI void mj_fwdPosition(const mjModel* m, mjData* d);

// Velocity stage: comVel, subtreeVel, passive forces.
MJAPI void mj_fwdVelocity(const mjModel* m, mjData* d);

// Actuation stage: actuator forces.
MJAPI void mj_fwdActuation(const mjModel* m, mjData* d);

// Acceleration stage: crb, factorM, fwdConstraint (solve for qacc).
MJAPI void mj_fwdAcceleration(const mjModel* m, mjData* d);

// Solve for constraint forces given qacc.
MJAPI void mj_fwdConstraint(const mjModel* m, mjData* d);

// Inverse position: kinematics, comPos, tendon, transmission, camlight,
//                   collision, makeConstraint.
MJAPI void mj_invPosition(const mjModel* m, mjData* d);

// Inverse velocity: comVel, subtreeVel, passive.
MJAPI void mj_invVelocity(const mjModel* m, mjData* d);

// Inverse constraint: solve for constraint forces given qacc.
MJAPI void mj_invConstraint(const mjModel* m, mjData* d);

// Compare forward and inverse dynamics results.
MJAPI void mj_compareFwdInv(const mjModel* m, mjData* d);
```

### Integrators

```c
// Euler (semi-implicit) integrator.
MJAPI void mj_Euler(const mjModel* m, mjData* d);

// N-th order Runge-Kutta integrator (N typically 4).
MJAPI void mj_RungeKutta(const mjModel* m, mjData* d, int N);
```

**Integrator summary:**
| Integrator | mjModel.opt.integrator | Best for |
|-----------|------------------------|----------|
| Euler | mjINT_EULER | Legacy; implicit joint damping only |
| implicitfast | mjINT_IMPLICITFAST | Recommended default; stability + speed |
| implicit | mjINT_IMPLICIT | Fast spinning bodies, gyroscopic forces |
| RK4 | mjINT_RK4 | Energy-conserving systems |

---

## 4. Forward Kinematics

After calling these functions, `mjData.xpos`, `xmat`, `xquat`, `geom_xpos`, `site_xpos`, etc.
are populated. They are called automatically by `mj_forward()` / `mj_fwdPosition()`.

```c
// Run forward kinematics.
// Computes: xpos, xquat, xmat, xipos, ximat, geom_xpos, geom_xmat,
//           site_xpos, site_xmat, subtree_com, cdof, cinert, crb.
// Reads: qpos
// NOTE: also reads qvel/qfrc_applied/xfrc_applied (violates pure position-stage assumption).
MJAPI void mj_kinematics(const mjModel* m, mjData* d);

// Map inertias and motion dofs to global frame centered at CoM.
// Must be called after mj_kinematics.
// Computes: cdof, cinert.
MJAPI void mj_comPos(const mjModel* m, mjData* d);

// Compute body and subtree velocities in CoM-centered frame.
// Computes: cvel, cdof_dot.
MJAPI void mj_comVel(const mjModel* m, mjData* d);

// Compute subtree CoM velocities (needed for some Jacobians).
MJAPI void mj_subtreeVel(const mjModel* m, mjData* d);

// Map global Cartesian position/orientation to joint coordinates (body-relative).
// Inputs: pos, quat in local body frame; body = body id.
// Outputs: xpos[3] (global position), xmat[9] (global rotation matrix).
MJAPI void mj_local2Global(mjData* d, mjtNum xpos[3], mjtNum xmat[9],
                           const mjtNum pos[3], const mjtNum quat[4],
                           int body, mjtByte sameframe);
```

### Reading kinematics results from mjData

After `mj_kinematics()` or `mj_forward()`:

```c
// Body frame position (global):      d->xpos  + body*3
// Body frame orientation (quat):     d->xquat + body*4
// Body frame orientation (matrix):   d->xmat  + body*9
// Body CoM position (global):        d->xipos + body*3
// Body inertia frame orientation:    d->ximat + body*9
// Geom position:                     d->geom_xpos + geom*3
// Geom orientation:                  d->geom_xmat + geom*9
// Site position:                     d->site_xpos + site*3
// Site orientation:                  d->site_xmat + site*9
// Subtree CoM:                       d->subtree_com + body*3
```

---

## 5. Jacobian Computation

All Jacobian functions require `mj_kinematics()` (and `mj_comPos()`) to have been called first.
The Jacobian maps joint velocities → end-effector velocity:

```
v_ee = J_p · qvel     (translational, 3 x nv)
ω_ee = J_r · qvel     (rotational,    3 x nv)
```

The transpose maps end-effector forces → joint torques: `τ = J^T · f_ee`

**Important:** All Jacobian computations are essentially free in CPU cost (O(nv) per call).

```c
// Compute 3-by-nv end-effector Jacobian of global point attached to given body.
//   jacp: translational Jacobian (3 x nv), nullable
//   jacr: rotational Jacobian    (3 x nv), nullable
//   point: 3D point in global coordinates
//   body: body id the point is attached to
// Either jacp or jacr may be NULL to skip that part.
MJAPI void mj_jac(const mjModel* m, const mjData* d,
                  mjtNum* jacp, mjtNum* jacr,
                  const mjtNum point[3], int body);

// Jacobian at body frame origin.
//   Equivalent to mj_jac with point = d->xpos + body*3
MJAPI void mj_jacBody(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int body);

// Jacobian at body center of mass.
//   Equivalent to mj_jac with point = d->xipos + body*3
MJAPI void mj_jacBodyCom(const mjModel* m, const mjData* d,
                         mjtNum* jacp, mjtNum* jacr, int body);

// Jacobian at subtree center of mass.
//   Only translational Jacobian (jacp, 3 x nv).
//   Requires mj_subtreeVel to have been called.
MJAPI void mj_jacSubtreeCom(const mjModel* m, mjData* d,
                            mjtNum* jacp, int body);

// Jacobian at geom center.
//   Equivalent to mj_jac with point = d->geom_xpos + geom*3
MJAPI void mj_jacGeom(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int geom);

// Jacobian at site position.
//   Equivalent to mj_jac with point = d->site_xpos + site*3
MJAPI void mj_jacSite(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int site);
```

### Usage pattern (Python-style)

```python
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
# Full 6xnv Jacobian:
J = np.vstack([jacp, jacr])
# Pseudo-inverse for IK:
J_pinv = np.linalg.pinv(J)
dq = J_pinv @ delta_pose
```

---

## 6. Dynamics Functions

### Forward and Inverse Dynamics (high-level)

```c
// Full forward dynamics: populates qacc from qpos, qvel, ctrl/qfrc_applied.
MJAPI void mj_forward(const mjModel* m, mjData* d);

// Full inverse dynamics: populates qfrc_inverse from qpos, qvel, qacc.
MJAPI void mj_inverse(const mjModel* m, mjData* d);
```

### Recursive Newton-Euler (RNE)

```c
// RNE: compute M(qpos)*qacc + C(qpos,qvel).
//   flg_acc = 1: result = M·qacc + C(q,v)    (full inverse dynamics, no constraints)
//   flg_acc = 0: result = C(q,v)              (bias forces only: Coriolis + gravity)
//   result: nv-dimensional output vector
// Note: this does NOT include constraint forces. It computes the ideal inverse
// dynamics ignoring contacts. For full inverse with contacts, use mj_inverse().
MJAPI void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result);
```

**RNE use cases:**
- `flg_acc=0`: gravity + Coriolis compensation vector `c(q,v)` → `mjData.qfrc_bias`
- `flg_acc=1`: required joint torques for desired acceleration (ignoring constraints)

### Composite Rigid Body (CRB)

```c
// Composite Rigid Body algorithm: compute joint-space inertia matrix M.
// Result stored in mjData.qM (sparse format).
MJAPI void mj_crb(const mjModel* m, mjData* d);
```

### Tendon and Transmission

```c
// Compute tendon lengths and moment arms.
MJAPI void mj_tendon(const mjModel* m, mjData* d);

// Compute actuator transmission lengths and moment arms.
MJAPI void mj_transmission(const mjModel* m, mjData* d);
```

---

## 7. Mass Matrix Operations

The joint-space inertia matrix `M` is stored in a **sparse lower-triangular** format in `mjData.qM`.

```c
// Convert sparse inertia matrix M into full (dense) matrix.
//   dst: output dense matrix (nv x nv)
//   M:   input sparse matrix (from mjData.qM)
MJAPI void mj_fullM(const mjModel* m, mjtNum* dst, const mjtNum* M);

// Multiply vector by inertia matrix: res = M * vec.
//   Uses sparse internal representation for efficiency.
//   res, vec: nv-dimensional vectors
MJAPI void mj_mulM(const mjModel* m, const mjData* d,
                   mjtNum* res, const mjtNum* vec);

// Compute L'*D*L factorization of M (stored in mjData.qLD, qLDiag).
// Must be called before mj_solveM.
MJAPI void mj_factorM(const mjModel* m, mjData* d);

// Solve linear system M * x = y using precomputed factorization.
//   x: output (nv x n)
//   y: input  (nv x n)
//   n: number of right-hand-side vectors
// Must call mj_factorM first.
MJAPI void mj_solveM(const mjModel* m, mjData* d,
                     mjtNum* x, const mjtNum* y, int n);

// Solve M * x = y but use provided sqrtInvD scaling.
MJAPI void mj_solveM2(const mjModel* m, mjData* d,
                      mjtNum* x, const mjtNum* y,
                      const mjtNum* sqrtInvD, int n);
```

### Solver type queries

```c
// Return 1 if pyramidal friction cone is used.
MJAPI int mj_isPyramidal(const mjModel* m);

// Return 1 if sparse constraint Jacobian representation is used.
MJAPI int mj_isSparse(const mjModel* m);

// Return 1 if dual (constraint-space) solver is used.
MJAPI int mj_isDual(const mjModel* m);
```

---

## 8. Collision Detection

Collision detection is called automatically in `mj_fwdPosition()`. You can call it directly
to get contact information without running full dynamics.

```c
// Run collision detection pipeline.
// Populates: mjData.contact[0..ncon-1], mjData.ncon.
// Pipeline stages:
//   1. Broad phase: sweep-and-prune along principal eigenvector axis
//   2. Mid phase: AABB hierarchy per body
//   3. Narrow phase: GJK/EPA (native) or MPR/libccd (legacy)
MJAPI void mj_collision(const mjModel* m, mjData* d);

// Return smallest signed distance between two geoms.
//   geom1, geom2: geom ids
//   distmax: maximum distance to check (performance cutoff)
//   fromto[6]: optional output — segment endpoints [p1x,p1y,p1z, p2x,p2y,p2z]
//              p1 is on geom1, p2 is on geom2. Nullable.
//   Returns: signed distance (neg = penetration, pos = separation)
// Note: does NOT require mj_collision to have been called first.
MJAPI mjtNum mj_geomDistance(const mjModel* m, mjData* d,
                             int geom1, int geom2,
                             mjtNum distmax, mjtNum fromto[6]);
```

### Collision filtering rules (applied in order)

1. Collision function availability (supported geom type pair)
2. Bounding sphere test (accounting for margin)
3. Kinematic hierarchy: exclude same-body and parent-child pairs (unless `contype`/`conaffinity` override)
4. `contype & conaffinity` bitwise compatibility test

### Geom types (mjtGeom enum)

| Value | Type | Notes |
|-------|------|-------|
| mjGEOM_PLANE | Plane | infinite |
| mjGEOM_SPHERE | Sphere | |
| mjGEOM_CAPSULE | Capsule | cylinder + hemispherical caps |
| mjGEOM_CYLINDER | Cylinder | |
| mjGEOM_ELLIPSOID | Ellipsoid | |
| mjGEOM_BOX | Box | |
| mjGEOM_MESH | Mesh | convex decomposition required for non-convex |
| mjGEOM_SDF | SDF | signed distance field |
| mjGEOM_HFIELD | Heightfield | |

### Accessing contacts after mj_collision()

```c
// Number of active contacts:
int ncon = d->ncon;

// Iterate contacts:
for (int i = 0; i < ncon; i++) {
    mjContact* c = &d->contact[i];
    // c->dist        — signed distance (neg = penetrating)
    // c->pos[3]      — contact point in world frame
    // c->frame[9]    — contact frame (normal = frame[0:3])
    // c->geom[0,1]   — geom indices
    // c->dim         — contact dimensionality (1,3,4,6)
    // c->friction[5] — friction coefficients
    // c->efc_address — index into efc arrays (-1 if excluded)
}
```

---

## 9. Contact Forces

```c
// Compute 6D contact force/torque in contact frame for contact[id].
//   result[6]: output [fx, fy, fz, tx, ty, tz] in contact frame
//              first axis (fx) is normal force, fy/fz are friction
// Must be called after constraint solve (mj_forward or mj_fwdConstraint).
MJAPI void mj_contactForce(const mjModel* m, const mjData* d,
                           int id, mjtNum result[6]);

// Build constraint equations (populate efc_* arrays).
MJAPI void mj_makeConstraint(const mjModel* m, mjData* d);

// Compute efc_AR = J * M⁻¹ * J^T (inverse inertia in constraint space).
MJAPI void mj_projectConstraint(const mjModel* m, mjData* d);

// Compute constraint velocity (efc_vel) and reference acceleration (efc_aref).
MJAPI void mj_referenceConstraint(const mjModel* m, mjData* d);

// Compute efc_state, efc_force, qfrc_constraint from constraint solution jar.
//   jar: Jacobian-scaled constraint force (input from solver)
//   cost[1]: optional output — constraint cost
//   flg_coneHessian: compute cone Hessian H in each mjContact if 1
MJAPI void mj_constraintUpdate(const mjModel* m, mjData* d,
                               const mjtNum* jar, mjtNum cost[1],
                               int flg_coneHessian);
```

### Pyramid/Ellipse force encoding

```c
// Convert contact force to pyramidal representation.
//   pyramid: output (2*(dim-1) x 1)
//   force:   input (dim x 1)
//   mu:      friction coefficients (dim-1 x 1)
MJAPI void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force,
                             const mjtNum* mu, int dim);

// Convert pyramidal representation back to contact force.
MJAPI void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid,
                             const mjtNum* mu, int dim);
```

---

## 10. Ray Casting

```c
// Intersect ray (origin=pnt, direction=vec, x>=0) with visible geoms.
//   pnt[3]: ray origin in world frame
//   vec[3]: ray direction in world frame (need not be unit length)
//   geomgroup: array of 6 bytes, 1=include that group, NULL=include all
//   flg_static: 1=include static geoms (non-dynamic bodies), 0=skip
//   bodyexclude: body id to exclude from test (-1 = none)
//   geomid[1]: output — id of nearest hit geom, -1 if no hit. Nullable.
//   normal[3]: output — surface normal at hit point. Nullable.
//   Returns: distance x to nearest surface, or -1 if no intersection.
MJAPI mjtNum mj_ray(const mjModel* m, const mjData* d,
                    const mjtNum pnt[3], const mjtNum vec[3],
                    const mjtByte* geomgroup, mjtByte flg_static,
                    int bodyexclude, int geomid[1], mjtNum normal[3]);

// Intersect ray with a specific heightfield geom.
//   geomid: id of the hfield geom
//   normal[3]: output surface normal. Nullable.
//   Returns: distance or -1.
MJAPI mjtNum mj_rayHfield(const mjModel* m, const mjData* d,
                          int geomid, const mjtNum pnt[3],
                          const mjtNum vec[3], mjtNum normal[3]);

// Intersect ray with a specific mesh geom.
//   geomid: id of the mesh geom
//   normal[3]: output surface normal. Nullable.
//   Returns: distance or -1.
MJAPI mjtNum mj_rayMesh(const mjModel* m, const mjData* d,
                        int geomid, const mjtNum pnt[3],
                        const mjtNum vec[3], mjtNum normal[3]);
```

---

## 11. Object Velocity and Acceleration

```c
// Compute object 6D velocity [rot(3); lin(3)] in object-centered frame.
//   objtype: mjtObj type (mjOBJ_BODY, mjOBJ_GEOM, mjOBJ_SITE, ...)
//   objid: object id
//   res[6]: output — [angular velocity (3); linear velocity (3)]
//   flg_local: 0=world orientation, 1=object-local orientation
// Requires mj_fwdVelocity (comVel) to have been called.
MJAPI void mj_objectVelocity(const mjModel* m, const mjData* d,
                             int objtype, int objid,
                             mjtNum res[6], int flg_local);

// Compute object 6D acceleration [rot(3); lin(3)] in object-centered frame.
//   Same signature as mj_objectVelocity.
//   Requires mj_forward (full pipeline) to have been called.
MJAPI void mj_objectAcceleration(const mjModel* m, const mjData* d,
                                 int objtype, int objid,
                                 mjtNum res[6], int flg_local);
```

---

## 12. Constraint Pipeline

```c
// Build constraint Jacobian and related arrays.
MJAPI void mj_makeConstraint(const mjModel* m, mjData* d);

// Inverse constraint inertia: efc_AR = J * M⁻¹ * J^T
MJAPI void mj_projectConstraint(const mjModel* m, mjData* d);

// Reference acceleration and constraint velocity.
MJAPI void mj_referenceConstraint(const mjModel* m, mjData* d);

// Update constraint state given solved forces.
MJAPI void mj_constraintUpdate(const mjModel* m, mjData* d,
                               const mjtNum* jar, mjtNum cost[1],
                               int flg_coneHessian);
```

---

## 13. Finite-Difference Derivatives

### State transition Jacobians (for control/planning)

```c
// Finite-differenced transition matrices (control theory notation):
//   d(x_next)/dx = A    (2*nv+na x 2*nv+na)
//   d(x_next)/du = B    (2*nv+na x nu)
//   d(sensor)/dx = C    (nsensordata x 2*nv+na)
//   d(sensor)/du = D    (nsensordata x nu)
//
//   eps:          finite difference epsilon
//   flg_centered: 0=forward, 1=centered differences
//   A,B,C,D: nullable — only compute if non-NULL
//
// Useful for LQR/iLQR/trajectory optimization.
MJAPI void mjd_transitionFD(const mjModel* m, mjData* d,
                            mjtNum eps, mjtByte flg_centered,
                            mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D);
```

### Inverse dynamics Jacobians

```c
// Finite-differenced Jacobians of inverse dynamics:
//   (force, sensors) = mj_inverse(state, acceleration)
//
//   DfDq: nv x nv  — d(qfrc_inverse)/d(qpos)
//   DfDv: nv x nv  — d(qfrc_inverse)/d(qvel)
//   DfDa: nv x nv  — d(qfrc_inverse)/d(qacc)
//   DsDq: nv x nsensordata
//   DsDv: nv x nsensordata
//   DsDa: nv x nsensordata
//   DmDq: nv x nM  — d(qM)/d(qpos) (mass matrix Jacobian)
//
//   flg_actuation: subtract qfrc_actuator from qfrc_inverse if 1
//   All outputs nullable.
MJAPI void mjd_inverseFD(const mjModel* m, mjData* d,
                         mjtNum eps, mjtByte flg_actuation,
                         mjtNum* DfDq, mjtNum* DfDv, mjtNum* DfDa,
                         mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa,
                         mjtNum* DmDq);

// Derivatives of mju_subQuat (quaternion subtraction → 3D velocity).
//   Da[9]: d(result)/d(qa)
//   Db[9]: d(result)/d(qb)
MJAPI void mjd_subQuat(const mjtNum qa[4], const mjtNum qb[4],
                       mjtNum Da[9], mjtNum Db[9]);

// Derivatives of mju_quatIntegrate.
//   Dquat[9]: d(result)/d(quat)
//   Dvel[9]:  d(result)/d(vel)
//   Dscale[3]: d(result)/d(scale)
MJAPI void mjd_quatIntegrate(const mjtNum vel[3], mjtNum scale,
                             mjtNum Dquat[9], mjtNum Dvel[9],
                             mjtNum Dscale[3]);
```

---

## 14. Quaternion Utilities

MuJoCo quaternion convention: `[w, x, y, z]` (scalar first).

```c
// Rotate vector by quaternion: res = rotate(vec, quat).
MJAPI void mju_rotVecQuat(mjtNum res[3], const mjtNum vec[3],
                          const mjtNum quat[4]);

// Quaternion conjugate (inverse for unit quaternion): res = conj(quat).
MJAPI void mju_negQuat(mjtNum res[4], const mjtNum quat[4]);

// Quaternion product: res = quat1 * quat2.
MJAPI void mju_mulQuat(mjtNum res[4], const mjtNum quat1[4],
                       const mjtNum quat2[4]);

// Quaternion * axis product: res = quat * quat(0, axis).
MJAPI void mju_mulQuatAxis(mjtNum res[4], const mjtNum quat[4],
                           const mjtNum axis[3]);

// Axis-angle to quaternion.
//   axis: rotation axis (need not be unit)
//   angle: rotation angle in radians
MJAPI void mju_axisAngle2Quat(mjtNum res[4], const mjtNum axis[3],
                              mjtNum angle);

// Quaternion to angular velocity representation.
//   Computes the 3D angular velocity that would rotate identity to quat in time dt.
//   res[3]: angular velocity vector
MJAPI void mju_quat2Vel(mjtNum res[3], const mjtNum quat[4], mjtNum dt);

// Quaternion to 3x3 rotation matrix.
MJAPI void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]);

// 3x3 rotation matrix to quaternion.
MJAPI void mju_mat2Quat(mjtNum quat[4], const mjtNum mat[9]);

// Integrate quaternion by angular velocity.
//   quat: quaternion to integrate (modified in-place)
//   vel[3]: angular velocity
//   scale: time scale (dt)
MJAPI void mju_quatIntegrate(mjtNum quat[4], const mjtNum vel[3],
                             mjtNum scale);

// Compute quaternion that rotates Z-axis to given vector.
MJAPI void mju_quatZ2Vec(mjtNum quat[4], const mjtNum vec[3]);

// Euler angles to quaternion.
//   euler[3]: rotation angles
//   seq: axis sequence string (e.g. "xyz", "ZYX")
MJAPI void mju_euler2Quat(mjtNum quat[4], const mjtNum euler[3],
                          const char* seq);

// Rotation matrix to quaternion (robust variant).
//   Returns 0 on success, 1 if matrix is not a valid rotation.
MJAPI int mju_mat2Rot(mjtNum quat[4], const mjtNum mat[9]);

// Subtract quaternions, express result as 3D velocity:
//   res = log(qb⁻¹ * qa) expressed as axis-angle
MJAPI void mju_subQuat(mjtNum res[3], const mjtNum qa[4],
                       const mjtNum qb[4]);

// Transform vector by pose (position + quaternion).
//   res = pos + rotate(vec, quat)
MJAPI void mju_trnVecPose(mjtNum res[3], const mjtNum pos[3],
                          const mjtNum quat[4], const mjtNum vec[3]);
```

### Quaternion normalization

```c
// Normalize quaternion in place; return length before normalization.
MJAPI mjtNum mju_normalize4(mjtNum vec[4]);

// Re-normalize all quaternions in qpos (model-aware).
MJAPI void mj_normalizeQuat(const mjModel* m, mjtNum* qpos);
```

---

## 15. Vector Math Utilities

### Fixed-size 3D vector operations

```c
MJAPI void   mju_zero3(mjtNum res[3]);
MJAPI void   mju_copy3(mjtNum res[3], const mjtNum data[3]);
MJAPI void   mju_scl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl);
MJAPI void   mju_add3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]);
MJAPI void   mju_sub3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]);
MJAPI void   mju_addTo3(mjtNum res[3], const mjtNum vec[3]);         // res += vec
MJAPI void   mju_subFrom3(mjtNum res[3], const mjtNum vec[3]);       // res -= vec
MJAPI void   mju_addToScl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl); // res += vec*scl
MJAPI void   mju_addScl3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3], mjtNum scl);
MJAPI mjtNum mju_normalize3(mjtNum vec[3]);   // normalize in place; return original length
MJAPI mjtNum mju_norm3(const mjtNum vec[3]);  // return length (no normalization)
MJAPI mjtNum mju_dot3(const mjtNum vec1[3], const mjtNum vec2[3]);
MJAPI mjtNum mju_dist3(const mjtNum pos1[3], const mjtNum pos2[3]);  // Cartesian distance
MJAPI void   mju_cross(mjtNum res[3], const mjtNum a[3], const mjtNum b[3]); // res = a × b
```

### General-size vector operations

```c
MJAPI void   mju_zero(mjtNum* res, int n);
MJAPI void   mju_fill(mjtNum* res, mjtNum val, int n);
MJAPI void   mju_copy(mjtNum* res, const mjtNum* vec, int n);
MJAPI mjtNum mju_sum(const mjtNum* vec, int n);
MJAPI mjtNum mju_L1(const mjtNum* vec, int n);           // sum(|vec[i]|)
MJAPI void   mju_scl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);
MJAPI void   mju_add(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);
MJAPI void   mju_sub(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, int n);
MJAPI void   mju_addTo(mjtNum* res, const mjtNum* vec, int n);       // res += vec
MJAPI void   mju_subFrom(mjtNum* res, const mjtNum* vec, int n);     // res -= vec
MJAPI void   mju_addToScl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n); // res += vec*scl
MJAPI void   mju_addScl(mjtNum* res, const mjtNum* vec1, const mjtNum* vec2, mjtNum scl, int n);
MJAPI mjtNum mju_normalize(mjtNum* res, int n);   // normalize in place; return original length
MJAPI mjtNum mju_norm(const mjtNum* res, int n);  // return L2 norm
MJAPI mjtNum mju_dot(const mjtNum* vec1, const mjtNum* vec2, int n);
```

---

## 16. Matrix Math Utilities

### Matrix-vector products

```c
// res = mat * vec   (nr x nc matrix times nc-vector → nr-vector)
MJAPI void mju_mulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);

// res = mat' * vec  (nc x nr transposed; nc-vector result from nr-vector)
MJAPI void mju_mulMatTVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);

// Scalar: return vec1' * mat * vec2  (n x n matrix)
MJAPI mjtNum mju_mulVecMatVec(const mjtNum* vec1, const mjtNum* mat,
                              const mjtNum* vec2, int n);

// Fixed-size 3x3 variants:
MJAPI void mju_mulMatVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);
MJAPI void mju_mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);
```

### Matrix-matrix products

```c
// res = mat1 * mat2    (r1 x c1 times c1 x c2 → r1 x c2)
MJAPI void mju_mulMatMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                         int r1, int c1, int c2);

// res = mat1 * mat2'   (r1 x c1 times r2 x c1 → r1 x r2)
MJAPI void mju_mulMatMatT(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                          int r1, int c1, int r2);

// res = mat1' * mat2   (r1 x c1 transposed times r1 x c2 → c1 x c2)
MJAPI void mju_mulMatTMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2,
                          int r1, int c1, int c2);

// res = mat' * diag * mat  (nc x nr * diag(nr) * nr x nc → nc x nc)
// If diag is NULL: res = mat' * mat
MJAPI void mju_sqrMatTD(mjtNum* res, const mjtNum* mat, const mjtNum* diag, int nr, int nc);
```

### Matrix utilities

```c
// Transpose: res = mat'  (nr x nc → nc x nr)
MJAPI void mju_transpose(mjtNum* res, const mjtNum* mat, int nr, int nc);

// Symmetrize: res = (mat + mat') / 2  (n x n square)
MJAPI void mju_symmetrize(mjtNum* res, const mjtNum* mat, int n);

// Set identity: mat = I_n
MJAPI void mju_eye(mjtNum* mat, int n);
```

### Eigenvalue decomposition

```c
// Eigenvalue decomposition of symmetric 3x3 matrix.
//   mat = eigvec * diag(eigval) * eigvec'
//   eigval[3]: eigenvalues (ascending)
//   eigvec[9]: eigenvectors as columns
//   quat[4]:   eigenvectors as quaternion (rotation matrix form)
//   Returns: number of iterations until convergence
MJAPI int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9],
                   mjtNum quat[4], const mjtNum mat[9]);
```

### Spatial vector transforms

```c
// Coordinate transform of 6D motion or force vector [rot(3); lin(3)].
//   rotnew2old: 3x3 rotation matrix, NULL=no rotation
//   newpos, oldpos: frame positions
//   flg_force: 0=motion vector, 1=force vector
MJAPI void mju_transformSpatial(mjtNum res[6], const mjtNum vec[6],
                                int flg_force, const mjtNum newpos[3],
                                const mjtNum oldpos[3],
                                const mjtNum rotnew2old[9]);
```

### Bounded QP solver (for IK/optimization)

```c
// Minimize: 0.5 * x' * H * x + x' * g
// Subject to: lower <= x <= upper
//   res:   output (n x 1)
//   R:     output Cholesky factor of H (n x n), nullable
//   index: output active set indices (n x 1), nullable
//   H:     Hessian (n x n, positive definite)
//   g:     gradient (n x 1)
//   lower, upper: bounds (n x 1), nullable (±∞ if NULL)
//   Returns: rank, or -1 if failed
MJAPI int mju_boxQP(mjtNum* res, mjtNum* R, int* index,
                    const mjtNum* H, const mjtNum* g, int n,
                    const mjtNum* lower, const mjtNum* upper);
```

---

## 17. Sparse / Cholesky Math

### Dense Cholesky

```c
// Cholesky decomposition in-place: mat = L * L'
//   mindiag: minimum diagonal value (regularization)
//   Returns: rank (number of non-zero diagonal elements)
MJAPI int mju_cholFactor(mjtNum* mat, int n, mjtNum mindiag);

// Solve (L * L') * res = vec using precomputed Cholesky factor.
MJAPI void mju_cholSolve(mjtNum* res, const mjtNum* mat,
                         const mjtNum* vec, int n);

// Rank-one update: L*L' ±= x*x'
//   flg_plus: 1=add, 0=subtract
//   Returns: rank
MJAPI int mju_cholUpdate(mjtNum* mat, mjtNum* x, int n, int flg_plus);
```

### Band-diagonal Cholesky (for banded systems)

```c
// Band-dense Cholesky factorization.
//   mat layout: (ntotal-ndense) x nband band rows + ndense x ntotal dense rows
//   diagadd, diagmul: add diagadd + diagmul*mat_ii to diagonal before factorize
//   Returns: minimum diagonal value (0 if rank-deficient)
MJAPI mjtNum mju_cholFactorBand(mjtNum* mat, int ntotal, int nband,
                                int ndense, mjtNum diagadd, mjtNum diagmul);

// Solve banded system using band Cholesky factor.
MJAPI void mju_cholSolveBand(mjtNum* res, const mjtNum* mat,
                             const mjtNum* vec, int ntotal, int nband, int ndense);

// Convert banded → dense. Fill upper triangle if flg_sym > 0.
MJAPI void mju_band2Dense(mjtNum* res, const mjtNum* mat, int ntotal,
                          int nband, int ndense, mjtByte flg_sym);

// Convert dense → banded.
MJAPI void mju_dense2Band(mjtNum* res, const mjtNum* mat, int ntotal,
                          int nband, int ndense);

// Multiply band matrix by nvec vectors.
MJAPI void mju_bandMulMatVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense,
                             int nvec, mjtByte flg_sym);

// Address of diagonal element i in band-dense storage.
MJAPI int mju_bandDiag(int i, int ntotal, int nband, int ndense);
```

### Sparse matrix conversion

```c
// Dense → CSR sparse.
//   rownnz, rowadr, colind: CSR arrays
//   nnz: size of res and colind buffers
//   Returns: 1 if buffers too small, 0 on success
MJAPI int mju_dense2sparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                           int* rownnz, int* rowadr, int* colind, int nnz);

// CSR sparse → dense.
MJAPI void mju_sparse2dense(mjtNum* res, const mjtNum* mat, int nr, int nc,
                            const int* rownnz, const int* rowadr, const int* colind);
```

---

## 18. State Management

### Position integration utilities

```c
// Differentiate position: qvel = (qpos2 - qpos1) / dt
// Handles quaternion subtraction correctly via SO(3) tangent space.
//   qvel: output (nv x 1)
//   qpos1, qpos2: input positions (nq x 1)
MJAPI void mj_differentiatePos(const mjModel* m, mjtNum* qvel,
                               mjtNum dt, const mjtNum* qpos1,
                               const mjtNum* qpos2);

// Integrate position: qpos += qvel * dt
// Handles quaternion integration correctly.
MJAPI void mj_integratePos(const mjModel* m, mjtNum* qpos,
                           const mjtNum* qvel, mjtNum dt);

// Re-normalize all quaternions in qpos.
MJAPI void mj_normalizeQuat(const mjModel* m, mjtNum* qpos);
```

### Spring-damper analytic integration

```c
// Analytically integrate spring-damper ODE.
//   pos0: initial position
//   vel0: initial velocity
//   Kp:   spring stiffness
//   Kv:   damping coefficient
//   dt:   time step
//   Returns: position at dt
MJAPI mjtNum mju_springDamper(mjtNum pos0, mjtNum vel0,
                              mjtNum Kp, mjtNum Kv, mjtNum dt);
```

### Full state get/set

```c
// Get concatenated state vector.
//   state: output buffer
//   sig:   bitmask of mjtState flags (e.g. mjSTATE_QPOS | mjSTATE_QVEL)
MJAPI void mj_getState(const mjModel* m, const mjData* d,
                       mjtNum* state, int sig);

// Set state from concatenated vector.
MJAPI void mj_setState(const mjModel* m, mjData* d,
                       const mjtNum* state, int sig);
```

### Memory stack

```c
// Allocate array of mjtNums on mjData stack (lifetime = current stack frame).
// Aborts on overflow.
MJAPI mjtNum* mj_stackAllocNum(mjData* d, size_t size);

// Push a new stack frame.
MJAPI void mj_markStack(mjData* d);

// Pop the current stack frame (frees all allocations since last mj_markStack).
MJAPI void mj_freeStack(mjData* d);
```

---

## 19. Name Lookup

```c
// Get id of object with given type and name. Returns -1 if not found.
//   type: mjtObj enum value (mjOBJ_BODY, mjOBJ_GEOM, mjOBJ_SITE, mjOBJ_JOINT, ...)
MJAPI int mj_name2id(const mjModel* m, int type, const char* name);

// Get name of object with given type and id. Returns NULL if not found.
MJAPI const char* mj_id2name(const mjModel* m, int type, int id);
```

**Common mjtObj values:**

| Value | Description |
|-------|-------------|
| mjOBJ_BODY | Rigid body |
| mjOBJ_JOINT | Joint |
| mjOBJ_GEOM | Geometry |
| mjOBJ_SITE | Site (reference point) |
| mjOBJ_CAMERA | Camera |
| mjOBJ_ACTUATOR | Actuator |
| mjOBJ_SENSOR | Sensor |
| mjOBJ_TENDON | Tendon |
| mjOBJ_MESH | Mesh asset |

---

## 20. Energy and Sensors

```c
// Evaluate position-dependent sensors (encoders, positions, etc.).
MJAPI void mj_sensorPos(const mjModel* m, mjData* d);

// Evaluate velocity-dependent sensors.
MJAPI void mj_sensorVel(const mjModel* m, mjData* d);

// Evaluate acceleration and force-dependent sensors.
MJAPI void mj_sensorAcc(const mjModel* m, mjData* d);

// Compute potential energy from position (stored in mjData.energy[0]).
MJAPI void mj_energyPos(const mjModel* m, mjData* d);

// Compute kinetic energy from velocity (stored in mjData.energy[1]).
MJAPI void mj_energyVel(const mjModel* m, mjData* d);
```

---

## 21. Miscellaneous Utilities

### Printing / debugging

```c
// Print mjData contents to text file.
MJAPI void mj_printData(const mjModel* m, const mjData* d, const char* filename);

// Print mjModel contents to text file.
MJAPI void mj_printModel(const mjModel* m, const char* filename);
```

### Error and warning

```c
// Fatal error (does not return).
MJAPI void mju_error(const char* msg, ...);

// Non-fatal warning (returns to caller).
MJAPI void mju_warning(const char* msg, ...);

// Write log entry to MUJOCO_LOG.TXT.
MJAPI void mju_writeLog(const char* type, const char* msg);
```

---

## Appendix A: Motion Planning Workflow

### Typical call sequence for Jacobian-based IK

```python
# 1. Set joint configuration
data.qpos[:] = q_current

# 2. Run forward kinematics
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)

# 3. Read end-effector pose
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
ee_pos = data.site_xpos[site_id].copy()      # (3,)
ee_mat = data.site_xmat[site_id].reshape(3,3) # (3,3)

# 4. Compute Jacobian
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp, jacr])  # (6, nv)

# 5. Compute error
pos_err = target_pos - ee_pos
rot_err = ...  # axis-angle error from mat

# 6. IK step (damped least squares)
lam = 0.01  # damping
dq = J.T @ np.linalg.solve(J @ J.T + lam*np.eye(6), np.concatenate([pos_err, rot_err]))

# 7. Integrate
mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
mujoco.mj_normalizeQuat(model, data.qpos)
```

### Typical call sequence for dynamics

```python
# Forward dynamics (compute qacc from qpos, qvel, ctrl):
mujoco.mj_forward(model, data)
# → data.qacc populated

# Gravity + Coriolis only (no control, no acceleration):
data.qacc[:] = 0
mujoco.mj_rne(model, data, 0, result)  # result = c(q,v)

# Full inverse dynamics (compute required torques):
# Set data.qpos, data.qvel, data.qacc to desired trajectory
mujoco.mj_inverse(model, data)
# → data.qfrc_inverse populated
```

### Collision / distance queries

```python
# After mj_forward() or mj_fwdPosition():
print(f"Contacts: {data.ncon}")
for i in range(data.ncon):
    c = data.contact[i]
    print(f"  geoms {c.geom[0]},{c.geom[1]}  dist={c.dist:.4f}  pos={c.pos}")

# Point-to-point distance (no collision needed):
dist = mujoco.mj_geomDistance(model, data, geom1_id, geom2_id, max_dist, fromto)
# fromto[0:3] = nearest point on geom1
# fromto[3:6] = nearest point on geom2

# Ray cast:
geomid = np.zeros(1, dtype=np.int32)
normal = np.zeros(3)
dist = mujoco.mj_ray(model, data, ray_origin, ray_dir,
                     None, 1, -1, geomid, normal)
```

### State transition for trajectory optimization / MPC

```python
# Finite-differenced A, B matrices:
nstate = 2 * model.nv + model.na
A = np.zeros((nstate, nstate))
B = np.zeros((nstate, model.nu))
mujoco.mjd_transitionFD(model, data, eps=1e-6, flg_centered=True,
                        A=A, B=B, C=None, D=None)
# Use A, B for LQR/iLQR
```

---

## Appendix B: Solver Algorithms

MuJoCo provides three solvers for the constraint optimization problem:

| Solver | Best for | Notes |
|--------|----------|-------|
| Newton | Small-medium systems | Exact Newton, Cholesky, line search |
| CG | Large systems | Polak-Ribière-Plus, exact line search |
| PGS | Real-time, many contacts | Projected Gauss-Seidel, 1st-order convergence |

**Constraint islands:** MuJoCo automatically identifies disjoint contact subgraphs and
solves them independently in parallel, improving both convergence and CPU efficiency.

---

## Appendix C: Computation Pipeline Order

```
mj_forward() calls (in order):
  1. mj_fwdPosition()
     → mj_kinematics()       ← xpos, xquat, xmat, geom_xpos, site_xpos
     → mj_comPos()           ← cdof, cinert
     → mj_tendon()           ← ten_length, ten_J
     → mj_transmission()     ← actuator_length, actuator_moment
     → mj_camlight()
     → mj_collision()        ← contact[], ncon
     → mj_makeConstraint()   ← efc_J, efc_pos, efc_D, efc_aref
  2. mj_fwdVelocity()
     → mj_comVel()           ← cvel, cdof_dot
     → mj_subtreeVel()
     → passive forces        ← qfrc_passive
  3. mj_fwdActuation()       ← actuator_force, qfrc_actuator
  4. mj_fwdAcceleration()
     → mj_crb()              ← qM (mass matrix, sparse)
     → mj_factorM()          ← qLD (L'DL factorization)
     → mj_fwdConstraint()    ← efc_force, qacc, qfrc_constraint
```
