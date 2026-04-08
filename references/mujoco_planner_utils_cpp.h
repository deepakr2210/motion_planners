/**
 * mujoco_planner_utils_cpp.h
 *
 * Reference sheet: MuJoCo C++ API utilities for custom motion planners.
 *
 * This file is NOT meant to be compiled — it is a curated reference of
 * every MuJoCo function useful when building planners (RRT, iLQR, MPC,
 * trajectory optimisation, IK, etc.).  Each entry shows the exact
 * signature from <mujoco/mujoco.h>, the array dimensions, and a
 * one-line description of inputs/outputs.
 *
 * Includes:
 *   mujoco/mujoco.h   — all public API
 *   mujoco/mjdata.h   — mjData / mjContact struct fields
 *   mujoco/mjmodel.h  — mjModel struct fields
 *
 * Notation used in comments
 *   nq  = m->nq    number of position coordinates (may include quaternions)
 *   nv  = m->nv    number of velocity / generalised-force DOFs
 *   na  = m->na    number of actuator activations
 *   nu  = m->nu    number of actuator controls
 *   nb  = m->nbody number of bodies  (index 0 = world)
 *   nG  = m->ngeom number of geoms
 *   nS  = m->nsite number of sites
 *   nM  = m->nM    number of non-zeros in sparse inertia matrix
 *   All mjtNum arrays are double (or float in single-precision builds).
 */

#pragma once
#include <mujoco/mujoco.h>

// ============================================================
// 0. LIFECYCLE – load model, allocate / copy / free data
// ============================================================

// Load MJCF / URDF XML.  Returns heap-allocated mjModel*.
//   in : filename (path), vfs (virtual FS, nullable), error buf
//   out: mjModel*  (call mj_deleteModel when done)
mjModel* mj_loadXML(const char* filename,
                    const mjVFS* vfs,        // nullable
                    char* error, int error_sz);

// Allocate mjData for a model (all state zeroed except qpos=qpos0).
//   in : m
//   out: mjData*  (call mj_deleteData when done)
mjData* mj_makeData(const mjModel* m);

// Deep-copy mjData (essential for rollout branching).
//   in : dest (nullable → allocates), m, src
//   out: mjData* pointing at dest
mjData* mj_copyData(mjData* dest, const mjModel* m, const mjData* src);

// Reset mjData to model defaults (qpos = qpos0, everything else = 0).
void mj_resetData(const mjModel* m, mjData* d);

// Reset to a named keyframe stored in the model.
//   key: 0-based index into m->key_*  arrays
void mj_resetDataKeyframe(const mjModel* m, mjData* d, int key);

// Free model / data.
void mj_deleteModel(mjModel* m);
void mj_deleteData(mjData* d);


// ============================================================
// 1. STATE MANAGEMENT
// ============================================================

// Signature for the bitmask `sig` is a combination of mjtState flags:
//   mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT   (= mjSTATE_PHYSICS)
//   mjSTATE_CTRL | mjSTATE_QFRC_APPLIED | ...   (= mjSTATE_USER)

// Return byte size of a state vector described by sig.
int mj_stateSize(const mjModel* m, int sig);

// Serialize selected state components into a flat mjtNum array.
//   out: state[mj_stateSize(m, sig)]
void mj_getState(const mjModel* m, const mjData* d,
                 mjtNum* state, int sig);

// Deserialize back into mjData.
void mj_setState(const mjModel* m, mjData* d,
                 const mjtNum* state, int sig);

// Copy state between two mjData objects.
void mj_copyState(const mjModel* m,
                  const mjData* src, mjData* dst, int sig);


// ============================================================
// 2. FORWARD / INVERSE DYNAMICS – full pipeline
// ============================================================

// Full forward dynamics (no time integration).
// Populates ALL position/velocity/acceleration arrays in mjData.
// Call sequence: sets xpos, xmat, xquat, geom_xpos, site_xpos,
//                cdof, cinert, qM, qLD, qfrc_bias, qacc, contacts, …
void mj_forward(const mjModel* m, mjData* d);

// Step simulation one dt (calls mj_forward then integrates qpos/qvel).
void mj_step(const mjModel* m, mjData* d);

// Split step: mj_step1 runs up to control callback, mj_step2 after.
// Use to inject d->ctrl / d->qfrc_applied between them.
void mj_step1(const mjModel* m, mjData* d);
void mj_step2(const mjModel* m, mjData* d);

// Forward with skip: avoid re-running stages already valid.
//   skipstage: mjSTAGE_NONE / mjSTAGE_POS / mjSTAGE_VEL
void mj_forwardSkip(const mjModel* m, mjData* d,
                    int skipstage, int skipsensor);

// Inverse dynamics.  Must set d->qacc before calling.
//   out: d->qfrc_inverse  (nv)  — joint torques to produce qacc
void mj_inverse(const mjModel* m, mjData* d);
void mj_inverseSkip(const mjModel* m, mjData* d,
                    int skipstage, int skipsensor);


// ============================================================
// 3. KINEMATICS STAGES (sub-steps of mj_forward)
// ============================================================

// Run all kinematics-like computations (positions, CoM, tendons).
// Populates: xpos, xquat, xmat, xipos, geom_xpos, site_xpos, cdof, cinert
void mj_fwdKinematics(const mjModel* m, mjData* d);

// Alias: run only the rigid-body kinematics pass.
void mj_kinematics(const mjModel* m, mjData* d);

// Map inertias/dofs to CoM-centred global frame.
// Required before Jacobians, mj_crb, mj_rne.
void mj_comPos(const mjModel* m, mjData* d);

// Compute body CoM velocities cvel[nb×6] and cdof_dot[nv×6].
// Required before velocity Jacobians and mj_subtreeVel.
void mj_comVel(const mjModel* m, mjData* d);

// Compute subtree linear velocity and angular momentum.
// Required before mj_jacSubtreeCom.
void mj_subtreeVel(const mjModel* m, mjData* d);

// Compute tendon lengths, velocities and moment arms.
void mj_tendon(const mjModel* m, mjData* d);

// Compute actuator transmission lengths and moment arms.
void mj_transmission(const mjModel* m, mjData* d);

// Transform local body pose to global Cartesian frame.
//   in : pos[3], quat[4] in body-local frame, body id, sameframe flag
//   out: xpos[3], xmat[9] in world frame
void mj_local2Global(mjData* d,
                     mjtNum xpos[3], mjtNum xmat[9],
                     const mjtNum pos[3], const mjtNum quat[4],
                     int body, mjtByte sameframe);


// ============================================================
// 4. JACOBIANS
// ============================================================
// All Jacobians have shape 3×nv (jacp = translational, jacr = rotational).
// Either jacp or jacr may be NULL to skip that part.
// Prerequisites: mj_kinematics + mj_comPos must have been called.

// Jacobian of an arbitrary world-space point attached to body.
//   in : point[3] (world coords), body id
//   out: jacp[3×nv], jacr[3×nv]
void mj_jac(const mjModel* m, const mjData* d,
            mjtNum* jacp, mjtNum* jacr,
            const mjtNum point[3], int body);

// Jacobian at body frame origin.
void mj_jacBody(const mjModel* m, const mjData* d,
                mjtNum* jacp, mjtNum* jacr, int body);

// Jacobian at body centre of mass.
void mj_jacBodyCom(const mjModel* m, const mjData* d,
                   mjtNum* jacp, mjtNum* jacr, int body);

// Translational Jacobian at subtree CoM (rotation not available).
//   Requires mj_subtreeVel to have been called.
void mj_jacSubtreeCom(const mjModel* m, mjData* d,
                      mjtNum* jacp, int body);

// Jacobian at geom centre.
void mj_jacGeom(const mjModel* m, const mjData* d,
                mjtNum* jacp, mjtNum* jacr, int geom);

// Jacobian at site position.
void mj_jacSite(const mjModel* m, const mjData* d,
                mjtNum* jacp, mjtNum* jacr, int site);

// Jacobian of a point AND an axis direction (useful for 5-DOF IK).
//   in : point[3] (world), axis[3] (world), body id
//   out: jacPoint[3×nv], jacAxis[3×nv]
void mj_jacPointAxis(const mjModel* m, mjData* d,
                     mjtNum* jacPoint, mjtNum* jacAxis,
                     const mjtNum point[3], const mjtNum axis[3],
                     int body);

// Time derivative of the translational/rotational Jacobian (Jdot).
//   out: jacp[3×nv], jacr[3×nv]
void mj_jacDot(const mjModel* m, const mjData* d,
               mjtNum* jacp, mjtNum* jacr,
               const mjtNum point[3], int body);

// Subtree angular-momentum matrix L(q), shape nv×3.
//   d(H_subtree)/dt = L * qacc  where H is angular momentum.
void mj_angmomMat(const mjModel* m, mjData* d,
                  mjtNum* mat,   // nv × 3
                  int body);

// Multiply constraint Jacobian J (nefc×nv) by a vector.
//   res = J  * vec  (res: nefc)
void mj_mulJacVec(const mjModel* m, const mjData* d,
                  mjtNum* res, const mjtNum* vec);

//   res = J' * vec  (res: nv)
void mj_mulJacTVec(const mjModel* m, const mjData* d,
                   mjtNum* res, const mjtNum* vec);


// ============================================================
// 5. INERTIA / DYNAMICS
// ============================================================

// Composite Rigid Body (CRB) algorithm — fills d->qM (sparse, nM).
void mj_crb(const mjModel* m, mjData* d);

// Build mass matrix (fills d->qM from d->crb).
void mj_makeM(const mjModel* m, mjData* d);

// Expand sparse qM to a full nv×nv dense matrix.
//   in : M (= d->qM, length nM)
//   out: dst[nv×nv]
void mj_fullM(const mjModel* m, mjtNum* dst, const mjtNum* M);

// L'DL Cholesky factorisation of M into d->qLD, d->qLDiagInv.
// Must call mj_crb / mj_makeM first.
void mj_factorM(const mjModel* m, mjData* d);

// Solve M * x = y using the pre-computed factorisation.
//   in : y[nv × n]
//   out: x[nv × n]  (n column vectors solved simultaneously)
void mj_solveM(const mjModel* m, mjData* d,
               mjtNum* x, const mjtNum* y, int n);

// Half-solve: x = sqrt(inv(D)) * inv(L') * y
void mj_solveM2(const mjModel* m, mjData* d,
                mjtNum* x, const mjtNum* y,
                const mjtNum* sqrtInvD, int n);

// Multiply M (sparse) by a vector: res = M * vec.
void mj_mulM(const mjModel* m, const mjData* d,
             mjtNum* res, const mjtNum* vec);

// Multiply M^(1/2) (Cholesky factor) by a vector.
void mj_mulM2(const mjModel* m, const mjData* d,
              mjtNum* res, const mjtNum* vec);

// Add inertia matrix to dst (lower triangle; dst can be sparse or dense).
void mj_addM(const mjModel* m, mjData* d,
             mjtNum* dst,
             int* rownnz, int* rowadr, int* colind); // nullable → dense

// Recursive Newton-Euler (RNE).
//   flg_acc=1 → result = M(q)*qacc + C(q,v)   [needs d->qacc]
//   flg_acc=0 → result = C(q,v)                [bias / Coriolis+gravity]
//   out: result[nv]
void mj_rne(const mjModel* m, mjData* d,
            int flg_acc, mjtNum* result);

// RNE post-constraint: computes cacc, cfrc_ext, cfrc_int.
void mj_rnePostConstraint(const mjModel* m, mjData* d);

// Key mjData fields populated after mj_forward / mj_fwdVelocity:
//   d->qfrc_bias     [nv]  — C(q,v): Coriolis + gravity
//   d->qM            [nM]  — sparse inertia matrix
//   d->qLD           [nM]  — L'DL factorisation
//   d->qacc          [nv]  — joint acceleration (output of fwd dynamics)
//   d->qfrc_actuator [nv]  — force from actuators
//   d->qfrc_passive  [nv]  — spring / damper / gravity-comp forces
//   d->qfrc_inverse  [nv]  — inverse dynamics result


// ============================================================
// 6. COLLISION DETECTION
// ============================================================

// Run the full broadphase + narrowphase collision pipeline.
//   out: d->ncon, d->contact[0..ncon-1]
void mj_collision(const mjModel* m, mjData* d);

// Signed distance between two specific geoms (no full pipeline needed).
//   in : geom1, geom2 ids; distmax (ignore if further away)
//   out: fromto[6] = [point_on_geom1[3], point_on_geom2[3]]  (nullable)
//   ret: signed distance (negative = penetration), or distmax if too far
mjtNum mj_geomDistance(const mjModel* m, mjData* d,
                       int geom1, int geom2,
                       mjtNum distmax, mjtNum fromto[6]);

// Extract contact force/torque for a detected contact.
//   in : id (index into d->contact[])
//   out: result[6] = [fx,fy,fz, tx,ty,tz] in contact frame
void mj_contactForce(const mjModel* m, const mjData* d,
                     int id, mjtNum result[6]);

// Encode contact force into pyramidal cone representation.
//   in : force[dim], mu[dim/2], dim (condim: 3/4/6)
//   out: pyramid[2*dim]
void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force,
                       const mjtNum* mu, int dim);

// Decode pyramidal representation back to force.
void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid,
                       const mjtNum* mu, int dim);

// Key mjContact fields (accessed via d->contact[i]):
//   .dist        — signed distance (neg = penetration)
//   .pos[3]      — contact point midpoint in world frame
//   .frame[9]    — contact frame; normal = frame[0:3] (geom0 → geom1)
//   .geom[2]     — geom ids
//   .dim         — constraint dimension (1/3/4/6)
//   .friction[5] — tangent1, tangent2, spin, roll1, roll2
//   .efc_address — row in efc_J where this contact starts (-1 if excluded)


// ============================================================
// 7. RAY CASTING
// ============================================================

// Intersect a ray with all visible geoms.
//   in : pnt[3] ray origin, vec[3] direction (need not be unit)
//        geomgroup bitmask (nullable = no filter)
//        flg_static: include static geoms?
//        bodyexclude: skip geoms of this body (-1 = none)
//   out: geomid[1] — id of nearest hit geom (nullable)
//        normal[3] — surface normal at hit point (nullable)
//   ret: distance along ray, or -1 if no intersection
mjtNum mj_ray(const mjModel* m, const mjData* d,
              const mjtNum pnt[3], const mjtNum vec[3],
              const mjtByte* geomgroup,   // nullable
              mjtByte flg_static,
              int bodyexclude,
              int geomid[1],              // nullable
              mjtNum normal[3]);          // nullable

// Batch version: cast nray rays from the same point.
//   in : vec[nray×3], cutoff (ignore hits beyond this distance)
//   out: geomid[nray], dist[nray], normal[nray×3]
void mj_multiRay(const mjModel* m, mjData* d,
                 const mjtNum pnt[3], const mjtNum* vec,
                 const mjtByte* geomgroup, mjtByte flg_static,
                 int bodyexclude,
                 int* geomid, mjtNum* dist, mjtNum* normal,
                 int nray, mjtNum cutoff);

// Ray vs. height-field geom (faster than mj_ray for a single hfield).
mjtNum mj_rayHfield(const mjModel* m, const mjData* d,
                    int geomid, const mjtNum pnt[3],
                    const mjtNum vec[3], mjtNum normal[3]);

// Ray vs. mesh geom.
mjtNum mj_rayMesh(const mjModel* m, const mjData* d,
                  int geomid, const mjtNum pnt[3],
                  const mjtNum vec[3], mjtNum normal[3]);

// Ray vs. arbitrary primitive geom (no mjData needed).
//   in : pos[3], mat[9] geom pose; size[3] geom half-extents; geomtype
mjtNum mju_rayGeom(const mjtNum pos[3], const mjtNum mat[9],
                   const mjtNum size[3],
                   const mjtNum pnt[3], const mjtNum vec[3],
                   int geomtype, mjtNum normal[3]);


// ============================================================
// 8. OBJECT VELOCITY / ACCELERATION
// ============================================================

// 6D velocity [angular(3); linear(3)] in object-centred frame.
//   objtype: mjOBJ_BODY / mjOBJ_GEOM / mjOBJ_SITE / …
//   flg_local: 1 = local orientation, 0 = world orientation
//   out: res[6]
void mj_objectVelocity(const mjModel* m, const mjData* d,
                       int objtype, int objid,
                       mjtNum res[6], int flg_local);

// 6D acceleration [angular; linear].
void mj_objectAcceleration(const mjModel* m, const mjData* d,
                            int objtype, int objid,
                            mjtNum res[6], int flg_local);


// ============================================================
// 9. APPLYING EXTERNAL FORCES
// ============================================================

// Apply Cartesian force+torque at a point on a body.
// Adds the equivalent generalised force into qfrc_target.
//   in : force[3], torque[3] (either nullable), point[3] world, body id
//   out: qfrc_target[nv]  (modified in-place)
void mj_applyFT(const mjModel* m, mjData* d,
                const mjtNum force[3],   // nullable
                const mjtNum torque[3],  // nullable
                const mjtNum point[3], int body,
                mjtNum* qfrc_target);    // typically d->qfrc_applied

// Direct fields (set before mj_forward):
//   d->ctrl[nu]            — actuator control inputs
//   d->qfrc_applied[nv]    — generalised external forces
//   d->xfrc_applied[nb×6]  — Cartesian force/torque on each body


// ============================================================
// 10. POSITION INTEGRATION & DIFFERENTIATION
// ============================================================

// Velocity = (qpos2 - qpos1) / dt in the tangent space (handles quats).
//   out: qvel[nv]
void mj_differentiatePos(const mjModel* m,
                          mjtNum* qvel, mjtNum dt,
                          const mjtNum* qpos1,
                          const mjtNum* qpos2);

// Integrate: qpos += qvel * dt  (quaternion-aware).
void mj_integratePos(const mjModel* m,
                     mjtNum* qpos, const mjtNum* qvel, mjtNum dt);

// Re-normalise all quaternions in a qpos vector.
void mj_normalizeQuat(const mjModel* m, mjtNum* qpos);


// ============================================================
// 11. FINITE-DIFFERENCE DERIVATIVES
// ============================================================

// State-space transition matrices for control theory / LQR / MPC.
//   d(x_next) = A*dx + B*du
//   d(sensor) = C*dx + D*du
//   x = [qpos(nq); qvel(nv); act(na)]  (size = 2*nv+na)
//   dims: A[2nv+na × 2nv+na], B[2nv+na × nu],
//         C[nsensordata × 2nv+na], D[nsensordata × nu]
//   All output matrices are nullable.
void mjd_transitionFD(const mjModel* m, mjData* d,
                      mjtNum eps, mjtByte flg_centered,
                      mjtNum* A, mjtNum* B,   // nullable
                      mjtNum* C, mjtNum* D);  // nullable

// Jacobians of inverse dynamics w.r.t. q, v, a.
//   Outputs (all nullable, transposed from control-theory convention):
//     DfDq[nv×nv], DfDv[nv×nv], DfDa[nv×nv]
//     DsDq[nv×nsensordata], DsDv, DsDa
//     DmDq[nv×nM]
void mjd_inverseFD(const mjModel* m, mjData* d,
                   mjtNum eps, mjtByte flg_actuation,
                   mjtNum* DfDq, mjtNum* DfDv, mjtNum* DfDa,
                   mjtNum* DsDq, mjtNum* DsDv, mjtNum* DsDa,
                   mjtNum* DmDq);

// Derivatives of mju_subQuat (quaternion difference).
//   Da[9], Db[9]  — Jacobians w.r.t. qa and qb
void mjd_subQuat(const mjtNum qa[4], const mjtNum qb[4],
                 mjtNum Da[9], mjtNum Db[9]);

// Derivatives of mju_quatIntegrate.
void mjd_quatIntegrate(const mjtNum vel[3], mjtNum scale,
                       mjtNum Dquat[9], mjtNum Dvel[9],
                       mjtNum Dscale[3]);


// ============================================================
// 12. SIGNED DISTANCE FUNCTIONS (SDF)
// ============================================================

// Evaluate SDF value at world point x for a given geom.
//   in : s (mjSDF descriptor obtained via mjc_getSDF), x[3]
//   ret: signed distance (negative inside)
mjtNum mjc_distance(const mjModel* m, const mjData* d,
                    const mjSDF* s, const mjtNum x[3]);

// Gradient of the SDF (points outward).
//   out: gradient[3]
void mjc_gradient(const mjModel* m, const mjData* d,
                  const mjSDF* s, mjtNum gradient[3],
                  const mjtNum x[3]);

// Get the SDF plugin for a given geom id.
const mjpPlugin* mjc_getSDF(const mjModel* m, int id);


// ============================================================
// 13. QUATERNION UTILITIES
// ============================================================
// Convention: [w, x, y, z]

// Rotate vector by quaternion: res = q * vec * q^-1
void mju_rotVecQuat(mjtNum res[3],
                    const mjtNum vec[3], const mjtNum quat[4]);

// Quaternion inverse / conjugate.
void mju_negQuat(mjtNum res[4], const mjtNum quat[4]);

// Quaternion product: res = q1 * q2
void mju_mulQuat(mjtNum res[4],
                 const mjtNum quat1[4], const mjtNum quat2[4]);

// Quaternion * axis (rotate q by axis-angle): res = q * exp(axis/2)
void mju_mulQuatAxis(mjtNum res[4],
                     const mjtNum quat[4], const mjtNum axis[3]);

// axis-angle → quaternion.
void mju_axisAngle2Quat(mjtNum res[4],
                        const mjtNum axis[3], mjtNum angle);

// Quaternion difference → 3D angular velocity:
//   res = 2 * log(quat) / dt
void mju_quat2Vel(mjtNum res[3], const mjtNum quat[4], mjtNum dt);

// Orientation error as 3D angular velocity: qb * q(res) = qa
//   (i.e. res is the axis-angle to rotate from qb to qa)
void mju_subQuat(mjtNum res[3],
                 const mjtNum qa[4], const mjtNum qb[4]);

// Quaternion → 3×3 rotation matrix (row-major, 9 elements).
void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]);

// 3×3 rotation matrix → quaternion.
void mju_mat2Quat(mjtNum quat[4], const mjtNum mat[9]);

// Quaternion time derivative given angular velocity.
//   dquat/dt = 0.5 * [0; omega] * quat
void mju_derivQuat(mjtNum res[4],
                   const mjtNum quat[4], const mjtNum vel[3]);

// Integrate quaternion: quat += vel * scale
void mju_quatIntegrate(mjtNum quat[4],
                       const mjtNum vel[3], mjtNum scale);

// Build rotation that takes z-axis to vec.
void mju_quatZ2Vec(mjtNum quat[4], const mjtNum vec[3]);

// Extract rotation quaternion from arbitrary 3×3 matrix.
//   Refines the input quaternion iteratively; returns iteration count.
int mju_mat2Rot(mjtNum quat[4], const mjtNum mat[9]);

// Euler angles (rad, intrinsic/extrinsic per seq string) → quaternion.
//   seq: 3-char string from "xyzXYZ" (lower=intrinsic, upper=extrinsic)
void mju_euler2Quat(mjtNum quat[4],
                    const mjtNum euler[3], const char* seq);

// Normalize to unit quaternion; return original length.
mjtNum mju_normalize4(mjtNum vec[4]);


// ============================================================
// 14. POSE UTILITIES
// ============================================================

// Compose two poses: T_res = T1 * T2
void mju_mulPose(mjtNum posres[3], mjtNum quatres[4],
                 const mjtNum pos1[3], const mjtNum quat1[4],
                 const mjtNum pos2[3], const mjtNum quat2[4]);

// Invert a pose: T_res = T^(-1)
void mju_negPose(mjtNum posres[3], mjtNum quatres[4],
                 const mjtNum pos[3], const mjtNum quat[4]);

// Transform a vector by a pose: res = pos + quat * vec
void mju_trnVecPose(mjtNum res[3],
                    const mjtNum pos[3], const mjtNum quat[4],
                    const mjtNum vec[3]);


// ============================================================
// 15. VECTOR / MATRIX UTILITIES
// ============================================================

// --- Fixed-size 3-vectors ---
void   mju_zero3(mjtNum res[3]);
void   mju_copy3(mjtNum res[3], const mjtNum data[3]);
void   mju_scl3 (mjtNum res[3], const mjtNum vec[3], mjtNum scl);
void   mju_add3 (mjtNum res[3], const mjtNum a[3], const mjtNum b[3]);
void   mju_sub3 (mjtNum res[3], const mjtNum a[3], const mjtNum b[3]);
void   mju_addToScl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl);
void   mju_cross (mjtNum res[3], const mjtNum a[3], const mjtNum b[3]);
mjtNum mju_dot3  (const mjtNum a[3], const mjtNum b[3]);
mjtNum mju_norm3 (const mjtNum vec[3]);
mjtNum mju_dist3 (const mjtNum a[3], const mjtNum b[3]);
mjtNum mju_normalize3(mjtNum vec[3]);           // normalises in-place, returns old length

// 3×3 matrix × vector.
void mju_mulMatVec3 (mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);
void mju_mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]);

// --- Generic n-vectors ---
void   mju_zero  (mjtNum* res, int n);
void   mju_fill  (mjtNum* res, mjtNum val, int n);
void   mju_copy  (mjtNum* res, const mjtNum* vec, int n);
void   mju_scl   (mjtNum* res, const mjtNum* vec, mjtNum scl, int n);
void   mju_add   (mjtNum* res, const mjtNum* a, const mjtNum* b, int n);
void   mju_sub   (mjtNum* res, const mjtNum* a, const mjtNum* b, int n);
void   mju_addTo (mjtNum* res, const mjtNum* vec, int n);
void   mju_addToScl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n);
void   mju_addScl(mjtNum* res, const mjtNum* a, const mjtNum* b, mjtNum scl, int n);
mjtNum mju_dot   (const mjtNum* a, const mjtNum* b, int n);
mjtNum mju_norm  (const mjtNum* vec, int n);
mjtNum mju_normalize(mjtNum* res, int n);
mjtNum mju_sum   (const mjtNum* vec, int n);
mjtNum mju_L1    (const mjtNum* vec, int n);   // L1 norm = sum(|x|)

// --- Dense matrices (row-major) ---
void   mju_transpose(mjtNum* res, const mjtNum* mat, int nr, int nc);
void   mju_symmetrize(mjtNum* res, const mjtNum* mat, int n);
void   mju_eye  (mjtNum* mat, int n);
void   mju_mulMatVec (mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);
void   mju_mulMatTVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int nr, int nc);
mjtNum mju_mulVecMatVec(const mjtNum* v1, const mjtNum* mat, const mjtNum* v2, int n);
void   mju_mulMatMat (mjtNum* res, const mjtNum* A, const mjtNum* B, int r1, int c1, int c2);
void   mju_mulMatMatT(mjtNum* res, const mjtNum* A, const mjtNum* B, int r1, int c1, int r2);
void   mju_mulMatTMat(mjtNum* res, const mjtNum* A, const mjtNum* B, int r1, int c1, int c2);
// res = mat' * diag * mat  (or mat'*mat if diag==NULL)
void   mju_sqrMatTD(mjtNum* res, const mjtNum* mat, const mjtNum* diag, int nr, int nc);

// Spatial (6D) vector transform: res = X * vec
//   rotnew2old: 3×3 rotation (nullable = identity), flg_force: motion(0)/force(1)
void mju_transformSpatial(mjtNum res[6], const mjtNum vec[6], int flg_force,
                          const mjtNum newpos[3], const mjtNum oldpos[3],
                          const mjtNum rotnew2old[9]);

// --- Sparse conversions ---
int  mju_dense2sparse(mjtNum* res, const mjtNum* mat, int nr, int nc,
                      int* rownnz, int* rowadr, int* colind, int nnz);
void mju_sparse2dense(mjtNum* res, const mjtNum* mat, int nr, int nc,
                      const int* rownnz, const int* rowadr, const int* colind);


// ============================================================
// 16. LINEAR ALGEBRA – DECOMPOSITIONS
// ============================================================

// Cholesky factorisation: mat = L*L'  (in-place, lower triangle).
//   ret: numerical rank
int  mju_cholFactor(mjtNum* mat, int n, mjtNum mindiag);

// Solve (L*L') * res = vec  using Cholesky factor.
void mju_cholSolve(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n);

// Rank-1 Cholesky update: L*L' ± x*x'.
int  mju_cholUpdate(mjtNum* mat, mjtNum* x, int n, int flg_plus);

// Band-diagonal Cholesky (for banded systems, e.g. trajectory opt).
mjtNum mju_cholFactorBand(mjtNum* mat, int ntotal, int nband, int ndense,
                          mjtNum diagadd, mjtNum diagmul);
void   mju_cholSolveBand (mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                          int ntotal, int nband, int ndense);

// 3×3 symmetric eigenvalue decomposition.
//   mat = eigvec * diag(eigval) * eigvec'
//   out: eigval[3], eigvec[9], quat[4] (rotation equivalent to eigvec)
int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4],
             const mjtNum mat[9]);


// ============================================================
// 17. BOX-CONSTRAINED QP SOLVER
// ============================================================

// Solve: min 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper
//   in : H[n×n] (SPD), g[n], lower[n], upper[n] (nullable)
//        res[n] used as warm-start
//   out: res[n] solution,  R[n*(n+7)] subspace Cholesky factor
//        index[n] free-dimension indices (nullable)
//   ret: rank of unconstrained subspace, or -1 on failure
//   Use case: constrained IK, CBF QPs, trajectory optimisation steps.
int mju_boxQP(mjtNum* res, mjtNum* R, int* index,
              const mjtNum* H, const mjtNum* g, int n,
              const mjtNum* lower, const mjtNum* upper);

// Convenience allocator for mju_boxQP buffers (free with mju_free).
void mju_boxQPmalloc(mjtNum** res, mjtNum** R, int** index,
                     mjtNum** H, mjtNum** g, int n,
                     mjtNum** lower, mjtNum** upper);


// ============================================================
// 18. OBJECT LOOKUP
// ============================================================

// Get object id by name.
//   type: mjOBJ_BODY / mjOBJ_GEOM / mjOBJ_SITE / mjOBJ_JOINT / …
//   ret: id, or -1 if not found
int mj_name2id(const mjModel* m, int type, const char* name);

// Get object name by id.  Returns NULL if not found.
const char* mj_id2name(const mjModel* m, int type, int id);


// ============================================================
// 19. ENERGY
// ============================================================

// Compute potential energy → d->energy[0]  (requires mj_fwdPosition)
void mj_energyPos(const mjModel* m, mjData* d);

// Compute kinetic energy  → d->energy[1]   (requires mj_fwdVelocity)
void mj_energyVel(const mjModel* m, mjData* d);


// ============================================================
// 20. SCALAR UTILITIES
// ============================================================

mjtNum mju_min(mjtNum a, mjtNum b);
mjtNum mju_max(mjtNum a, mjtNum b);
mjtNum mju_clip(mjtNum x, mjtNum min, mjtNum max);
mjtNum mju_sign(mjtNum x);
int    mju_round(mjtNum x);
int    mju_isBad(mjtNum x);   // 1 if NaN or |x| > mjMAXVAL
int    mju_isZero(const mjtNum* vec, int n);
mjtNum mju_sigmoid(mjtNum x); // quintic sigmoid on [0,1]
mjtNum mju_standardNormal(mjtNum* num2); // standard normal sample
mjtNum mju_Halton(int index, int base);  // quasi-random Halton sequence


// ============================================================
// 21. KEY mjData FIELDS QUICK REFERENCE
// ============================================================
/*
  STATE
    d->qpos  [nq]      joint positions (may include quaternion components)
    d->qvel  [nv]      joint velocities (always in tangent space)
    d->qacc  [nv]      joint accelerations
    d->ctrl  [nu]      control inputs
    d->act   [na]      actuator activations

  KINEMATICS (populated after mj_kinematics / mj_fwdKinematics)
    d->xpos  [nb×3]    body frame origins in world
    d->xquat [nb×4]    body frame quaternions
    d->xmat  [nb×9]    body frame rotation matrices
    d->xipos [nb×3]    body CoM positions
    d->geom_xpos [nG×3]  geom positions
    d->geom_xmat [nG×9]  geom orientations
    d->site_xpos [nS×3]  site positions
    d->site_xmat [nS×9]  site orientations
    d->xanchor   [nJ×3]  joint anchor positions
    d->xaxis     [nJ×3]  joint axes

  INERTIA (populated after mj_crb / mj_makeM)
    d->qM    [nM]     sparse inertia matrix
    d->qLD   [nM]     L'DL factorisation
    d->crb   [nb×10]  com-based composite inertia (mass, inertia, CoM)

  DYNAMICS
    d->qfrc_bias     [nv]  Coriolis + gravity = C(q,v)
    d->qfrc_actuator [nv]  actuator generalised forces
    d->qfrc_passive  [nv]  spring + damper + gravity-comp
    d->qfrc_applied  [nv]  user-set external generalised forces
    d->xfrc_applied  [nb×6]  user-set Cartesian forces (force:torque)
    d->qfrc_inverse  [nv]  result of mj_inverse

  CONTACTS
    d->ncon            number of active contacts
    d->contact[ncon]   array of mjContact structs

  CONSTRAINTS
    d->nefc            total number of constraint rows
    d->efc_J [nefc×nv] constraint Jacobian (sparse)
    d->efc_force[nefc] constraint forces

  ENERGY
    d->energy[0]  potential energy
    d->energy[1]  kinetic energy
*/
