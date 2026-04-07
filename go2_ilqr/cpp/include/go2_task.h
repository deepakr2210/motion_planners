/**
 * go2_task.h  –  GO2-specific iLQR cost functions
 *
 * State layout (floating-base MuJoCo model):
 *   qpos[0:3]   base position  (x, y, z)
 *   qpos[3:7]   base quaternion (w, x, y, z)
 *   qpos[7:19]  joint angles   (FR_hip, FR_thigh, FR_calf,
 *                                FL_hip, FL_thigh, FL_calf,
 *                                RR_hip, RR_thigh, RR_calf,
 *                                RL_hip, RL_thigh, RL_calf)
 *   qvel[0:3]   base linear velocity  (world frame)
 *   qvel[3:6]   base angular velocity (body frame)
 *   qvel[6:18]  joint velocities
 *
 * Tangent space (ndx = 36 = 2*nv):
 *   dx[0:3]    base position perturbation
 *   dx[3:6]    base orientation perturbation (rotation vector, body frame)
 *   dx[6:18]   joint angle perturbation
 *   dx[18:21]  base linear-velocity perturbation
 *   dx[21:24]  base angular-velocity perturbation
 *   dx[24:36]  joint-velocity perturbation
 *
 * Cost structure (Gauss-Newton):
 *   l(x, u) = 0.5 Σ_i  w_i ||r_i(x, u)||²
 *
 * This guarantees PSD Hessians  lxx = Σ J_i^T W_i J_i.
 */

#pragma once

#include <array>
#include <memory>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>

#include "ilqr.h"

namespace go2_ilqr {

// ---------------------------------------------------------------------------
// GO2 constants
// ---------------------------------------------------------------------------

/// Default joint angles for standing pose (12 DOF, FR/FL/RR/RL order)
static const Eigen::Matrix<double, 12, 1> GO2_DEFAULT_QPOS =
    (Eigen::Matrix<double, 12, 1>() <<
        0.0,  0.9, -1.8,   // FR: hip, thigh, calf
        0.0,  0.9, -1.8,   // FL
        0.0,  0.9, -1.8,   // RR
        0.0,  0.9, -1.8    // RL
    ).finished();

static constexpr double GO2_STANDING_HEIGHT = 0.27;  ///< trunk CoM [m]

// ---------------------------------------------------------------------------
// Quaternion utilities
// ---------------------------------------------------------------------------

/**
 * Orientation error as a 3-D rotation vector.
 *
 *   err = 2 * vec( q_ref^{-1} ⊗ q )
 *
 * For small errors this equals the axis-angle of the relative rotation and
 * corresponds directly to the orientation components of the tangent state.
 *
 * @param q      current quaternion (w, x, y, z)
 * @param q_ref  reference quaternion (w, x, y, z)
 * @return 3-D rotation-vector error
 */
Eigen::Vector3d QuatError(const Eigen::Vector4d& q,
                           const Eigen::Vector4d& q_ref);

// ---------------------------------------------------------------------------
// StandingTask
// ---------------------------------------------------------------------------

struct StandingTaskConfig {
    double w_height    = 200.0;  ///< base height
    double w_orient    = 100.0;  ///< base orientation
    double w_linvel    =  10.0;  ///< base linear velocity
    double w_angvel    =  10.0;  ///< base angular velocity
    double w_joint_pos =   5.0;  ///< joint angles vs. default
    double w_joint_vel =   1.0;  ///< joint velocities
    double w_ctrl      =   0.001;///< control effort
    double w_terminal  =  10.0;  ///< terminal cost multiplier

    double target_height = GO2_STANDING_HEIGHT;
    Eigen::Matrix<double, 12, 1> target_joint_pos = GO2_DEFAULT_QPOS;
};


class StandingTask : public CostFunction {
public:
    StandingTask(const mjModel*           model,
                 const StandingTaskConfig& cfg = StandingTaskConfig{});

    double RunningCost(
        const Eigen::VectorXd& x, const Eigen::VectorXd& u, int t) const override;

    void RunningCostDerivatives(
        const Eigen::VectorXd& x, const Eigen::VectorXd& u, int t,
        Eigen::VectorXd& lx, Eigen::VectorXd& lu,
        Eigen::MatrixXd& lxx, Eigen::MatrixXd& luu,
        Eigen::MatrixXd& lux) const override;

    double TerminalCost(const Eigen::VectorXd& x) const override;

    void TerminalCostDerivatives(
        const Eigen::VectorXd& x,
        Eigen::VectorXd& lx,
        Eigen::MatrixXd& lxx) const override;

private:
    void ComputeResiduals(const Eigen::VectorXd& x,
                          const Eigen::VectorXd& u,
                          double&   cost,
                          Eigen::VectorXd& lx,
                          Eigen::VectorXd& lu,
                          bool terminal) const;

    const mjModel*     model_;
    StandingTaskConfig cfg_;
    int nq_, nv_, nu_, ndx_;

    // Pre-computed constant Jacobians (w.r.t. tangent state)
    Eigen::MatrixXd Jh_;   // (1,  36)  height
    Eigen::MatrixXd Jo_;   // (3,  36)  orientation
    Eigen::MatrixXd Jv_;   // (3,  36)  linear velocity
    Eigen::MatrixXd Jw_;   // (3,  36)  angular velocity
    Eigen::MatrixXd Jqp_;  // (12, 36)  joint positions
    Eigen::MatrixXd Jqv_;  // (12, 36)  joint velocities
    Eigen::MatrixXd Ju_;   // (12, 12)  control

    // Pre-computed constant Hessians
    Eigen::MatrixXd lxx_running_;
    Eigen::MatrixXd luu_running_;
    Eigen::MatrixXd lxx_terminal_;

    Eigen::Vector4d q_ref_;  // identity quaternion
};


// ---------------------------------------------------------------------------
// VelocityTask
// ---------------------------------------------------------------------------

struct VelocityTaskConfig {
    double target_vx   = 0.5;   ///< desired forward velocity  [m/s]
    double target_vy   = 0.0;   ///< desired lateral velocity  [m/s]
    double target_yaw  = 0.0;   ///< desired yaw rate          [rad/s]

    double target_height = GO2_STANDING_HEIGHT;
    Eigen::Matrix<double, 12, 1> target_joint_pos = GO2_DEFAULT_QPOS;

    double w_height    = 150.0;
    double w_orient    =  50.0;
    double w_vel_xy    = 100.0;  ///< vx / vy tracking
    double w_yaw       =  50.0;  ///< yaw-rate tracking
    double w_vz        =  20.0;  ///< penalise vertical velocity
    double w_angvel_xy =   5.0;  ///< roll / pitch rate
    double w_joint_pos =   2.0;
    double w_joint_vel =   0.5;
    double w_ctrl      =   0.001;
    double w_terminal  =   5.0;
};


class VelocityTask : public CostFunction {
public:
    VelocityTask(const mjModel*           model,
                 const VelocityTaskConfig& cfg = VelocityTaskConfig{});

    double RunningCost(
        const Eigen::VectorXd& x, const Eigen::VectorXd& u, int t) const override;

    void RunningCostDerivatives(
        const Eigen::VectorXd& x, const Eigen::VectorXd& u, int t,
        Eigen::VectorXd& lx, Eigen::VectorXd& lu,
        Eigen::MatrixXd& lxx, Eigen::MatrixXd& luu,
        Eigen::MatrixXd& lux) const override;

    double TerminalCost(const Eigen::VectorXd& x) const override;

    void TerminalCostDerivatives(
        const Eigen::VectorXd& x,
        Eigen::VectorXd& lx,
        Eigen::MatrixXd& lxx) const override;

private:
    void ComputeResiduals(const Eigen::VectorXd& x,
                          const Eigen::VectorXd& u,
                          double& cost,
                          Eigen::VectorXd& lx,
                          Eigen::VectorXd& lu,
                          bool terminal) const;

    const mjModel*    model_;
    VelocityTaskConfig cfg_;
    int nq_, nv_, nu_, ndx_;

    // Pre-computed Jacobians
    Eigen::MatrixXd Jh_, Jo_, Jqp_, Jqv_, Ju_;
    Eigen::MatrixXd Jvx_, Jvy_, Jvz_, Jwz_, Jwxy_;

    // Pre-computed Hessians
    Eigen::MatrixXd lxx_running_, luu_running_, lxx_terminal_;
    Eigen::Vector4d q_ref_;
};

}  // namespace go2_ilqr
