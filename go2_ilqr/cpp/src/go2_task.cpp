/**
 * go2_task.cpp  –  GO2 cost function implementations
 */

#include "go2_task.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include <mujoco/mujoco.h>

namespace go2_ilqr {

// ---------------------------------------------------------------------------
// Quaternion utilities
// ---------------------------------------------------------------------------

static Eigen::Vector4d QuatMul(const Eigen::Vector4d& q1,
                                const Eigen::Vector4d& q2) {
    // Hamilton product  q1 * q2  (w, x, y, z)
    return Eigen::Vector4d{
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0],
    };
}

static Eigen::Vector4d QuatInv(const Eigen::Vector4d& q) {
    return Eigen::Vector4d{q[0], -q[1], -q[2], -q[3]};
}

Eigen::Vector3d QuatError(const Eigen::Vector4d& q,
                           const Eigen::Vector4d& q_ref) {
    Eigen::Vector4d q_err = QuatMul(QuatInv(q_ref), q);
    // Ensure shortest path
    if (q_err[0] < 0.0) q_err = -q_err;
    return 2.0 * q_err.tail<3>();
}

// ---------------------------------------------------------------------------
// Helper: build constant Jacobians for the 36-D tangent state
// ---------------------------------------------------------------------------
//
//  Tangent layout (nv = 18):
//    dx[0:3]   = dbase_pos
//    dx[3:6]   = dbase_ori  (rotation vector)
//    dx[6:18]  = djoint_pos
//    dx[18:21] = dbase_linvel
//    dx[21:24] = dbase_angvel
//    dx[24:36] = djoint_vel

static Eigen::MatrixXd JacHeight(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 2 * nv);
    J(0, 2) = 1.0;   // Δz of base position
    return J;
}

static Eigen::MatrixXd JacOrientation(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 2 * nv);
    J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    return J;
}

static Eigen::MatrixXd JacLinVel(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 2 * nv);
    J.block<3, 3>(0, nv) = Eigen::Matrix3d::Identity();
    return J;
}

static Eigen::MatrixXd JacAngVel(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 2 * nv);
    J.block<3, 3>(0, nv + 3) = Eigen::Matrix3d::Identity();
    return J;
}

static Eigen::MatrixXd JacJointPos(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(12, 2 * nv);
    J.block<12, 12>(0, 6) = Eigen::Matrix<double, 12, 12>::Identity();
    return J;
}

static Eigen::MatrixXd JacJointVel(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(12, 2 * nv);
    J.block<12, 12>(0, nv + 6) = Eigen::Matrix<double, 12, 12>::Identity();
    return J;
}

static Eigen::MatrixXd JacVx(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 2 * nv);
    J(0, nv + 0) = 1.0;
    return J;
}

static Eigen::MatrixXd JacVy(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 2 * nv);
    J(0, nv + 1) = 1.0;
    return J;
}

static Eigen::MatrixXd JacVz(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 2 * nv);
    J(0, nv + 2) = 1.0;
    return J;
}

static Eigen::MatrixXd JacYaw(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, 2 * nv);
    J(0, nv + 5) = 1.0;
    return J;
}

static Eigen::MatrixXd JacAngVelXY(int nv) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 2 * nv);
    J(0, nv + 3) = 1.0;
    J(1, nv + 4) = 1.0;
    return J;
}

// ---------------------------------------------------------------------------
// StandingTask
// ---------------------------------------------------------------------------

StandingTask::StandingTask(const mjModel*            model,
                            const StandingTaskConfig& cfg)
    : model_(model), cfg_(cfg)
{
    nq_  = model->nq;
    nv_  = model->nv;
    nu_  = model->nu;
    ndx_ = 2 * nv_;

    // Constant Jacobians
    Jh_  = JacHeight(nv_);
    Jo_  = JacOrientation(nv_);
    Jv_  = JacLinVel(nv_);
    Jw_  = JacAngVel(nv_);
    Jqp_ = JacJointPos(nv_);
    Jqv_ = JacJointVel(nv_);
    Ju_  = Eigen::MatrixXd::Identity(nu_, nu_);

    // Pre-compute constant Hessian blocks  lxx = Σ J^T w J
    auto c = cfg_;
    lxx_running_ = (
        c.w_height    * Jh_.transpose()  * Jh_  +
        c.w_orient    * Jo_.transpose()  * Jo_  +
        c.w_linvel    * Jv_.transpose()  * Jv_  +
        c.w_angvel    * Jw_.transpose()  * Jw_  +
        c.w_joint_pos * Jqp_.transpose() * Jqp_ +
        c.w_joint_vel * Jqv_.transpose() * Jqv_
    );  // (36 × 36)
    luu_running_  = c.w_ctrl * Ju_.transpose() * Ju_;  // (12 × 12)
    lxx_terminal_ = c.w_terminal * lxx_running_;

    q_ref_ = Eigen::Vector4d{1.0, 0.0, 0.0, 0.0};
}

void StandingTask::ComputeResiduals(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u,
    double& cost, Eigen::VectorXd& lx, Eigen::VectorXd& lu,
    bool terminal) const
{
    auto& c = cfg_;

    // Extract state
    const Eigen::Vector3d base_pos    = x.segment<3>(0);
    const Eigen::Vector4d base_quat   = x.segment<4>(3);
    const auto            joint_pos   = x.segment<12>(7);
    const Eigen::Vector3d base_linvel = x.segment<3>(nq_);
    const Eigen::Vector3d base_angvel = x.segment<3>(nq_ + 3);
    const auto            joint_vel   = x.segment<12>(nq_ + 6);

    // Residuals
    Eigen::Matrix<double, 1,  1> r_h;  r_h[0] = base_pos[2] - c.target_height;
    Eigen::Vector3d r_o  = QuatError(base_quat, q_ref_);
    Eigen::Vector3d r_v  = base_linvel;
    Eigen::Vector3d r_w  = base_angvel;
    Eigen::Matrix<double, 12, 1> r_qp = joint_pos - c.target_joint_pos;
    Eigen::Matrix<double, 12, 1> r_qv = joint_vel;
    Eigen::VectorXd r_u  = u;

    double scale = terminal ? c.w_terminal : 1.0;

    cost = 0.5 * scale * (
        c.w_height    * r_h.squaredNorm()  +
        c.w_orient    * r_o.squaredNorm()  +
        c.w_linvel    * r_v.squaredNorm()  +
        c.w_angvel    * r_w.squaredNorm()  +
        c.w_joint_pos * r_qp.squaredNorm() +
        c.w_joint_vel * r_qv.squaredNorm()
    );
    if (!terminal) cost += 0.5 * c.w_ctrl * r_u.squaredNorm();

    // Gradient lx = Σ J_i^T w_i r_i
    lx = scale * (
        c.w_height    * Jh_.transpose()  * r_h  +
        c.w_orient    * Jo_.transpose()  * r_o  +
        c.w_linvel    * Jv_.transpose()  * r_v  +
        c.w_angvel    * Jw_.transpose()  * r_w  +
        c.w_joint_pos * Jqp_.transpose() * r_qp +
        c.w_joint_vel * Jqv_.transpose() * r_qv
    );

    lu = terminal ? Eigen::VectorXd::Zero(nu_)
                  : c.w_ctrl * Ju_.transpose() * r_u;
}

double StandingTask::RunningCost(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u, int) const
{
    double cost;
    Eigen::VectorXd lx(ndx_), lu(nu_);
    ComputeResiduals(x, u, cost, lx, lu, /*terminal=*/false);
    return cost;
}

void StandingTask::RunningCostDerivatives(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u, int,
    Eigen::VectorXd& lx, Eigen::VectorXd& lu,
    Eigen::MatrixXd& lxx, Eigen::MatrixXd& luu,
    Eigen::MatrixXd& lux) const
{
    double cost;
    ComputeResiduals(x, u, cost, lx, lu, /*terminal=*/false);
    lxx = lxx_running_;
    luu = luu_running_;
    lux = Eigen::MatrixXd::Zero(nu_, ndx_);
}

double StandingTask::TerminalCost(const Eigen::VectorXd& x) const {
    double cost;
    Eigen::VectorXd lx(ndx_), lu(nu_);
    Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(nu_);
    ComputeResiduals(x, u_zero, cost, lx, lu, /*terminal=*/true);
    return cost;
}

void StandingTask::TerminalCostDerivatives(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& lx,
    Eigen::MatrixXd& lxx) const
{
    double cost;
    Eigen::VectorXd lu(nu_);
    Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(nu_);
    ComputeResiduals(x, u_zero, cost, lx, lu, /*terminal=*/true);
    lxx = lxx_terminal_;
}


// ---------------------------------------------------------------------------
// VelocityTask
// ---------------------------------------------------------------------------

VelocityTask::VelocityTask(const mjModel*            model,
                             const VelocityTaskConfig& cfg)
    : model_(model), cfg_(cfg)
{
    nq_  = model->nq;
    nv_  = model->nv;
    nu_  = model->nu;
    ndx_ = 2 * nv_;

    // Jacobians
    Jh_    = JacHeight(nv_);
    Jo_    = JacOrientation(nv_);
    Jqp_   = JacJointPos(nv_);
    Jqv_   = JacJointVel(nv_);
    Ju_    = Eigen::MatrixXd::Identity(nu_, nu_);
    Jvx_   = JacVx(nv_);
    Jvy_   = JacVy(nv_);
    Jvz_   = JacVz(nv_);
    Jwz_   = JacYaw(nv_);
    Jwxy_  = JacAngVelXY(nv_);

    auto& c = cfg_;
    lxx_running_ = (
        c.w_height    * Jh_.transpose()   * Jh_   +
        c.w_orient    * Jo_.transpose()   * Jo_   +
        c.w_vel_xy    * Jvx_.transpose()  * Jvx_  +
        c.w_vel_xy    * Jvy_.transpose()  * Jvy_  +
        c.w_vz        * Jvz_.transpose()  * Jvz_  +
        c.w_yaw       * Jwz_.transpose()  * Jwz_  +
        c.w_angvel_xy * Jwxy_.transpose() * Jwxy_ +
        c.w_joint_pos * Jqp_.transpose()  * Jqp_  +
        c.w_joint_vel * Jqv_.transpose()  * Jqv_
    );
    luu_running_  = c.w_ctrl * Ju_.transpose() * Ju_;
    lxx_terminal_ = c.w_terminal * lxx_running_;

    q_ref_ = Eigen::Vector4d{1.0, 0.0, 0.0, 0.0};
}

void VelocityTask::ComputeResiduals(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u,
    double& cost, Eigen::VectorXd& lx, Eigen::VectorXd& lu,
    bool terminal) const
{
    auto& c = cfg_;

    const Eigen::Vector3d base_pos    = x.segment<3>(0);
    const Eigen::Vector4d base_quat   = x.segment<4>(3);
    const auto            joint_pos   = x.segment<12>(7);
    const Eigen::Vector3d base_linvel = x.segment<3>(nq_);
    const Eigen::Vector3d base_angvel = x.segment<3>(nq_ + 3);
    const auto            joint_vel   = x.segment<12>(nq_ + 6);

    Eigen::Matrix<double,1,1> r_h;  r_h[0] = base_pos[2] - c.target_height;
    Eigen::Vector3d r_o   = QuatError(base_quat, q_ref_);
    Eigen::Matrix<double,1,1> r_vx; r_vx[0] = base_linvel[0] - c.target_vx;
    Eigen::Matrix<double,1,1> r_vy; r_vy[0] = base_linvel[1] - c.target_vy;
    Eigen::Matrix<double,1,1> r_vz; r_vz[0] = base_linvel[2];
    Eigen::Matrix<double,1,1> r_wz; r_wz[0] = base_angvel[2] - c.target_yaw;
    Eigen::Vector2d r_wxy = base_angvel.segment<2>(0);
    auto r_qp = joint_pos - c.target_joint_pos;
    auto r_qv = joint_vel;
    Eigen::VectorXd r_u  = u;

    double scale = terminal ? c.w_terminal : 1.0;

    cost = 0.5 * scale * (
        c.w_height    * r_h.squaredNorm()   +
        c.w_orient    * r_o.squaredNorm()   +
        c.w_vel_xy    * r_vx.squaredNorm()  +
        c.w_vel_xy    * r_vy.squaredNorm()  +
        c.w_vz        * r_vz.squaredNorm()  +
        c.w_yaw       * r_wz.squaredNorm()  +
        c.w_angvel_xy * r_wxy.squaredNorm() +
        c.w_joint_pos * r_qp.squaredNorm()  +
        c.w_joint_vel * r_qv.squaredNorm()
    );
    if (!terminal) cost += 0.5 * c.w_ctrl * r_u.squaredNorm();

    lx = scale * (
        c.w_height    * Jh_.transpose()   * r_h   +
        c.w_orient    * Jo_.transpose()   * r_o   +
        c.w_vel_xy    * Jvx_.transpose()  * r_vx  +
        c.w_vel_xy    * Jvy_.transpose()  * r_vy  +
        c.w_vz        * Jvz_.transpose()  * r_vz  +
        c.w_yaw       * Jwz_.transpose()  * r_wz  +
        c.w_angvel_xy * Jwxy_.transpose() * r_wxy +
        c.w_joint_pos * Jqp_.transpose()  * r_qp  +
        c.w_joint_vel * Jqv_.transpose()  * r_qv
    );

    lu = terminal ? Eigen::VectorXd::Zero(nu_)
                  : c.w_ctrl * Ju_.transpose() * r_u;
}

double VelocityTask::RunningCost(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u, int) const
{
    double cost; Eigen::VectorXd lx(ndx_), lu(nu_);
    ComputeResiduals(x, u, cost, lx, lu, false);
    return cost;
}

void VelocityTask::RunningCostDerivatives(
    const Eigen::VectorXd& x, const Eigen::VectorXd& u, int,
    Eigen::VectorXd& lx, Eigen::VectorXd& lu,
    Eigen::MatrixXd& lxx, Eigen::MatrixXd& luu,
    Eigen::MatrixXd& lux) const
{
    double cost;
    ComputeResiduals(x, u, cost, lx, lu, false);
    lxx = lxx_running_;
    luu = luu_running_;
    lux = Eigen::MatrixXd::Zero(nu_, ndx_);
}

double VelocityTask::TerminalCost(const Eigen::VectorXd& x) const {
    double cost; Eigen::VectorXd lx(ndx_), lu(nu_);
    ComputeResiduals(x, Eigen::VectorXd::Zero(nu_), cost, lx, lu, true);
    return cost;
}

void VelocityTask::TerminalCostDerivatives(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& lx,
    Eigen::MatrixXd& lxx) const
{
    double cost; Eigen::VectorXd lu(nu_);
    ComputeResiduals(x, Eigen::VectorXd::Zero(nu_), cost, lx, lu, true);
    lxx = lxx_terminal_;
}

}  // namespace go2_ilqr
