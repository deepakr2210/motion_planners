/**
 * ilqr.cpp  –  iLQR solver implementation
 */

#include "ilqr.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>

namespace go2_ilqr {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

ILQR::ILQR(const mjModel*                model,
           std::shared_ptr<CostFunction> cost,
           const ILQRConfig&             cfg)
    : model_(model),
      cost_(std::move(cost)),
      cfg_(cfg),
      data_(mj_makeData(model)),
      data_fd_(mj_makeData(model))
{
    if (!data_ || !data_fd_)
        throw std::runtime_error("ILQR: mj_makeData failed");

    nq_  = model->nq;
    nv_  = model->nv;
    nu_  = model->nu;
    nx_  = nq_ + nv_;
    ndx_ = 2 * nv_;
    T_   = cfg_.horizon;

    // Allocate trajectory
    X_ = Eigen::MatrixXd::Zero(T_ + 1, nx_);
    U_ = Eigen::MatrixXd::Zero(T_,     nu_);

    // Allocate Jacobians
    A_.resize(T_, Eigen::MatrixXd::Zero(ndx_, ndx_));
    B_.resize(T_, Eigen::MatrixXd::Zero(ndx_, nu_));

    // Allocate cost derivatives
    lx_.resize( T_ + 1, Eigen::VectorXd::Zero(ndx_));
    lu_.resize( T_,     Eigen::VectorXd::Zero(nu_));
    lxx_.resize(T_ + 1, Eigen::MatrixXd::Zero(ndx_, ndx_));
    luu_.resize(T_,     Eigen::MatrixXd::Zero(nu_,  nu_));
    lux_.resize(T_,     Eigen::MatrixXd::Zero(nu_,  ndx_));

    // Allocate policy gains
    k_.resize(T_, Eigen::VectorXd::Zero(nu_));
    K_.resize(T_, Eigen::MatrixXd::Zero(nu_, ndx_));

    // Allocate value function
    Vx_  = Eigen::VectorXd::Zero(ndx_);
    Vxx_ = Eigen::MatrixXd::Zero(ndx_, ndx_);

    // Initialise regularisation
    mu_    = cfg_.mu_init;
    delta_ = cfg_.delta_0;
}

ILQR::~ILQR() {
    if (data_)    mj_deleteData(data_);
    if (data_fd_) mj_deleteData(data_fd_);
}

// ---------------------------------------------------------------------------
// MuJoCo helpers
// ---------------------------------------------------------------------------

void ILQR::SetState(mjData* d, const Eigen::VectorXd& x) const {
    assert(x.size() == nx_);
    mju_copy(d->qpos, x.data(),        nq_);
    mju_copy(d->qvel, x.data() + nq_,  nv_);
    mj_forward(model_, d);
}

Eigen::VectorXd ILQR::GetState(const mjData* d) const {
    Eigen::VectorXd x(nx_);
    mju_copy(x.data(),        d->qpos, nq_);
    mju_copy(x.data() + nq_,  d->qvel, nv_);
    return x;
}

Eigen::VectorXd ILQR::ClampCtrl(const Eigen::VectorXd& u) const {
    return u.cwiseMax(-cfg_.ctrl_limit).cwiseMin(cfg_.ctrl_limit);
}

Eigen::VectorXd ILQR::Step(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& u) const {
    SetState(data_, x);
    Eigen::VectorXd uc = ClampCtrl(u);
    mju_copy(data_->ctrl, uc.data(), nu_);
    mj_step(model_, data_);
    return GetState(data_);
}

// ---------------------------------------------------------------------------
// Tangent-space operations
// ---------------------------------------------------------------------------

Eigen::VectorXd ILQR::StateDiff(const Eigen::VectorXd& x1,
                                  const Eigen::VectorXd& x2) const {
    Eigen::VectorXd dx(ndx_);
    // Position part: use mj_differentiatePos (quaternion-aware)
    mj_differentiatePos(model_,
                        dx.data(),          // output dpos (nv)
                        1.0,                // dt = 1 → unit perturbation
                        x2.data(),          // qpos reference
                        x1.data());         // qpos perturbed
    // Velocity part: simple subtraction
    dx.tail(nv_) = x1.tail(nv_) - x2.tail(nv_);
    return dx;
}

Eigen::VectorXd ILQR::StateAdd(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& dx) const {
    Eigen::VectorXd x_new(nx_);
    // Copy and integrate position
    mju_copy(x_new.data(), x.data(), nq_);
    mj_integratePos(model_, x_new.data(), dx.data(), 1.0);
    // Add velocity
    x_new.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
    return x_new;
}

// ---------------------------------------------------------------------------
// Forward rollout
// ---------------------------------------------------------------------------

double ILQR::ForwardRollout(const Eigen::VectorXd& x0,
                              const Eigen::MatrixXd& U) {
    X_.row(0) = x0.transpose();
    double total = 0.0;
    for (int t = 0; t < T_; ++t) {
        X_.row(t + 1) = Step(X_.row(t), U.row(t).transpose()).transpose();
        total += cost_->RunningCost(X_.row(t), U.row(t), t);
    }
    total += cost_->TerminalCost(X_.row(T_));
    return total;
}

std::pair<Eigen::MatrixXd, double>
ILQR::ForwardRolloutNew(const Eigen::VectorXd& x0,
                         const Eigen::MatrixXd& U_new) const {
    Eigen::MatrixXd X_new(T_ + 1, nx_);
    X_new.row(0) = x0.transpose();
    double total = 0.0;
    for (int t = 0; t < T_; ++t) {
        X_new.row(t + 1) = Step(X_new.row(t),
                                  U_new.row(t).transpose()).transpose();
        total += cost_->RunningCost(X_new.row(t), U_new.row(t), t);
    }
    total += cost_->TerminalCost(X_new.row(T_));
    return {X_new, total};
}

// ---------------------------------------------------------------------------
// Dynamics linearisation  (mjd_transitionFD)
// ---------------------------------------------------------------------------

void ILQR::ComputeDynamicsJacobians() {
    const double eps = cfg_.fd_eps;

    std::vector<mjtNum> A_flat(ndx_ * ndx_);
    std::vector<mjtNum> B_flat(ndx_ * nu_);

    for (int t = 0; t < T_; ++t) {
        SetState(data_fd_, X_.row(t));
        Eigen::VectorXd uc = ClampCtrl(U_.row(t).transpose());
        mju_copy(data_fd_->ctrl, uc.data(), nu_);

        // centred FD: flg_centered = 1
        mjd_transitionFD(model_, data_fd_,
                         eps, /*centered=*/1,
                         A_flat.data(), B_flat.data(),
                         nullptr, nullptr);

        // Copy row-major flat arrays → Eigen matrices
        for (int r = 0; r < ndx_; ++r)
            for (int c = 0; c < ndx_; ++c)
                A_[t](r, c) = A_flat[r * ndx_ + c];

        for (int r = 0; r < ndx_; ++r)
            for (int c = 0; c < nu_; ++c)
                B_[t](r, c) = B_flat[r * nu_ + c];
    }
}

// ---------------------------------------------------------------------------
// Cost derivatives  (analytical, from CostFunction interface)
// ---------------------------------------------------------------------------

void ILQR::ComputeCostDerivatives() {
    for (int t = 0; t < T_; ++t) {
        cost_->RunningCostDerivatives(
            X_.row(t), U_.row(t), t,
            lx_[t], lu_[t], lxx_[t], luu_[t], lux_[t]);
    }
    cost_->TerminalCostDerivatives(X_.row(T_), lx_[T_], lxx_[T_]);
}

// ---------------------------------------------------------------------------
// Backward pass  (Riccati recursion)
// ---------------------------------------------------------------------------

bool ILQR::BackwardPass() {
    Vx_  = lx_[T_];
    Vxx_ = lxx_[T_];
    dV1_ = 0.0;
    dV2_ = 0.0;

    for (int t = T_ - 1; t >= 0; --t) {
        const Eigen::MatrixXd& A   = A_[t];       // ndx × ndx
        const Eigen::MatrixXd& B   = B_[t];       // ndx × nu
        const Eigen::VectorXd& lx  = lx_[t];      // ndx
        const Eigen::VectorXd& lu  = lu_[t];      // nu
        const Eigen::MatrixXd& lxx = lxx_[t];     // ndx × ndx
        const Eigen::MatrixXd& luu = luu_[t];     // nu  × nu
        const Eigen::MatrixXd& lux = lux_[t];     // nu  × ndx

        // ---- Q-function ----
        Eigen::MatrixXd VxxA = Vxx_ * A;          // ndx × ndx
        Eigen::MatrixXd VxxB = Vxx_ * B;          // ndx × nu

        Eigen::MatrixXd Q_xx = lxx + A.transpose() * VxxA;
        Eigen::MatrixXd Q_uu = luu + B.transpose() * VxxB
                                   + mu_ * Eigen::MatrixXd::Identity(nu_, nu_);
        Eigen::MatrixXd Q_ux = lux + B.transpose() * VxxA;
        Eigen::VectorXd Q_x  = lx  + A.transpose() * Vx_;
        Eigen::VectorXd Q_u  = lu  + B.transpose() * Vx_;

        // Symmetrise
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Cholesky factorisation  (returns L such that Q_uu = L L^T)
        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu);
        if (llt.info() != Eigen::Success)
            return false;   // not PD → caller increases μ

        // Gains
        k_[t] = llt.solve(-Q_u);    // (nu,)
        K_[t] = llt.solve(-Q_ux);   // (nu, ndx)

        // Expected improvement
        dV1_ += Q_u.dot(k_[t]);
        dV2_ += 0.5 * k_[t].dot(Q_uu * k_[t]);

        // Value function update
        Eigen::VectorXd Quu_k = Q_uu * k_[t];
        Vx_  = Q_x + K_[t].transpose() * (Quu_k + Q_u) + Q_ux.transpose() * k_[t];
        Vxx_ = Q_xx
             + K_[t].transpose() * Q_uu * K_[t]
             + K_[t].transpose() * Q_ux
             + Q_ux.transpose()  * K_[t];
        Vxx_ = 0.5 * (Vxx_ + Vxx_.transpose());
    }
    return true;
}

// ---------------------------------------------------------------------------
// Forward pass with Armijo line search
// ---------------------------------------------------------------------------

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, double, double>
ILQR::ForwardPassLineSearch(const Eigen::VectorXd& x0,
                              const Eigen::MatrixXd& U_ref,
                              double cost_ref) const {
    double alpha = 1.0;

    while (alpha >= cfg_.alpha_min) {
        Eigen::MatrixXd U_new(T_, nu_);
        Eigen::MatrixXd X_new(T_ + 1, nx_);
        X_new.row(0) = x0.transpose();
        double total = 0.0;

        for (int t = 0; t < T_; ++t) {
            // Tangent-space deviation
            Eigen::VectorXd dx = StateDiff(X_new.row(t), X_.row(t));

            // Candidate control
            Eigen::VectorXd u_new = U_ref.row(t).transpose()
                                  + alpha * k_[t]
                                  + K_[t] * dx;
            u_new = ClampCtrl(u_new);
            U_new.row(t) = u_new.transpose();

            X_new.row(t + 1) = Step(X_new.row(t), u_new).transpose();
            total += cost_->RunningCost(X_new.row(t), u_new, t);
        }
        total += cost_->TerminalCost(X_new.row(T_));

        // Armijo condition
        const double dJ_expected = alpha * dV1_ + alpha * alpha * dV2_;
        const double dJ_actual   = cost_ref - total;

        if (dJ_actual >= 0.1 * std::abs(dJ_expected)) {
            return {U_new, X_new, total, alpha};
        }

        alpha *= 0.5;
    }

    // Line search failed
    return {Eigen::MatrixXd{}, Eigen::MatrixXd{}, cost_ref, 0.0};
}

// ---------------------------------------------------------------------------
// Regularisation schedule  (mirrors mujoco_mpc approach)
// ---------------------------------------------------------------------------

void ILQR::IncreaseRegularisation() {
    delta_ = std::max(delta_ * cfg_.delta_0, cfg_.delta_0);
    mu_    = std::max(mu_  * delta_,         cfg_.mu_min);
    mu_    = std::min(mu_,                   cfg_.mu_max);
}

void ILQR::DecreaseRegularisation() {
    delta_ = std::min(delta_ / cfg_.delta_0, 1.0 / cfg_.delta_0);
    mu_    = std::max(mu_ * delta_,          cfg_.mu_min);
}

// ---------------------------------------------------------------------------
// Main solve
// ---------------------------------------------------------------------------

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, double>
ILQR::Solve(const Eigen::VectorXd& x0, const Eigen::MatrixXd* U_init) {

    // Initialise controls
    if (U_init && U_init->rows() == T_ && U_init->cols() == nu_)
        U_ = *U_init;
    else
        U_.setZero();

    // Reset regularisation
    mu_    = cfg_.mu_init;
    delta_ = cfg_.delta_0;

    double cost = ForwardRollout(x0, U_);

    if (cfg_.verbose) {
        std::cout << std::setw(5)  << "Iter"
                  << std::setw(16) << "Cost"
                  << std::setw(14) << "dCost"
                  << std::setw(10) << "alpha"
                  << std::setw(12) << "mu"
                  << "\n" << std::string(58, '-') << "\n";
        std::cout << std::setw(5)  << "init"
                  << std::setw(16) << std::fixed << std::setprecision(6) << cost
                  << std::setw(14) << "–"
                  << std::setw(10) << "–"
                  << std::setw(12) << std::scientific << mu_
                  << "\n";
    }

    auto t_start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < cfg_.max_iter; ++iter) {
        // ---- derivatives ----
        ComputeDynamicsJacobians();
        ComputeCostDerivatives();

        // ---- backward pass with regularisation retries ----
        bool bp_ok = false;
        for (int reg_try = 0; reg_try < 10; ++reg_try) {
            if (BackwardPass()) { bp_ok = true; break; }
            IncreaseRegularisation();
            if (mu_ >= cfg_.mu_max) break;
        }

        if (!bp_ok) {
            if (cfg_.verbose)
                std::cout << "  Backward pass failed at iter "
                          << iter << ", stopping.\n";
            break;
        }

        // ---- forward pass / line search ----
        auto [U_new, X_new, new_cost, alpha] =
            ForwardPassLineSearch(x0, U_, cost);

        if (alpha == 0.0) {           // line search failed
            IncreaseRegularisation();
            if (cfg_.verbose)
                std::cout << std::setw(5)  << iter + 1
                          << "  line-search failed  mu="
                          << std::scientific << mu_ << "\n";
            continue;
        }

        // ---- accept step ----
        const double dcost = cost - new_cost;
        cost = new_cost;
        U_   = std::move(U_new);
        X_   = std::move(X_new);
        DecreaseRegularisation();

        if (cfg_.verbose) {
            std::cout << std::setw(5)  << iter + 1
                      << std::setw(16) << std::fixed << std::setprecision(6) << cost
                      << std::setw(14) << dcost
                      << std::setw(10) << std::fixed << std::setprecision(4) << alpha
                      << std::setw(12) << std::scientific << mu_
                      << "\n";
        }

        // ---- convergence ----
        if (std::abs(dcost) < cfg_.tol * (1.0 + std::abs(cost))) {
            if (cfg_.verbose)
                std::cout << "  Converged at iter " << iter + 1 << ".\n";
            break;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    if (cfg_.verbose) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "  Wall time: " << std::fixed << std::setprecision(3)
                  << elapsed << " s\n";
    }

    return {U_, X_, cost};
}

}  // namespace go2_ilqr
