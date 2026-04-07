/**
 * ilqr.h  –  iLQR solver for MuJoCo-based systems
 *
 * Algorithm
 * ---------
 * Implements the DDP / iLQR algorithm:
 *   Tassa et al., IROS 2012 – "Synthesis and stabilization of complex
 *   behaviours through online trajectory optimisation"
 *
 * State representation
 * --------------------
 *   nx  = nq + nv   (full MuJoCo state)
 *   ndx = 2 * nv    (tangent-space state; handles floating-base quaternion)
 *   nu  = model->nu (control dimension)
 *
 * Dynamics Jacobians
 * ------------------
 *   A[t]  (ndx × ndx)  = ∂x_{t+1}/∂x_t  via mjd_transitionFD
 *   B[t]  (ndx × nu)   = ∂x_{t+1}/∂u_t  via mjd_transitionFD
 *
 * Cost interface
 * --------------
 *   Users subclass CostFunction and provide analytical gradients /
 *   Hessians.  The Gauss-Newton structure (J^T W J) guarantees PSD
 *   Hessians without second-order finite differences.
 *
 * Dependencies
 * ------------
 *   MuJoCo >= 3.0, Eigen3
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>

namespace go2_ilqr {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct ILQRConfig {
    int    horizon    = 100;     ///< planning horizon T
    double mu_init    = 1.0;     ///< initial Tikhonov regularisation
    double mu_min     = 1e-6;    ///< minimum regularisation
    double mu_max     = 1e10;    ///< maximum regularisation
    double delta_0    = 2.0;     ///< regularisation scaling factor
    double alpha_min  = 1e-8;    ///< minimum Armijo line-search step
    double tol        = 1e-4;    ///< convergence: |ΔJ| / (1 + |J|)
    int    max_iter   = 50;      ///< maximum outer iterations
    double fd_eps     = 1e-6;    ///< FD epsilon for dynamics Jacobians
    double ctrl_limit = 33.5;    ///< symmetric torque clamp [Nm]
    bool   verbose    = true;    ///< print iteration info
};


// ---------------------------------------------------------------------------
// Abstract cost function
// ---------------------------------------------------------------------------

class CostFunction {
public:
    virtual ~CostFunction() = default;

    /** Scalar running cost l(x, u, t). */
    virtual double RunningCost(
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& u,
        int t) const = 0;

    /**
     * Analytical running-cost derivatives at (x, u, t).
     *
     * @param[out] lx   gradient  ∂l/∂x  (ndx)
     * @param[out] lu   gradient  ∂l/∂u  (nu)
     * @param[out] lxx  Hessian   ∂²l/∂x²  (ndx × ndx)  – PSD
     * @param[out] luu  Hessian   ∂²l/∂u²  (nu  × nu)   – PSD
     * @param[out] lux  cross     ∂²l/∂u∂x (nu  × ndx)
     */
    virtual void RunningCostDerivatives(
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& u,
        int t,
        Eigen::VectorXd& lx,
        Eigen::VectorXd& lu,
        Eigen::MatrixXd& lxx,
        Eigen::MatrixXd& luu,
        Eigen::MatrixXd& lux) const = 0;

    /** Scalar terminal cost lf(x). */
    virtual double TerminalCost(const Eigen::VectorXd& x) const = 0;

    /**
     * Analytical terminal-cost derivatives at x.
     *
     * @param[out] lx   gradient  ∂lf/∂x   (ndx)
     * @param[out] lxx  Hessian   ∂²lf/∂x² (ndx × ndx)  – PSD
     */
    virtual void TerminalCostDerivatives(
        const Eigen::VectorXd& x,
        Eigen::VectorXd& lx,
        Eigen::MatrixXd& lxx) const = 0;
};


// ---------------------------------------------------------------------------
// iLQR solver
// ---------------------------------------------------------------------------

class ILQR {
public:
    /**
     * @param model  compiled MuJoCo model (not owned; must outlive solver)
     * @param cost   cost function (shared ownership)
     * @param cfg    solver configuration
     */
    ILQR(const mjModel*                 model,
         std::shared_ptr<CostFunction>  cost,
         const ILQRConfig&              cfg = ILQRConfig{});

    ~ILQR();

    // --- main interface ---

    /**
     * Solve from initial state x0.
     *
     * @param x0      initial full state [qpos; qvel]  (nx)
     * @param U_init  initial control guess (T × nu);  zeros if nullptr
     * @return        {U_opt (T×nu), X_opt ((T+1)×nx), final_cost}
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, double>
    Solve(const Eigen::VectorXd&       x0,
          const Eigen::MatrixXd*       U_init = nullptr);

    // --- accessors ---
    int nq()  const { return nq_; }
    int nv()  const { return nv_; }
    int nu()  const { return nu_; }
    int ndx() const { return ndx_; }
    int T()   const { return cfg_.horizon; }

    /** Compute tangent-space difference  dx = x1 ⊖ x2. */
    Eigen::VectorXd StateDiff(const Eigen::VectorXd& x1,
                               const Eigen::VectorXd& x2) const;

    /** Apply tangent vector  x_new = x ⊕ dx. */
    Eigen::VectorXd StateAdd(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& dx) const;

private:
    // --- MuJoCo helpers ---
    void   SetState(mjData* d, const Eigen::VectorXd& x) const;
    Eigen::VectorXd GetState(const mjData* d) const;
    Eigen::VectorXd Step(const Eigen::VectorXd& x,
                          const Eigen::VectorXd& u) const;
    Eigen::VectorXd ClampCtrl(const Eigen::VectorXd& u) const;

    // --- algorithm phases ---
    double ForwardRollout(const Eigen::VectorXd& x0,
                           const Eigen::MatrixXd& U);
    std::pair<Eigen::MatrixXd, double>
           ForwardRolloutNew(const Eigen::VectorXd&  x0,
                              const Eigen::MatrixXd&  U_new) const;

    void   ComputeDynamicsJacobians();
    void   ComputeCostDerivatives();
    bool   BackwardPass();

    /** @return {U_new, X_new, new_cost, alpha} or failure (alpha=0). */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, double, double>
           ForwardPassLineSearch(const Eigen::VectorXd& x0,
                                  const Eigen::MatrixXd& U_ref,
                                  double                 cost_ref) const;

    // --- regularisation ---
    void IncreaseRegularisation();
    void DecreaseRegularisation();

    // --- members ---
    const mjModel*                model_;
    std::shared_ptr<CostFunction> cost_;
    ILQRConfig                    cfg_;
    mjData*                       data_;     ///< primary data (rollout)
    mjData*                       data_fd_;  ///< scratch data (FD Jacobians)

    int nq_, nv_, nu_, nx_, ndx_, T_;

    // Trajectory
    Eigen::MatrixXd X_;   ///< (T+1) × nx
    Eigen::MatrixXd U_;   ///< T × nu

    // Dynamics Jacobians
    std::vector<Eigen::MatrixXd> A_;  ///< T × (ndx × ndx)
    std::vector<Eigen::MatrixXd> B_;  ///< T × (ndx × nu)

    // Cost derivatives
    std::vector<Eigen::VectorXd> lx_;   ///< (T+1) × ndx
    std::vector<Eigen::VectorXd> lu_;   ///< T × nu
    std::vector<Eigen::MatrixXd> lxx_;  ///< (T+1) × (ndx × ndx)
    std::vector<Eigen::MatrixXd> luu_;  ///< T × (nu × nu)
    std::vector<Eigen::MatrixXd> lux_;  ///< T × (nu × ndx)

    // Policy gains
    std::vector<Eigen::VectorXd> k_;   ///< T × nu  (feedforward)
    std::vector<Eigen::MatrixXd> K_;   ///< T × (nu × ndx)  (feedback)

    // Value function
    Eigen::VectorXd Vx_;
    Eigen::MatrixXd Vxx_;

    // Expected improvement terms
    double dV1_ = 0.0;
    double dV2_ = 0.0;

    // Regularisation state
    double mu_    = 1.0;
    double delta_ = 2.0;
};

}  // namespace go2_ilqr
