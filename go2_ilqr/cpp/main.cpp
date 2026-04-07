/**
 * main.cpp  –  GO2 iLQR demo
 *
 * Usage:
 *   ./go2_ilqr_demo [--task standing|velocity] [--horizon N] [--max_iter N]
 *
 * The binary looks for the MJCF model at
 *   <binary_dir>/../models/go2.xml  (installed from assets/go2/go2.xml)
 * or the path set via GO2_MODEL env variable.
 */

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include <mujoco/mujoco.h>
#include <Eigen/Dense>

#include "ilqr.h"
#include "go2_task.h"

using namespace go2_ilqr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static mjModel* LoadModel(const std::string& path) {
    char err[1024] = {};
    mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    if (!m)
        throw std::runtime_error("Failed to load model: " + path
                                 + "\n  " + err);
    return m;
}

static Eigen::VectorXd MakeDefaultX0(const mjModel* model) {
    // Default standing state
    //   nq = 19: base_pos(3) + base_quat(4) + joint_angles(12)
    //   nv = 18: base_vel(6) + joint_vel(12)
    const int nq = model->nq;
    const int nv = model->nv;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nq + nv);

    // Base position: slightly above default height to account for CoM offset
    x0[0] = 0.0;   // x
    x0[1] = 0.0;   // y
    x0[2] = GO2_STANDING_HEIGHT + 0.084;  // z (CoM above ground)

    // Base quaternion: identity (upright)
    x0[3] = 1.0;   // w
    x0[4] = 0.0;   // x
    x0[5] = 0.0;   // y
    x0[6] = 0.0;   // z

    // Joint angles: default standing pose
    x0.segment<12>(7) = GO2_DEFAULT_QPOS;

    // Velocities: zero
    return x0;
}

static void PrintTrajectoryStats(
    const mjModel*          model,
    const Eigen::MatrixXd&  X,
    const Eigen::MatrixXd&  U,
    const std::string&      task_name)
{
    const int T  = static_cast<int>(U.rows());
    const int nq = model->nq;

    // Base height statistics
    double h_mean = 0, h_min = 1e9, h_max = -1e9;
    for (int t = 0; t <= T; ++t) {
        double h = X(t, 2);
        h_mean += h;
        h_min   = std::min(h_min, h);
        h_max   = std::max(h_max, h);
    }
    h_mean /= (T + 1);

    // Max joint deviation
    double max_jdev = 0.0;
    for (int t = 0; t <= T; ++t) {
        auto jdev = (X.block<1, 12>(t, 7) - GO2_DEFAULT_QPOS.transpose()).norm();
        max_jdev = std::max(max_jdev, static_cast<double>(jdev));
    }

    // Control norm statistics
    double u_mean = 0, u_max = 0;
    for (int t = 0; t < T; ++t) {
        double n = U.row(t).norm();
        u_mean += n;
        u_max   = std::max(u_max, n);
    }
    u_mean /= T;

    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << " Trajectory statistics: " << task_name << "\n";
    std::cout << std::string(55, '=') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Horizon         : " << T << " steps\n";
    std::cout << "  Base height     : mean=" << h_mean
              << "  [" << h_min << ", " << h_max << "] m\n";
    std::cout << "  Max joint dev.  : " << max_jdev << " rad\n";
    std::cout << "  Ctrl norm       : mean=" << u_mean
              << "  max=" << u_max << " Nm\n";
    std::cout << std::string(55, '=') << "\n\n";
}


// ---------------------------------------------------------------------------
// Standing demo
// ---------------------------------------------------------------------------

static void RunStanding(const std::string& model_path,
                         int horizon, int max_iter)
{
    std::cout << "\nLoading model: " << model_path << "\n";
    mjModel* model = LoadModel(model_path);

    std::cout << "  nq=" << model->nq
              << "  nv=" << model->nv
              << "  nu=" << model->nu << "\n";

    Eigen::VectorXd x0 = MakeDefaultX0(model);
    std::cout << "Initial base z: " << x0[2] << " m\n";

    StandingTaskConfig task_cfg;
    task_cfg.w_height    = 200.0;
    task_cfg.w_orient    = 100.0;
    task_cfg.w_linvel    =  10.0;
    task_cfg.w_angvel    =  10.0;
    task_cfg.w_joint_pos =   5.0;
    task_cfg.w_joint_vel =   1.0;
    task_cfg.w_ctrl      =   0.001;
    task_cfg.w_terminal  =  10.0;

    auto task = std::make_shared<StandingTask>(model, task_cfg);

    ILQRConfig cfg;
    cfg.horizon    = horizon;
    cfg.mu_init    = 1.0;
    cfg.mu_min     = 1e-6;
    cfg.mu_max     = 1e8;
    cfg.delta_0    = 2.0;
    cfg.alpha_min  = 1e-8;
    cfg.tol        = 1e-5;
    cfg.max_iter   = max_iter;
    cfg.fd_eps     = 1e-6;
    cfg.ctrl_limit = 33.5;
    cfg.verbose    = true;

    ILQR solver(model, task, cfg);

    std::cout << "\n── iLQR Optimisation (Standing Task) ──\n";
    auto [U_opt, X_opt, cost] = solver.Solve(x0);

    PrintTrajectoryStats(model, X_opt, U_opt, "Standing");

    mj_deleteModel(model);
}


// ---------------------------------------------------------------------------
// Velocity demo
// ---------------------------------------------------------------------------

static void RunVelocity(const std::string& model_path,
                         int horizon, int max_iter)
{
    std::cout << "\nLoading model: " << model_path << "\n";
    mjModel* model = LoadModel(model_path);

    std::cout << "  nq=" << model->nq
              << "  nv=" << model->nv
              << "  nu=" << model->nu << "\n";

    Eigen::VectorXd x0 = MakeDefaultX0(model);

    VelocityTaskConfig task_cfg;
    task_cfg.target_vx   = 0.5;
    task_cfg.target_vy   = 0.0;
    task_cfg.target_yaw  = 0.0;
    task_cfg.w_height    = 150.0;
    task_cfg.w_orient    =  50.0;
    task_cfg.w_vel_xy    = 100.0;
    task_cfg.w_yaw       =  50.0;
    task_cfg.w_vz        =  20.0;
    task_cfg.w_angvel_xy =   5.0;
    task_cfg.w_joint_pos =   2.0;
    task_cfg.w_joint_vel =   0.5;
    task_cfg.w_ctrl      =   0.001;
    task_cfg.w_terminal  =   5.0;

    auto task = std::make_shared<VelocityTask>(model, task_cfg);

    ILQRConfig cfg;
    cfg.horizon    = horizon;
    cfg.mu_init    = 1.0;
    cfg.mu_min     = 1e-6;
    cfg.mu_max     = 1e8;
    cfg.delta_0    = 2.0;
    cfg.alpha_min  = 1e-8;
    cfg.tol        = 1e-5;
    cfg.max_iter   = max_iter;
    cfg.fd_eps     = 1e-6;
    cfg.ctrl_limit = 33.5;
    cfg.verbose    = true;

    ILQR solver(model, task, cfg);

    std::cout << "\n── iLQR Optimisation (Velocity Task: vx="
              << task_cfg.target_vx << " m/s) ──\n";
    auto [U_opt, X_opt, cost] = solver.Solve(x0);

    PrintTrajectoryStats(model, X_opt, U_opt, "Velocity");

    const int nq = model->nq;
    std::cout << "  Target vx = " << task_cfg.target_vx << " m/s  |  "
              << "Achieved vx = " << X_opt(X_opt.rows()-1, nq) << " m/s\n\n";

    mj_deleteModel(model);
}


// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string task      = "standing";
    int         horizon   = 50;
    int         max_iter  = 30;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            task = argv[++i];
        } else if (std::strcmp(argv[i], "--horizon") == 0 && i + 1 < argc) {
            horizon = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max_iter") == 0 && i + 1 < argc) {
            max_iter = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: go2_ilqr_demo [--task standing|velocity] "
                         "[--horizon N] [--max_iter N]\n";
            return 0;
        }
    }

    // Resolve model path
    std::string model_path;
    if (const char* env = std::getenv("GO2_MODEL")) {
        model_path = env;
    } else {
        // Default: <binary_dir>/../models/go2.xml  (installed from assets/go2/go2.xml)
        namespace fs = std::filesystem;
        fs::path bin_dir = fs::path(argv[0]).parent_path();
        model_path = (bin_dir / ".." / "models" / "go2.xml").string();
    }

    try {
        if (task == "standing") {
            RunStanding(model_path, horizon, max_iter);
        } else if (task == "velocity") {
            RunVelocity(model_path, horizon, max_iter);
        } else {
            std::cerr << "Unknown task '" << task
                      << "'. Choose: standing | velocity\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
