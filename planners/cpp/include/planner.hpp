#pragma once

/**
 * CubicPlanner — joint-space cubic polynomial trajectory planner.
 *
 * Boundary conditions (zero velocity at start and end):
 *   q(t)  = q0 + a2*t^2 + a3*t^3
 *   qd(t) = 2*a2*t + 3*a3*t^2
 *
 *   a2 =  3/T^2 * (qf - q0)
 *   a3 = -2/T^3 * (qf - q0)
 */

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>


// ── Data structures ─────────────────────────────────────────────────────────

struct JointState {
    std::vector<double> q;           // positions [rad]
    std::vector<double> qd;          // velocities [rad/s]
    std::vector<double> qfrc_bias;   // gravity+Coriolis [Nm]
    double sim_time  = 0.0;
    double wall_time = 0.0;
};

struct Waypoint {
    double              t;    // time from trajectory start [s]
    std::vector<double> q;
    std::vector<double> qd;
};

struct Trajectory {
    std::vector<Waypoint> waypoints;
    double start_time = 0.0;   // wall-clock time at t=0
};


// ── Planner ─────────────────────────────────────────────────────────────────

class CubicPlanner {
public:
    explicit CubicPlanner(int ndof) : ndof_(ndof) {
        if (ndof <= 0)
            throw std::invalid_argument("ndof must be positive");
    }

    /**
     * Plan a cubic trajectory from q_start to q_goal.
     *
     * @param q_start   Initial joint configuration [rad]
     * @param q_goal    Target joint configuration  [rad]
     * @param duration  Trajectory duration         [s]
     * @param dt        Waypoint time step          [s]
     */
    Trajectory plan(const std::vector<double>& q_start,
                    const std::vector<double>& q_goal,
                    double duration,
                    double dt = 0.02) const
    {
        if (static_cast<int>(q_start.size()) != ndof_ ||
            static_cast<int>(q_goal.size())  != ndof_)
            throw std::invalid_argument("q size mismatch");

        const double T  = std::max(duration, 0.01);

        // Coefficients per joint
        std::vector<double> a2(ndof_), a3(ndof_);
        for (int j = 0; j < ndof_; ++j) {
            const double dq = q_goal[j] - q_start[j];
            a2[j] =  3.0 / (T * T) * dq;
            a3[j] = -2.0 / (T * T * T) * dq;
        }

        Trajectory traj;
        for (double t = 0.0; t <= T + 1e-9; t += dt) {
            Waypoint wp;
            wp.t = t;
            wp.q.resize(ndof_);
            wp.qd.resize(ndof_);
            for (int j = 0; j < ndof_; ++j) {
                wp.q[j]  = q_start[j] + a2[j] * t * t + a3[j] * t * t * t;
                wp.qd[j] = 2.0 * a2[j] * t + 3.0 * a3[j] * t * t;
            }
            traj.waypoints.push_back(wp);
        }

        return traj;
    }

private:
    int ndof_;
};
