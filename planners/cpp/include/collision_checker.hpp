#pragma once

/**
 * CollisionChecker — geometric collision checker for sphere obstacles.
 *
 * Two query types
 * ---------------
 *   isPointInCollision(p)
 *       Is a 3-D workspace point inside any obstacle sphere?
 *       Pure geometry — no MuJoCo required.  O(n_obstacles).
 *
 *   isConfigInCollision(q, model, data)   [approximate — fast]
 *       Run MuJoCo FK → check every link's body-frame origin + bounding radius
 *       against all obstacles.  Conservative (can have false positives near link
 *       edges) but very fast.  Good for RRT/trajopt inner loops.
 *
 *   isConfigInCollisionExact(q, model, data)   [exact — slower]
 *       Run full mj_forward → inspect data->ncon.  Accurate for any geometry,
 *       including self-collision.  ~10–50 × slower than the approximate check.
 *
 * Load obstacles from the YAML file written by obstacles/generator.py:
 *   auto cc = CollisionChecker::fromYaml("obstacles/data.yaml");
 *
 * Build requirements (handled by planners/cpp/CMakeLists.txt):
 *   - Eigen3
 *   - yaml-cpp
 *   - MuJoCo C API  (mujoco/mujoco.h + libmujoco)
 */

#include <Eigen/Core>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>


// ── Data ─────────────────────────────────────────────────────────────────────

struct Sphere {
    Eigen::Vector3d center;
    double          radius;
};


// ── CollisionChecker ─────────────────────────────────────────────────────────

class CollisionChecker {
public:

    // ── Constructors ──────────────────────────────────────────────────────────

    explicit CollisionChecker(std::vector<Sphere> obstacles)
        : obstacles_(std::move(obstacles)) {}

    /**
     * Load from a YAML file written by obstacles/generator.py.
     *
     * Expected format:
     *   obstacles:
     *     - center: [x, y, z]
     *       radius: r
     *     - ...
     */
    static CollisionChecker fromYaml(const std::string& path) {
        YAML::Node doc = YAML::LoadFile(path);
        std::vector<Sphere> obs;
        for (const auto& s : doc["obstacles"]) {
            Sphere sp;
            auto c     = s["center"];
            sp.center  = {c[0].as<double>(), c[1].as<double>(), c[2].as<double>()};
            sp.radius  = s["radius"].as<double>();
            obs.push_back(sp);
        }
        return CollisionChecker(std::move(obs));
    }

    // ── Workspace point query ──────────────────────────────────────────────────

    /**
     * Returns true if *p* is strictly inside any obstacle sphere.
     * No MuJoCo required.
     */
    bool isPointInCollision(const Eigen::Vector3d& p) const {
        for (const auto& s : obstacles_)
            if ((p - s.center).norm() < s.radius)
                return true;
        return false;
    }

    /**
     * Signed distance from *p* to the nearest obstacle surface [m].
     * Negative  → point is inside an obstacle.
     * +infinity → no obstacles.
     */
    double pointMinClearance(const Eigen::Vector3d& p) const {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& s : obstacles_)
            best = std::min(best, (p - s.center).norm() - s.radius);
        return best;
    }

    // ── Joint-space config query (approximate) ────────────────────────────────

    /**
     * FK + bounding-sphere check (fast, conservative).
     *
     * Steps:
     *   1. Save current qpos / qvel.
     *   2. Set qpos[:q.size()] = q, zero qvel.
     *   3. mj_kinematics → updates body-frame positions (xpos).
     *   4. Check each link body position + bounding radius vs. all spheres.
     *   5. Restore and return.
     *
     * @param q       Joint angles [rad], length ≤ model->nq.
     * @param model   Pointer to the loaded MjModel.
     * @param data    Pointer to the associated MjData (state is saved/restored).
     */
    bool isConfigInCollision(const std::vector<double>& q,
                             mjModel* model, mjData* data) const
    {
        // Approximate per-link bounding radii (conservative, body IDs 1-8)
        static const double kLinkRadii[] = {0.10, 0.09, 0.09, 0.09,
                                            0.07, 0.07, 0.06, 0.06};

        SavedState saved(model, data);

        const int ndof = std::min(static_cast<int>(q.size()), model->nq);
        for (int i = 0; i < ndof; i++)
            data->qpos[i] = q[i];
        mju_zero(data->qvel, model->nv);
        mj_kinematics(model, data);

        bool hit = false;
        const int nb = std::min(model->nbody, 9);   // body 0 = world
        for (int b = 1; b < nb && !hit; b++) {
            const Eigen::Vector3d bp(data->xpos[3*b],
                                     data->xpos[3*b+1],
                                     data->xpos[3*b+2]);
            const double lr = (b - 1 < 8) ? kLinkRadii[b - 1] : 0.06;
            for (const auto& s : obstacles_) {
                if ((bp - s.center).norm() < s.radius + lr) {
                    hit = true;
                    break;
                }
            }
        }

        return hit;   // SavedState dtor restores state
    }

    /**
     * Minimum clearance between any link bounding sphere and any obstacle [m].
     * Negative → in collision.
     */
    double configMinClearance(const std::vector<double>& q,
                               mjModel* model, mjData* data) const
    {
        static const double kLinkRadii[] = {0.10, 0.09, 0.09, 0.09,
                                            0.07, 0.07, 0.06, 0.06};

        SavedState saved(model, data);

        const int ndof = std::min(static_cast<int>(q.size()), model->nq);
        for (int i = 0; i < ndof; i++)
            data->qpos[i] = q[i];
        mju_zero(data->qvel, model->nv);
        mj_kinematics(model, data);

        double best = std::numeric_limits<double>::infinity();
        const int nb = std::min(model->nbody, 9);
        for (int b = 1; b < nb; b++) {
            const Eigen::Vector3d bp(data->xpos[3*b],
                                     data->xpos[3*b+1],
                                     data->xpos[3*b+2]);
            const double lr = (b - 1 < 8) ? kLinkRadii[b - 1] : 0.06;
            for (const auto& s : obstacles_)
                best = std::min(best, (bp - s.center).norm() - s.radius - lr);
        }

        return best;
    }

    // ── Joint-space config query (exact) ──────────────────────────────────────

    /**
     * Full mj_forward + contact count (exact, slower).
     *
     * Detects all MuJoCo contacts including self-collision and floor contacts.
     * The state (qpos, qvel) is saved and restored after the query.
     *
     * @param q       Joint angles [rad].
     * @param model   Pointer to MjModel.
     * @param data    Pointer to MjData (state is saved/restored).
     */
    bool isConfigInCollisionExact(const std::vector<double>& q,
                                   mjModel* model, mjData* data) const
    {
        SavedState saved(model, data);

        const int ndof = std::min(static_cast<int>(q.size()), model->nq);
        for (int i = 0; i < ndof; i++)
            data->qpos[i] = q[i];
        mju_zero(data->qvel, model->nv);
        mj_forward(model, data);

        return data->ncon > 0;
    }   // SavedState dtor restores state + calls mj_forward

    // ── Accessors ─────────────────────────────────────────────────────────────

    const std::vector<Sphere>& obstacles() const { return obstacles_; }
    std::size_t numObstacles()             const { return obstacles_.size(); }

private:

    // ── RAII state save/restore ───────────────────────────────────────────────

    struct SavedState {
        mjModel*             model;
        mjData*              data;
        std::vector<mjtNum>  qpos;
        std::vector<mjtNum>  qvel;

        SavedState(mjModel* m, mjData* d)
            : model(m), data(d),
              qpos(m->nq), qvel(m->nv)
        {
            mju_copy(qpos.data(), d->qpos, m->nq);
            mju_copy(qvel.data(), d->qvel, m->nv);
        }

        ~SavedState() {
            mju_copy(data->qpos, qpos.data(), model->nq);
            mju_copy(data->qvel, qvel.data(), model->nv);
            mj_kinematics(model, data);   // restore FK state
        }
    };

    std::vector<Sphere> obstacles_;
};
