/**
 * C++ trajectory planner node.
 *
 * - Subscribes to joint STATE on ZMQ port 5555.
 * - Publishes TRAJ on ZMQ port 5557.
 * - Cycles through predefined goal configurations using cubic interpolation.
 *
 * Build:
 *   mkdir -p build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 * Run:
 *   ./build/cpp_planner
 */

#include <chrono>
#include <csignal>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "planner.hpp"

using json = nlohmann::json;
using namespace std::chrono_literals;


// ── Configuration (mirrors config/sim_config.toml) ──────────────────────────

static constexpr int    NDOF             = 7;
static constexpr double TRAJECTORY_DT    = 0.02;   // waypoint spacing [s]
static constexpr double DEFAULT_DURATION = 4.0;    // move duration    [s]

static const std::string STATE_SUB_ADDR = "tcp://localhost:5555";
static const std::string TRAJ_PUB_ADDR  = "tcp://*:5557";

// Topic bytes (must match Python protocol.py)
static const std::string TOPIC_STATE = "STATE";
static const std::string TOPIC_TRAJ  = "TRAJ";

// Predefined goal configurations (rad) — same as Python planner
static const std::vector<std::vector<double>> GOALS = {
    { 0.4, -0.3,  0.2, -1.8,  0.3,  2.0,  1.0},
    {-0.4, -0.3, -0.2, -1.8, -0.3,  2.0, -1.0},
    { 0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398},  // home
};


// ── Utilities ────────────────────────────────────────────────────────────────

double wall_time_s() {
    using clk = std::chrono::system_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Serialize a Trajectory to the TRAJ JSON wire format
json trajectory_to_json(const Trajectory& traj) {
    json wps = json::array();
    for (const auto& wp : traj.waypoints) {
        wps.push_back({
            {"t",   wp.t},
            {"q",   wp.q},
            {"qd",  wp.qd},
        });
    }
    return {
        {"waypoints",  wps},
        {"start_time", traj.start_time},
        {"wall_time",  traj.start_time},
    };
}

// Parse a STATE JSON payload into JointState
JointState state_from_json(const std::string& raw) {
    auto j = json::parse(raw);
    JointState s;
    s.q          = j["q"].get<std::vector<double>>();
    s.qd         = j["qd"].get<std::vector<double>>();
    s.qfrc_bias  = j["qfrc_bias"].get<std::vector<double>>();
    s.sim_time   = j["sim_time"].get<double>();
    s.wall_time  = j["wall_time"].get<double>();
    return s;
}


// ── Main ─────────────────────────────────────────────────────────────────────

volatile bool g_running = true;

void sig_handler(int) { g_running = false; }

int main() {
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    zmq::context_t ctx(1);

    // SUB — receive robot state from sim
    zmq::socket_t state_sub(ctx, zmq::socket_type::sub);
    state_sub.connect(STATE_SUB_ADDR);
    state_sub.set(zmq::sockopt::subscribe, TOPIC_STATE);
    state_sub.set(zmq::sockopt::rcvtimeo, 2000);   // 2 s timeout

    // PUB — send trajectory to controller
    zmq::socket_t traj_pub(ctx, zmq::socket_type::pub);
    traj_pub.bind(TRAJ_PUB_ADDR);

    std::cout << "[planner-cpp] SUB " << STATE_SUB_ADDR << "\n";
    std::cout << "[planner-cpp] PUB " << TRAJ_PUB_ADDR  << "\n";
    std::cout << "[planner-cpp] waiting for first state...\n";

    CubicPlanner planner(NDOF);

    // ── Wait for first state ────────────────────────────────────────────────
    JointState current_state;
    while (g_running) {
        zmq::message_t topic_msg, payload_msg;
        auto r1 = state_sub.recv(topic_msg,   zmq::recv_flags::none);
        auto r2 = state_sub.recv(payload_msg, zmq::recv_flags::none);
        if (r1 && r2) {
            current_state = state_from_json(payload_msg.to_string());
            break;
        }
        std::cerr << "[planner-cpp] no state yet, retrying...\n";
    }

    std::cout << "[planner-cpp] first state received  q[0]="
              << current_state.q[0] << "\n";

    // ── Planning loop ────────────────────────────────────────────────────────
    size_t goal_idx = 0;

    while (g_running) {
        const auto& q_goal = GOALS[goal_idx % GOALS.size()];
        ++goal_idx;

        std::cout << "[planner-cpp] planning to goal " << goal_idx
                  << "  q_goal[0]=" << q_goal[0] << "\n";

        // Plan trajectory
        Trajectory traj = planner.plan(current_state.q, q_goal,
                                       DEFAULT_DURATION, TRAJECTORY_DT);
        traj.start_time = wall_time_s();

        // Serialise and publish (multipart: topic | payload)
        json payload = trajectory_to_json(traj);
        std::string payload_str = payload.dump();

        zmq::message_t t_msg(TOPIC_TRAJ.data(), TOPIC_TRAJ.size());
        zmq::message_t p_msg(payload_str.data(), payload_str.size());
        traj_pub.send(t_msg, zmq::send_flags::sndmore);
        traj_pub.send(p_msg, zmq::send_flags::none);

        std::cout << "[planner-cpp] published " << traj.waypoints.size()
                  << " waypoints over " << DEFAULT_DURATION << "s\n";

        // Wait for trajectory to execute, then read updated state
        std::this_thread::sleep_for(
            std::chrono::duration<double>(DEFAULT_DURATION + 0.5));

        // Drain state socket to get latest reading
        while (g_running) {
            zmq::message_t topic_msg, payload_msg;
            state_sub.set(zmq::sockopt::rcvtimeo, 100);
            auto r1 = state_sub.recv(topic_msg,   zmq::recv_flags::none);
            auto r2 = state_sub.recv(payload_msg, zmq::recv_flags::none);
            if (!r1 || !r2) break;   // timed out — got the latest
            current_state = state_from_json(payload_msg.to_string());
        }
        state_sub.set(zmq::sockopt::rcvtimeo, 2000);

        std::this_thread::sleep_for(500ms);
    }

    std::cout << "\n[planner-cpp] stopped\n";
    return 0;
}
