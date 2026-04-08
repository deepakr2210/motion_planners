#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>

#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <zmq.hpp>

#include "diff_ik_control.hpp"

using json = nlohmann::json;

int main(int argc, char** argv)
{
    const std::string cfg_path = (argc > 1) ? argv[1] : "config/sim_config.yaml";
    YAML::Node cfg = YAML::LoadFile(cfg_path);

    const std::string model_path = cfg["robot"]["robot_path"].as<std::string>();
    const int         ndof       = cfg["robot"]["ndof"].as<int>();
    const double      dt         = cfg["diff_ik_control"]["dt"].as<double>();
    const double      rate_hz    = cfg["diff_ik_control"]["rate_hz"] ?
                                   cfg["diff_ik_control"]["rate_hz"].as<double>() : 100.0;
    const std::vector<double> home_q =
        cfg["robot"]["home_q"].as<std::vector<double>>();

    // Load MuJoCo model
    char err[1000];
    mjModel* model = mj_loadXML(model_path.c_str(), nullptr, err, sizeof(err));
    if (!model) { std::cerr << "[diff_ik] model load failed: " << err << "\n"; return 1; }
    mjData* data = mj_makeData(model);

    DiffIKControl controller(model, data, "ee_site", dt);

    // ZMQ sockets
    zmq::context_t ctx;

    zmq::socket_t state_sub(ctx, zmq::socket_type::sub);
    state_sub.connect(cfg["zmq"]["state_sub_addr"].as<std::string>());
    state_sub.set(zmq::sockopt::subscribe, "STATE");

    zmq::socket_t twist_sub(ctx, zmq::socket_type::sub);
    twist_sub.connect(cfg["zmq"]["twist_sub_addr"].as<std::string>());
    twist_sub.set(zmq::sockopt::subscribe, "TWIST");

    zmq::socket_t cmd_push(ctx, zmq::socket_type::push);
    cmd_push.connect(cfg["zmq"]["cmd_push_addr"].as<std::string>());

    zmq::pollitem_t items[] = {
        { state_sub, 0, ZMQ_POLLIN, 0 },
        { twist_sub, 0, ZMQ_POLLIN, 0 },
    };

    std::vector<double> q_cmd = home_q;
    std::vector<double> v_cmd(6, 0.0);

    const auto rate_ns = std::chrono::nanoseconds(static_cast<long>(1e9 / rate_hz));
    auto t_next = std::chrono::steady_clock::now();

    std::cout << "[diff_ik] running at " << rate_hz << " Hz\n";

    while (true) {
        zmq::poll(items, 2, std::chrono::milliseconds(0));

        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t topic, payload;
            state_sub.recv(topic);
            state_sub.recv(payload);
            // state available here for monitoring — not used for integration
        }

        if (items[1].revents & ZMQ_POLLIN) {
            zmq::message_t topic, payload;
            twist_sub.recv(topic);
            twist_sub.recv(payload);
            auto j = json::parse(payload.to_string_view());
            v_cmd  = j["twist"].get<std::vector<double>>();
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= t_next) {
            t_next += rate_ns;

            q_cmd = controller.execute(q_cmd, v_cmd);

            json cmd;
            cmd["values"]    = q_cmd;
            cmd["mode"]      = "position";
            cmd["wall_time"] = 0.0;

            std::string payload = cmd.dump();
            zmq::message_t msg(payload.size());
            std::memcpy(msg.data(), payload.data(), payload.size());
            cmd_push.send(msg, zmq::send_flags::none);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    mj_deleteData(data);
    mj_deleteModel(model);
}
