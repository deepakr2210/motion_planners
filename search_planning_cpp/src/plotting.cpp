#include "plotting.hpp"
#include "matplotlibcpp.h"
#include <algorithm>
#include <map>

namespace plt = matplotlibcpp;

Plotting::Plotting(Node start, Node goal)
    : xI(start), xG(goal) {
    obs = env.obs;  // env is default-constructed before the body runs
}

void Plotting::update_obs(const NodeSet& new_obs) {
    obs = new_obs;
}

void Plotting::animation(const Path& path, const Path& visited, const std::string& name,
                          int n_loops) {
    int count = 0;
    while (true) {
        if (count > 0) {
            plt::pause(replay_pause);
            plt::clf();
        }
        plot_grid(name);
        plot_visited(visited);
        plot_path(path);
        count++;

        bool finite_and_done = (n_loops > 0 && count >= n_loops);
        if (finite_and_done) break;

        // Infinite loop: pause to let the user view the completed frame,
        // then replay. Exceptions (e.g. window closed) bubble up to main.
        if (n_loops < 0) plt::pause(replay_pause);
    }
    plt::show();
}

void Plotting::plot_grid(const std::string& name) {
    std::vector<double> obs_x, obs_y;
    obs_x.reserve(obs.size());
    obs_y.reserve(obs.size());
    for (const auto& o : obs) {
        obs_x.push_back(o.first);
        obs_y.push_back(o.second);
    }

    plt::plot(std::vector<double>{(double)xI.first}, std::vector<double>{(double)xI.second}, "bs");
    plt::plot(std::vector<double>{(double)xG.first}, std::vector<double>{(double)xG.second}, "gs");
    plt::plot(obs_x, obs_y, "sk");
    plt::title(name);
    plt::axis("equal");
}

void Plotting::plot_visited(Path visited, const std::string& color) {
    visited.erase(std::remove(visited.begin(), visited.end(), xI), visited.end());
    visited.erase(std::remove(visited.begin(), visited.end(), xG), visited.end());

    int total = static_cast<int>(visited.size());
    int idx   = 0;

    // Plot in batches to match the Python animation cadence (20/30/40 nodes per frame).
    // Batching avoids creating thousands of separate matplotlib artists.
    while (idx < total) {
        int length;
        if      (idx < total / 3)     length = 20;
        else if (idx < total * 2 / 3) length = 30;
        else                          length = 40;

        int end = std::min(idx + length, total);

        std::vector<double> bx, by;
        for (int i = idx; i < end; i++) {
            bx.push_back(visited[i].first);
            by.push_back(visited[i].second);
        }

        plt::plot(bx, by, std::map<std::string, std::string>{
            {"color", color}, {"marker", "o"}, {"linestyle", ""}
        });
        plt::pause(delay);

        idx = end;
    }
    plt::pause(delay);
}

void Plotting::plot_path(const Path& path, const std::string& color) {
    std::vector<double> px, py;
    for (const auto& p : path) {
        px.push_back(p.first);
        py.push_back(p.second);
    }
    plt::plot(px, py, std::map<std::string, std::string>{{"linewidth", "3"}, {"color", color}});
    plt::plot(std::vector<double>{(double)xI.first}, std::vector<double>{(double)xI.second}, "bs");
    plt::plot(std::vector<double>{(double)xG.first}, std::vector<double>{(double)xG.second}, "gs");
    plt::pause(0.01);
}
