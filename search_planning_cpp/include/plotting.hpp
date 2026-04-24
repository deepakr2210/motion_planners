#pragma once
#include "types.hpp"
#include "env.hpp"
#include <string>

// 2D visualization via matplotlibcpp (C++ wrapper around matplotlib).
// Mirrors libs/PathPlanning/Search_based_Planning/Search_2D/plotting.py
class Plotting {
public:
    Node xI, xG;
    NodeSet obs;

    // Seconds to pause after each animation batch. Increase to slow down (e.g. 0.05).
    double delay = 0.001;

    // Seconds to hold the completed frame before replaying. Used when n_loops != 1.
    double replay_pause = 1.5;

    Plotting(Node start, Node goal);
    void update_obs(const NodeSet& new_obs);

    // n_loops: how many times to replay the animation.
    //   1  = run once then keep window open (default)
    //  -1  = loop forever until window is closed
    //   N  = replay N times then keep window open
    void animation(const Path& path, const Path& visited, const std::string& name,
                   int n_loops = 1);
    void plot_grid(const std::string& name);
    void plot_visited(Path visited, const std::string& color = "gray");
    void plot_path(const Path& path, const std::string& color = "r");

private:
    Env env;
};
