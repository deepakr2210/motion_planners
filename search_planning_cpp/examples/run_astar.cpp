#include "algorithms/astar.hpp"
#include "plotting.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        Node s_start = {5, 5};
        Node s_goal  = {45, 25};

        AStar astar(s_start, s_goal, "euclidean");
        auto [path, visited] = astar.searching();

        std::cout << "A* done: path length=" << path.size()
                  << "  visited=" << visited.size() << "\n";

        Plotting plot(s_start, s_goal);
        plot.delay        = 0.02;  // seconds per animation batch  (increase = slower)
        plot.replay_pause = 1.5;   // seconds to hold before replay (used when n_loops != 1)
        plot.animation(path, visited, "A*", -1);  // -1 = loop forever, 1 = once, N = N times

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
