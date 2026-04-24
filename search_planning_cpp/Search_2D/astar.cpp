#include "common.cpp"

// A* search — 8-connected grid, lazy-deletion open set.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Astar.py

struct AStar {
    Node   s_start, s_goal;
    std::string heuristic_type;
    Env    env;

    AStar(Node start, Node goal, const std::string& h = "euclidean")
        : s_start(start), s_goal(goal), heuristic_type(h) {}

    std::pair<Path, Path> searching() {
        PriorityQueue open;
        Path     closed;
        NodeMap  parent;
        CostMap  g;

        parent[s_start] = s_start;
        g[s_start]      = 0.0;
        g[s_goal]       = std::numeric_limits<double>::infinity();
        open.push(f(s_start, g), s_start);

        while (!open.empty()) {
            auto [_, s] = open.pop();
            closed.push_back(s);
            if (s == s_goal) break;
            for (const auto& sn : neighbors(s)) {
                double nc = get_g(g, s) + cost(s, sn);
                if (nc < get_g(g, sn)) {
                    g[sn]      = nc;
                    parent[sn] = s;
                    open.push(f(sn, g), sn);
                }
            }
        }
        return {extract_path(parent), closed};
    }

private:
    std::vector<Node> neighbors(const Node& s) const {
        std::vector<Node> ns;
        for (const auto& u : env.motions)
            ns.push_back({s.first + u.first, s.second + u.second});
        return ns;
    }

    double cost(const Node& a, const Node& b) const {
        if (collision(a, b)) return std::numeric_limits<double>::infinity();
        return std::hypot(b.first - a.first, b.second - a.second);
    }

    bool collision(const Node& a, const Node& b) const {
        if (env.obs.count(a) || env.obs.count(b)) return true;
        if (a.first != b.first && a.second != b.second) {
            Node s1, s2;
            if (b.first - a.first == a.second - b.second) {
                s1 = {std::min(a.first, b.first), std::min(a.second, b.second)};
                s2 = {std::max(a.first, b.first), std::max(a.second, b.second)};
            } else {
                s1 = {std::min(a.first, b.first), std::max(a.second, b.second)};
                s2 = {std::max(a.first, b.first), std::min(a.second, b.second)};
            }
            if (env.obs.count(s1) || env.obs.count(s2)) return true;
        }
        return false;
    }

    double get_g(const CostMap& g, const Node& s) const {
        auto it = g.find(s);
        return it == g.end() ? std::numeric_limits<double>::infinity() : it->second;
    }

    double f(const Node& s, const CostMap& g) const { return get_g(g, s) + h(s); }

    double h(const Node& s) const {
        if (heuristic_type == "manhattan")
            return std::abs(s_goal.first - s.first) + std::abs(s_goal.second - s.second);
        return std::hypot(s_goal.first - s.first, s_goal.second - s.second);
    }

    Path extract_path(const NodeMap& parent) const {
        Path path = {s_goal};
        Node s = s_goal;
        while (s != s_start) { s = parent.at(s); path.push_back(s); }
        return path;
    }
};

int main() {
    try {
        Node start = {5, 5}, goal = {45, 25};

        AStar astar(start, goal, "euclidean");
        auto [path, visited] = astar.searching();
        std::cout << "A*  path=" << path.size() << "  visited=" << visited.size() << "\n";

        Plotting plot(start, goal);
        plot.delay        = 0.02;   // seconds per batch  — increase to slow down
        plot.replay_pause = 1.5;    // seconds between loops
        plot.animation(path, visited, "A*", -1);  // -1 = loop forever
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
