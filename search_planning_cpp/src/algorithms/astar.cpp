#include "algorithms/astar.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

AStar::AStar(Node start, Node goal, const std::string& h_type)
    : BaseAlgorithm(start, goal, h_type) {}

SearchResult AStar::searching() {
    PARENT[s_start] = s_start;
    g[s_start]      = 0.0;
    g[s_goal]       = std::numeric_limits<double>::infinity();
    OPEN.push(f_value(s_start), s_start);

    while (!OPEN.empty()) {
        auto [_, s] = OPEN.pop();
        CLOSED.push_back(s);

        if (s == s_goal) break;

        for (const auto& s_n : get_neighbor(s)) {
            double new_cost = get_g(s) + cost(s, s_n);
            if (new_cost < get_g(s_n)) {
                g[s_n]      = new_cost;
                PARENT[s_n] = s;
                OPEN.push(f_value(s_n), s_n);
            }
        }
    }

    return {extract_path(), CLOSED};
}

std::vector<Node> AStar::get_neighbor(const Node& s) const {
    std::vector<Node> nbrs;
    nbrs.reserve(env.motions.size());
    for (const auto& u : env.motions)
        nbrs.push_back({s.first + u.first, s.second + u.second});
    return nbrs;
}

double AStar::cost(const Node& s_start, const Node& s_goal) const {
    if (is_collision(s_start, s_goal))
        return std::numeric_limits<double>::infinity();
    return std::hypot(s_goal.first - s_start.first, s_goal.second - s_start.second);
}

bool AStar::is_collision(const Node& s_start, const Node& s_end) const {
    if (obs.count(s_start) || obs.count(s_end)) return true;

    // Diagonal move: check both corner cells to prevent corner-cutting through obstacles.
    if (s_start.first != s_end.first && s_start.second != s_end.second) {
        Node s1, s2;
        if (s_end.first - s_start.first == s_start.second - s_end.second) {
            s1 = {std::min(s_start.first, s_end.first), std::min(s_start.second, s_end.second)};
            s2 = {std::max(s_start.first, s_end.first), std::max(s_start.second, s_end.second)};
        } else {
            s1 = {std::min(s_start.first, s_end.first), std::max(s_start.second, s_end.second)};
            s2 = {std::max(s_start.first, s_end.first), std::min(s_start.second, s_end.second)};
        }
        if (obs.count(s1) || obs.count(s2)) return true;
    }
    return false;
}

double AStar::f_value(const Node& s) const {
    return get_g(s) + heuristic(s);
}

double AStar::get_g(const Node& s) const {
    auto it = g.find(s);
    return (it == g.end()) ? std::numeric_limits<double>::infinity() : it->second;
}

Path AStar::extract_path() const {
    Path path = {s_goal};
    Node s = s_goal;
    while (s != s_start) {
        s = PARENT.at(s);
        path.push_back(s);
    }
    return path;
}

double AStar::heuristic(const Node& s) const {
    if (heuristic_type == "manhattan")
        return std::abs(s_goal.first - s.first) + std::abs(s_goal.second - s.second);
    return std::hypot(s_goal.first - s.first, s_goal.second - s.second);
}
