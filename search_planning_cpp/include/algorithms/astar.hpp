#pragma once
#include "base.hpp"
#include "../queue.hpp"

// A* search on a 2D grid with 8-connected neighbors.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Astar.py
class AStar : public BaseAlgorithm {
public:
    AStar(Node start, Node goal, const std::string& heuristic_type);
    SearchResult searching() override;

private:
    PriorityQueue OPEN;
    Path          CLOSED;
    NodeMap       PARENT;
    CostMap       g;

    std::vector<Node> get_neighbor(const Node& s) const;
    double cost(const Node& s_start, const Node& s_goal) const;
    bool   is_collision(const Node& s_start, const Node& s_end) const;
    double f_value(const Node& s) const;
    double get_g(const Node& s) const;
    Path   extract_path() const;
    double heuristic(const Node& s) const;
};
