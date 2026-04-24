#pragma once
#include "../types.hpp"
#include "../env.hpp"
#include <string>

struct SearchResult {
    Path path;
    Path visited;
};

// Common base for all 2D search algorithms.
class BaseAlgorithm {
public:
    Node s_start, s_goal;
    std::string heuristic_type;
    Env env;
    NodeSet obs;

    BaseAlgorithm(Node start, Node goal, const std::string& h_type)
        : s_start(start), s_goal(goal), heuristic_type(h_type), obs(env.obs) {}

    virtual ~BaseAlgorithm() = default;
    virtual SearchResult searching() = 0;
};
