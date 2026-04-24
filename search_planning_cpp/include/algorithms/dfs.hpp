#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Depth-First Search — LIFO expansion, not optimal.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/DFS.py
class DFS : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("DFS: not yet implemented");
    }
};
