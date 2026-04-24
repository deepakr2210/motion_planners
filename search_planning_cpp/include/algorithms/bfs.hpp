#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Breadth-First Search — uniform cost, FIFO expansion.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/BFS.py
class BFS : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("BFS: not yet implemented");
    }
};
