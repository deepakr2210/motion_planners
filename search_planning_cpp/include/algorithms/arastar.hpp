#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Anytime Repairing A* (ARA*) — runs weighted A* with decreasing epsilon,
// returning progressively better solutions.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/ARAstar.py
class ARAStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("ARAStar: not yet implemented");
    }
};
