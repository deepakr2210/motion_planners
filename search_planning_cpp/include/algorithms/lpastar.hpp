#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Lifelong Planning A* (LPA*) — incrementally repairs the solution as costs change.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/LPAstar.py
class LPAStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("LPAStar: not yet implemented");
    }
};
