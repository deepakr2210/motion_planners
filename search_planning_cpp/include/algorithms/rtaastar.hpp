#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Real-Time Adaptive A* (RTAA*) — bounded lookahead with heuristic learning.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/RTAAStar.py
class RTAAStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("RTAAStar: not yet implemented");
    }
};
