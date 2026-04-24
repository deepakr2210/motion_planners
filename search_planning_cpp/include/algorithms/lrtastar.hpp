#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Learning Real-Time A* (LRTA*) — updates heuristic table online during execution.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/LRTAstar.py
class LRTAStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("LRTAStar: not yet implemented");
    }
};
