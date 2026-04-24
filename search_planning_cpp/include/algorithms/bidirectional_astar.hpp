#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Bidirectional A* — simultaneous forward (start) and backward (goal) search.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Bidirectional_a_star.py
class BidirectionalAStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("BidirectionalAStar: not yet implemented");
    }
};
