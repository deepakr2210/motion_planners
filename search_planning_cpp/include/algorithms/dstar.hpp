#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Dynamic A* (D*) — replanning algorithm for environments with changing costs.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/D_star.py
class DStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("DStar: not yet implemented");
    }
};
