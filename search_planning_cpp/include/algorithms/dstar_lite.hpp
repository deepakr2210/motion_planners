#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// D* Lite — optimized D* with a single backward search from goal.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/D_star_Lite.py
class DStarLite : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("DStarLite: not yet implemented");
    }
};
