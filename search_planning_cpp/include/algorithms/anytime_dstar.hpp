#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Anytime D* — D* variant with bounded suboptimality guarantee.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Anytime_D_star.py
class AnytimeDStar : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("AnytimeDStar: not yet implemented");
    }
};
