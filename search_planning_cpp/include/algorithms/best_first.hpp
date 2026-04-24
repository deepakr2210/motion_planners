#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Best-First (greedy) search — f(s) = h(s) only.
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Best_First.py
class BestFirst : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("BestFirst: not yet implemented");
    }
};
