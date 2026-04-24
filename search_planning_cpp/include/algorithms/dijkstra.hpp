#pragma once
#include "base.hpp"
#include "../queue.hpp"
#include <stdexcept>

// Dijkstra's algorithm — identical to A* but f(s) = g(s) only (h=0).
// Reference: libs/PathPlanning/Search_based_Planning/Search_2D/Dijkstra.py
class Dijkstra : public BaseAlgorithm {
public:
    using BaseAlgorithm::BaseAlgorithm;
    SearchResult searching() override {
        throw std::runtime_error("Dijkstra: not yet implemented");
    }
};
