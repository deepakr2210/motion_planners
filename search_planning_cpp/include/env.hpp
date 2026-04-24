#pragma once
#include "types.hpp"

// 2D grid environment.
// Mirrors libs/PathPlanning/Search_based_Planning/Search_2D/env.py
class Env {
public:
    int x_range = 51;
    int y_range = 31;
    Motions motions;
    NodeSet obs;

    Env();
    void update_obs(const NodeSet& new_obs);

private:
    NodeSet obs_map();
};
