#include "env.hpp"

Env::Env() {
    motions = {
        {-1,  0}, {-1,  1}, { 0,  1}, { 1,  1},
        { 1,  0}, { 1, -1}, { 0, -1}, {-1, -1}
    };
    obs = obs_map();
}

void Env::update_obs(const NodeSet& new_obs) {
    obs = new_obs;
}

NodeSet Env::obs_map() {
    NodeSet o;
    int x = x_range, y = y_range;

    for (int i = 0; i < x; i++) { o.insert({i, 0}); o.insert({i, y - 1}); }
    for (int i = 0; i < y; i++) { o.insert({0, i}); o.insert({x - 1, i}); }

    for (int i = 10; i <= 20; i++) o.insert({i, 15});
    for (int i = 0;  i <  15; i++) o.insert({20, i});

    for (int i = 15; i <  30; i++) o.insert({30, i});
    for (int i = 0;  i <  16; i++) o.insert({40, i});

    return o;
}
