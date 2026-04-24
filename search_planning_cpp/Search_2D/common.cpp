#pragma once
// common.cpp — types, environment, queues, and plotting for 2D grid search.
// #include this at the top of each algorithm file. Do not compile standalone.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <functional>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <algorithm>
#include <map>
#include <iostream>

// ── Types ─────────────────────────────────────────────────────────────────────

using Node    = std::pair<int, int>;
using Path    = std::vector<Node>;
using Motions = std::vector<Node>;

struct NodeHash {
    size_t operator()(const Node& n) const {
        size_t h1 = std::hash<int>{}(n.first);
        size_t h2 = std::hash<int>{}(n.second);
        return h1 ^ (h2 * 2654435761u + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using NodeSet = std::unordered_set<Node, NodeHash>;
using NodeMap = std::unordered_map<Node, Node, NodeHash>;
using CostMap = std::unordered_map<Node, double, NodeHash>;

// ── Environment ───────────────────────────────────────────────────────────────
// Mirrors libs/PathPlanning/Search_based_Planning/Search_2D/env.py

struct Env {
    int x_range = 51;
    int y_range = 31;
    Motions motions = {
        {-1,  0}, {-1,  1}, { 0,  1}, { 1,  1},
        { 1,  0}, { 1, -1}, { 0, -1}, {-1, -1}
    };
    NodeSet obs;

    Env() : obs(build_obs()) {}
    void update_obs(const NodeSet& o) { obs = o; }

private:
    NodeSet build_obs() {
        NodeSet o;
        for (int i = 0; i < x_range; i++) { o.insert({i, 0}); o.insert({i, y_range-1}); }
        for (int i = 0; i < y_range; i++) { o.insert({0, i}); o.insert({x_range-1, i}); }
        for (int i = 10; i <= 20; i++) o.insert({i, 15});
        for (int i = 0;  i <  15; i++) o.insert({20, i});
        for (int i = 15; i <  30; i++) o.insert({30, i});
        for (int i = 0;  i <  16; i++) o.insert({40, i});
        return o;
    }
};

// ── Queues ────────────────────────────────────────────────────────────────────

struct PriorityQueue {
    using Item = std::pair<double, Node>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    void push(double p, const Node& n) { pq.push({p, n}); }
    std::pair<double, Node> pop() { auto t = pq.top(); pq.pop(); return t; }
    bool empty() const { return pq.empty(); }
};

struct QueueFIFO {
    std::deque<Node> dq;
    void push(const Node& n) { dq.push_back(n); }
    Node pop() { auto f = dq.front(); dq.pop_front(); return f; }
    bool empty() const { return dq.empty(); }
};

struct QueueLIFO {
    std::deque<Node> dq;
    void push(const Node& n) { dq.push_back(n); }
    Node pop() { auto b = dq.back(); dq.pop_back(); return b; }
    bool empty() const { return dq.empty(); }
};

// ── Plotting ──────────────────────────────────────────────────────────────────
// Mirrors libs/PathPlanning/Search_based_Planning/Search_2D/plotting.py

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

struct Plotting {
    Env     _env;           // declared first — obs is copy-initialized from this
    Node    xI, xG;
    NodeSet obs;
    double  delay        = 0.001;  // seconds per animation batch (increase to slow down)
    double  replay_pause = 1.5;    // seconds to hold before replaying

    Plotting(Node start, Node goal)
        : xI(start), xG(goal), obs(_env.obs) {}

    void update_obs(const NodeSet& o) { obs = o; }

    // n_loops: 1 = play once (default), -1 = loop forever, N = replay N times
    void animation(const Path& path, const Path& visited,
                   const std::string& name, int n_loops = 1) {
        int count = 0;
        while (true) {
            if (count > 0) { plt::pause(replay_pause); plt::clf(); }
            plot_grid(name);
            plot_visited(visited);
            plot_path(path);
            count++;
            if (n_loops > 0 && count >= n_loops) break;
            if (n_loops < 0) plt::pause(replay_pause);
        }
        plt::show();
    }

    void plot_grid(const std::string& name) {
        std::vector<double> ox, oy;
        for (const auto& o : obs) { ox.push_back(o.first); oy.push_back(o.second); }
        plt::plot(std::vector<double>{(double)xI.first}, std::vector<double>{(double)xI.second}, "bs");
        plt::plot(std::vector<double>{(double)xG.first}, std::vector<double>{(double)xG.second}, "gs");
        plt::plot(ox, oy, "sk");
        plt::title(name);
        plt::axis("equal");
    }

    void plot_visited(Path visited, const std::string& color = "gray") {
        visited.erase(std::remove(visited.begin(), visited.end(), xI), visited.end());
        visited.erase(std::remove(visited.begin(), visited.end(), xG), visited.end());
        int total = (int)visited.size(), idx = 0;
        while (idx < total) {
            int len = (idx < total/3) ? 20 : (idx < total*2/3) ? 30 : 40;
            int end = std::min(idx + len, total);
            std::vector<double> bx, by;
            for (int i = idx; i < end; i++) {
                bx.push_back(visited[i].first);
                by.push_back(visited[i].second);
            }
            plt::plot(bx, by, std::map<std::string,std::string>{
                {"color", color}, {"marker", "o"}, {"linestyle", ""}
            });
            plt::pause(delay);
            idx = end;
        }
        plt::pause(delay);
    }

    void plot_path(const Path& path, const std::string& color = "r") {
        std::vector<double> px, py;
        for (const auto& p : path) { px.push_back(p.first); py.push_back(p.second); }
        plt::plot(px, py, std::map<std::string,std::string>{{"linewidth","3"},{"color",color}});
        plt::plot(std::vector<double>{(double)xI.first}, std::vector<double>{(double)xI.second}, "bs");
        plt::plot(std::vector<double>{(double)xG.first}, std::vector<double>{(double)xG.second}, "gs");
        plt::pause(delay);
    }
};
