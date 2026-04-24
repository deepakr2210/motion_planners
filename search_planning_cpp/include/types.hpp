#pragma once
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <string>

using Node = std::pair<int, int>;

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
using Path    = std::vector<Node>;
using Motions = std::vector<Node>;
