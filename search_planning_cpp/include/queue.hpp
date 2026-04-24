#pragma once
#include "types.hpp"
#include <queue>
#include <deque>
#include <functional>

// Min-heap priority queue — stores (priority, node) pairs.
// Mirrors the QueuePrior in Search_2D/queue.py
class PriorityQueue {
    using Item = std::pair<double, Node>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
public:
    void push(double priority, const Node& node) { pq.push({priority, node}); }
    std::pair<double, Node> pop() { auto t = pq.top(); pq.pop(); return t; }
    bool empty() const { return pq.empty(); }
    size_t size() const { return pq.size(); }
};

// FIFO — for BFS
class QueueFIFO {
    std::deque<Node> dq;
public:
    void push(const Node& n) { dq.push_back(n); }
    Node pop() { auto f = dq.front(); dq.pop_front(); return f; }
    bool empty() const { return dq.empty(); }
};

// LIFO — for DFS
class QueueLIFO {
    std::deque<Node> dq;
public:
    void push(const Node& n) { dq.push_back(n); }
    Node pop() { auto b = dq.back(); dq.pop_back(); return b; }
    bool empty() const { return dq.empty(); }
};
