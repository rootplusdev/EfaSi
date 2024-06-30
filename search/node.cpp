//
// Created by Admin on 2024/6/28.
//

#include "node.h"
#include <iostream>

#include <cmath>

Node::Node(float prior_prob, int stage, float c_puct, Node *parent)
    : prior_prob(prior_prob)
    , stage(stage)
    , c_puct(c_puct)
    , win_rate(0.0f)
    , up_bound(0.0f)
    , parent(parent)
    , visits(0)
    , num_children(0){};

Node::~Node()
{
    for (auto [_, child] : children) {
        if (child.second) delete child.second;
    }
}

std::pair<int, Node *> Node::select_move()
{
//    float next_win_rate = (stage == 1) ? 0.1f-win_rate : win_rate-0.1f;
//
//    for (auto [move, child] : children) {
//        if (child->visits == 0) child->win_rate = next_win_rate;
//    }
    int   best_move;
    float best_value = -10000.0f;
    float value;
    float best_prob;
    Node *best_child = nullptr;

    for (auto [move, child] : children) {
        if (child.second){
            value = child.second->get_ucb_value();
            if (value > best_value) {
                best_move  = move;
                best_value = value;
                best_child = child.second;
            }
        }
        else{
            value = get_ucb_value(win_rate, child.first);
            if (value > best_value){
                best_move = move;
                best_value = value;
                best_child = nullptr;
                best_prob = child.first;
            }
        }
    }
    if (!best_child) best_child = new Node(best_prob, (stage+1)%2, c_puct, this);
    num_children++;
    return {best_move, best_child};
}

void Node::expand_node(int action, float prob, int index)
{
    // 虚假的扩展
//    auto child = new Node(prob, (stage + 1) % 2, c_puct, this);
    children[index] = {action, {prob, nullptr}};
}

void Node::update_win_rate(float value)
{
    win_rate = (visits * win_rate + value) / (visits + 1);
    visits++;
}

void Node::backup(float value)
{
    if (parent) {
        if (stage % 2 == 0)
            parent->backup(-value);

        else
            parent->backup(value);
    }
    update_win_rate(value);
}

float Node::get_ucb_value()
{
    up_bound = c_puct * prior_prob * sqrt(parent->visits) / (1 + visits);
    win_rate = (stage == 0) ? -win_rate : win_rate;
    return win_rate + up_bound;
}

float Node::get_ucb_value(float parent_rate, float prob)
{
    float u = c_puct * prob * sqrt(visits);
    float q = (stage == 1) ? 0.1f-parent_rate : parent_rate-0.1f;
    return u + q;
}

bool Node::is_leaf()
{
    return num_children == 0;
}
