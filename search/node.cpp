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
    , num_children(0)
    ,is_extend(false){};

Node::~Node()
{
    for (auto [_, child] : children) {
        if (child.second) delete child.second;
    }
}

std::pair<int, Node *> Node::select_move()
{
    int   best_move;
    float best_value = -10000.0f;
    float value;
    float best_prob;
    int best_index;
    Node *best_child = nullptr;

    for (int i = 0; i < children.size(); ++i) {
        if (children[i].second.second){
            value = children[i].second.second->get_ucb_value();
            if (value > best_value) {
                best_move  = children[i].first;
                best_value = value;
                best_child = children[i].second.second;
            }
        }
        else{
            value = get_ucb_value(win_rate, children[i].second.first);
            if (value > best_value){
                best_move = children[i].first;
                best_value = value;
                best_child = nullptr;
                best_prob = children[i].second.first;
                best_index = i;
            }
        }
    }
    if (!best_child){
        best_child = new Node(best_prob, (stage+1)%2, c_puct, this);
        children[best_index].second.second = best_child;
    }
    num_children++;
    return {best_move, best_child};
}

void Node::expand_node(int action, float prob, int index)
{
    // 虚假的扩展
//  auto child = new Node(prob, (stage + 1) % 2, c_puct, this);
    children[index] = {action, {prob, nullptr}};
    is_extend = true;
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
    return u + parent_rate - 0.1;
}

bool Node::is_leaf()
{
    return !is_extend;
}
