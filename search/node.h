//
// Created by Admin on 2024/6/28.
//

#ifndef NODE_H
#define NODE_H
#include <map>
static const int MAX_CHILDREN_NUM = 64;

class Node
{
public:
    float                  win_rate;
    float                  up_bound;
    float                  prior_prob;
    int                    visits;
    float                  c_puct;
    Node                  *parent;
    int                    stage;
    int                    num_children;
    bool                   is_extend;
    std::array<std::pair<int, std::pair<float, Node*>>, MAX_CHILDREN_NUM> children;
                           Node(float prior_prob, int stage, float c_puct, Node *parent);
    ~                      Node();
    std::pair<int, Node *> select_move();
    void                   expand_node(int action, float prob, int index);
    void                   update_win_rate(float value);
    void                   backup(float value);
    float                  get_ucb_value();
    float                  get_ucb_value(float parent_rate, float prob);
    bool                   is_leaf();
};

#endif  // NODE_H
