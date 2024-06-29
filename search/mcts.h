//
// Created by Admin on 2024/6/28.
//

#ifndef MCTS_H
#define MCTS_H
#include "../eval/mix9nnue.h"
#include "../game/game.h"
#include "node.h"

// 搜索信息
struct SearchInfo
{
    int   search_time_ms = 0;  // 搜索时间
    int   search_counts  = 0;  // 搜索次数
    int   num_pieces     = 0;
    float win_rate       = 0;
    int   nps            = 0;
};

class Mcts
{
public:
    int        node_count;
    float      win_rate;
               Mcts(float c_puct, unsigned ms);
    ~          Mcts();
    int        get_count() { return node_count; }
    float      get_win_rate() { return win_rate; }
    SearchInfo get_search_info() { return m_search_info; }
    int        get_action(Game *board, Evaluation::mix9::Mix9Evaluator &evaluator);
    void       reset_root(int stage);

private:
    float      c_pupt;
    unsigned   search_time;
    Node      *root;
    SearchInfo m_search_info;
};

#endif //MCTS_H
