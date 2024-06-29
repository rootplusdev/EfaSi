//
// Created by Admin on 2024/6/28.
//

#ifndef GAME_H
#define GAME_H

#define BOARD_LEN 19
#include "../eval/mix9nnue.h"

#include <string>
#include <vector>
#include <unordered_set>


class Game
{
public:
    // 棋盘大小
    const int boardLen = BOARD_LEN;
    const int EMPTY                       = 0;
    const int BLACK                       = 1;
    const int WHITE                       = -1;
    const int DRAW                        = 2;
    const int FLAT                        = 3;
    int       stage                       = 1;
    int       board[BOARD_LEN][BOARD_LEN] = {EMPTY};
    std::unordered_set<int> available_moves;
    std::vector<int> history_moves;

    Evaluation::Color get_eval_color() const;
    Evaluation::Color get_player_color(int player);
    // 记录上一步的动作
    int preAction = -1;
    // 棋盘棋子数量
    int numPieces = 0;
    // 每种颜色的棋子的数量
    int numColor[2]   = {0};
    int currentPlayer = BLACK;
    Game();
    void reset();
    bool inBoard(int x, int y) const;
    void doAction(int action);
    void undoAction(int action);
    int isGameOver() const;
};

#endif  // GAME_H
