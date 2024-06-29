//
// Created by Admin on 2024/6/28.
//

#include "game.h"

#include <cstring>

Game::Game()
{
    reset();
}

void Game::reset()
{
    memset(board, EMPTY, sizeof(board));
    stage       = 1;
    preAction   = -1;
    numPieces   = 0;
    numColor[0] = numColor[1] = 0;
    currentPlayer             = BLACK;
    for (int i = 0; i < BOARD_LEN * BOARD_LEN; i++) {
        available_moves.insert(i);
    }
}

bool Game::inBoard(int x, int y) const
{
    return x >= 0 && x < boardLen && y >= 0 && y < boardLen;
}

Evaluation::Color Game::get_player_color(int player)
{
    if (player == BLACK)
        return Evaluation::BLACK;
    if (player == WHITE)
        return Evaluation::WHITE;
    return Evaluation::EMPTY;
}

Evaluation::Color Game::get_eval_color() const
{
    if (currentPlayer == BLACK)
        return Evaluation::BLACK;
    if (currentPlayer == WHITE)
        return Evaluation::WHITE;
    return Evaluation::EMPTY;
}

void Game::doAction(int action)
{
    preAction = action;
    numPieces++;
    numColor[currentPlayer]++;
    board[action / boardLen][action % boardLen] = currentPlayer;
    if (stage != 0) {
        currentPlayer = -currentPlayer;
    }
    stage = (stage + 1) % 2;
    history_moves.push_back(action);
    available_moves.erase(action);
}

void Game::undoAction(int action)
{
    numPieces--;
    board[action / boardLen][action % boardLen] = EMPTY;
    history_moves.pop_back();
    available_moves.insert(action);
    preAction = history_moves.empty() ? -1 : history_moves.back();
    if (stage != 0) {
        numColor[currentPlayer]--;
        stage = 0;
    }
    else {
        numColor[-currentPlayer]--;
        currentPlayer = -currentPlayer;
        stage         = 1;
    }
}

int Game::isGameOver() const
{
    if (preAction < 0)
        return EMPTY;

    const int direction[4][2][2] = {{{0, -1}, {0, 1}},
                                    {{-1, 0}, {1, 0}},
                                    {{-1, -1}, {1, 1}},
                                    {{1, -1}, {-1, 1}}};

    int ox     = preAction / boardLen;
    int oy     = preAction % boardLen;
    int player = board[ox][oy];

    if (numColor[player] >= 6) {
        for (size_t i = 0; i < 4; i++) {
            int count = 1;
            for (size_t j = 0; j < 2; j++) {
                for (int x = ox + direction[i][j][0], y = oy + direction[i][j][1];
                     inBoard(x, y) && board[x][y] == player;
                     x += direction[i][j][0], y += direction[i][j][1]) {
                    count++;
                }
            }
            if (count >= 6) {
                return player;
            }
        }
    }

    if (numPieces == boardLen * boardLen)
        return DRAW;

    return EMPTY;
}