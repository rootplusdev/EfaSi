#include "eval/mix9nnue.h"
#include "game/game.h"
#include "search/mcts.h"

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

template <typename F>
void printBoard(std::ostream &os, int boardSize, F &&posTextFunc, int textWidth = 1)
{
    for (int y = 0; y < boardSize; y++)
        for (int x = 0; x < boardSize; x++) {
            if (x != 0 || y != 0)
                os << ' ';
            if (x == 0 && y != 0)
                os << '\n';
            posTextFunc(os, x, y);
            if (x == boardSize - 1)
                os << ' ' << y + 1;
        }

    os << '\n';
    for (int x = 0; x < boardSize; x++)
        os << std::setw(textWidth) << char(x + 65) << " ";
    os << '\n';
};

void printBoardAndEval(std::array<Evaluation::Color, 19 * 19> &board,
                       Evaluation::mix9::Mix9Evaluator        &evaluator,
                       Evaluation::Color                       side)
{
    constexpr char SideStr[] = {'X', 'O', 'W', 'E'};
    std::cout << "Board: side=" << SideStr[side] << std::endl;
    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        Evaluation::Color c = board[y * 19 + x];
        os << (c == Evaluation::BLACK ? 'X' : c == Evaluation::WHITE ? 'O' : '.');
    });

    Evaluation::ValueType value = evaluator.evaluateValue(side);
    std::cout << std::fixed << std::setprecision(4) << "Value: " << value.value()
              << ", win: " << value.win() << ", loss: " << value.loss()
              << ", draw: " << value.draw() << std::endl;

    Evaluation::PolicyBuffer policyBuffer(19);
    for (int index = 0; index < 19 * 19; index++)
        if (board[index] == Evaluation::EMPTY)
            policyBuffer.setComputeFlag(index);
    evaluator.evaluatePolicy(side, policyBuffer);

    std::cout << "Raw Policy:" << std::endl;
    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        if (policyBuffer.getComputeFlag(x, y)) {
            float policy = policyBuffer(x, y);
            os << std::setw(6) << std::fixed << std::setprecision(2) << policy;
        }
        else
            os << std::setw(6) << '_';
    });

    policyBuffer.applySoftmax();

    std::cout << "Policy:" << std::endl;
    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        float policy = policyBuffer(x, y);
        os << std::setw(4) << (int)std::round(policy * 1000);
    });
}

void test_mix9net()
{
    std::string blackWeightPath = "C:/Users/Admin/Desktop/C6NNUE/weights/model.bin";
    std::string whiteWeightPath = "C:/Users/Admin/Desktop/C6NNUE/weights/model.bin";

    std::unique_ptr<Evaluation::mix9::Mix9Evaluator> evaluator =
        std::make_unique<Evaluation::mix9::Mix9Evaluator>(19, blackWeightPath, whiteWeightPath);
    std::cout << "Mix9NNUE evaluation initialized." << std::endl;

    std::array<Evaluation::Color, 19 * 19> board;
    std::fill(board.begin(), board.end(), Evaluation::EMPTY);
    evaluator->initEmptyBoard();
    printBoardAndEval(board, *evaluator, Evaluation::BLACK);

    board[9 * 19 + 9] = Evaluation::BLACK;
    evaluator->move(Evaluation::BLACK, 9, 9);
    printBoardAndEval(board, *evaluator, Evaluation::WHITE);

    board[8 * 19 + 8] = Evaluation::WHITE;
    evaluator->move(Evaluation::WHITE, 8, 8);
    printBoardAndEval(board, *evaluator, Evaluation::WHITE);

    board[9 * 19 + 7] = Evaluation::WHITE;
    evaluator->move(Evaluation::WHITE, 7, 9);
    printBoardAndEval(board, *evaluator, Evaluation::BLACK);

    board[7 * 19 + 9] = Evaluation::BLACK;
    evaluator->move(Evaluation::BLACK, 9, 7);
    printBoardAndEval(board, *evaluator, Evaluation::BLACK);

    board[8 * 19 + 9] = Evaluation::BLACK;
    evaluator->move(Evaluation::BLACK, 9, 8);
    printBoardAndEval(board, *evaluator, Evaluation::WHITE);
}

void test_speed_mix9net()
{
    std::string model_path = "/Users/kim/Desktop/C6NNUE/weights/model.bin";
    std::unique_ptr<Evaluation::mix9::Mix9Evaluator> evaluator =
        std::make_unique<Evaluation::mix9::Mix9Evaluator>(19, model_path, model_path);

    const size_t N         = 100000;
    auto         startTime = std::chrono::high_resolution_clock::now();
    evaluator->initEmptyBoard();
    for (size_t i = 0; i < N; i++) {
        Evaluation::Color side = Evaluation::BLACK;
        evaluator->move(side, 9, 9);
        Evaluation::ValueType    value = evaluator->evaluateValue(side);
        Evaluation::PolicyBuffer policyBuffer(19);
        for (int index = 0; index < 19 * 19; index++)
            policyBuffer.setComputeFlag(index);
        evaluator->evaluatePolicy(side, policyBuffer);
        policyBuffer.applySoftmax();
        evaluator->undo(side, 9, 9);
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    auto elapsedMilliSeconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Time: " << elapsedMilliSeconds << "ms"
              << ", "
              << "Nps: " << 1000 * N / elapsedMilliSeconds << std::endl;
}

void do_one_step(Mcts *mcts, Game *game, Evaluation::mix9::Mix9Evaluator &evaluator)
{
    constexpr char SideStr[] = {'O', 'E', 'X', 'W'};
    auto           move      = mcts->get_action(game, evaluator);
    std::cout << "Board: side=" << SideStr[game->currentPlayer + 1] << std::endl;

    Evaluation::PolicyBuffer policyBuffer(19);
    for (int index = 0; index < 19 * 19; index++)
        if (game->board[index / 19][index % 19] == game->EMPTY)
            policyBuffer.setComputeFlag(index);
    evaluator.evaluatePolicy(game->get_eval_color(), policyBuffer);

    policyBuffer.applySoftmax();

    std::cout << "Policy:" << std::endl;
    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        float policy = policyBuffer(x, y);
        os << std::setw(4) << (int)std::round(policy * 1000);
    });

    evaluator.move(game->get_eval_color(), move % BOARD_LEN, move / BOARD_LEN);
    game->doAction(move);

    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        Evaluation::Color c = game->get_player_color(game->board[y][x]);
        os << (c == Evaluation::BLACK ? 'X' : c == Evaluation::WHITE ? 'O' : '.');
    });
    std::cout << "Move: " << move << std::endl;
}

void test_mcts()
{
    auto        game       = new Game();
    std::string model_path = "/Users/kim/Desktop/C6NNUE/weights/model.bin";
    std::unique_ptr<Evaluation::mix9::Mix9Evaluator> evaluator =
        std::make_unique<Evaluation::mix9::Mix9Evaluator>(19, model_path, model_path);
    evaluator->initEmptyBoard();
    auto mcts = new Mcts(1.0, 5000);

    constexpr char SideStr[] = {'O', 'E', 'X', 'W'};
    std::cout << "Board: side=" << SideStr[game->currentPlayer + 1] << std::endl;
    evaluator->move(game->get_eval_color(), 9, 9);
    game->doAction(180);
    printBoard(std::cout, 19, [&](auto &os, int x, int y) {
        Evaluation::Color c = game->get_player_color(game->board[y][x]);
        os << (c == Evaluation::BLACK ? 'X' : c == Evaluation::WHITE ? 'O' : '.');
    });

    do_one_step(mcts, game, *evaluator);
    do_one_step(mcts, game, *evaluator);
    do_one_step(mcts, game, *evaluator);
    do_one_step(mcts, game, *evaluator);
    do_one_step(mcts, game, *evaluator);
    do_one_step(mcts, game, *evaluator);
}

int main()
{
    test_mcts();
    return 0;
}