//
// Created by Admin on 2024/6/28.
//

#include "mcts.h"

#include <cfloat>
#include <iostream>
#include <queue>

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

Mcts::Mcts(float c_puct, unsigned ms)
    : node_count(0)
    , win_rate(0)
    , c_pupt(c_puct)
    , search_time(ms)
    , root(new Node(1.0, 0, c_puct, nullptr)) {};

Mcts::~Mcts()
{
    delete root;
}

void Mcts::reset_root(int stage)
{
    delete root;
    root = new Node(1.0f, stage, 1.0f, nullptr);
}

int Mcts::get_action(Game *board, Evaluation::mix9::Mix9Evaluator &evaluator)
{
    srand((unsigned)time(0));
    auto         startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, Evaluation::Color>> simulation_path;
    reset_root(board->stage);
    int num_count = 0;
//    for (int i = 0; i < 150000; ++i) {
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() < search_time) {
        num_count++;
        Node *node = root;
        while (!node->is_leaf()) {
            auto [selected_move, selected_node] = node->select_move();
            // std::cout << "select x=" << selected_move % BOARD_LEN
            //           << ", y=" << selected_move / BOARD_LEN
            //           << ", color=" << int(boardCopy.get_eval_color()) << std::endl;
            evaluator.move(board->get_eval_color(),
                           selected_move % BOARD_LEN,
                           selected_move / BOARD_LEN);
            simulation_path.emplace_back(selected_move, board->get_eval_color());
            board->doAction(selected_move);
            node = selected_node;
        }
        int   isOverAndWinner = board->isGameOver();
        float value           = 0.0f;
        if (isOverAndWinner == board->EMPTY) {
            Evaluation::ValueType v = evaluator.evaluateValue(board->get_eval_color());
            value                   = v.winLossRate();

            Evaluation::PolicyBuffer policyBuffer(BOARD_LEN);
            for (auto &legal_move : board->available_moves)
                policyBuffer.setComputeFlag(legal_move);
            evaluator.evaluatePolicy(board->get_eval_color(), policyBuffer);
            policyBuffer.applySoftmax();

            // 最大堆，默认使用std::less<>，即大的元素优先级高
            std::priority_queue<std::pair<float, int>> maxHeap;

            // 插入元素到最大堆中
            for (int availableAction : board->available_moves) {
                float priority = policyBuffer(availableAction);
                maxHeap.emplace(priority, availableAction);
            }

            // 取出堆中的前30个优先级最高的元素并展开节点
            // float pune_prior = 0.99f;
            int   pune_count = 30;
            while (!maxHeap.empty() && pune_count > 0) {
                auto [prior, action] = maxHeap.top();
                maxHeap.pop();
                node->expand_node(action, prior, pune_count-1);
                pune_count --;
            }

        }
        else {
            if (isOverAndWinner == board->currentPlayer) {
                value = 1.0f;
            }
            else if (isOverAndWinner == -board->currentPlayer) {
                value = -1.0f;
            }
            else  // if (isOverAndWinner == board->DRAW || isOverAndWinner == board->FLAT)
            {
                value = 0.0f;
            }
        }
        // 回退
        while (!simulation_path.empty()) {
            auto [last_move, last_color] = simulation_path.back();
            evaluator.undo(last_color, last_move % BOARD_LEN, last_move / BOARD_LEN);
            board->undoAction(last_move);
            simulation_path.pop_back();
        }
        node->backup(value);
    }
    std::vector<int> visits;
    std::vector<int> actions;


    for (auto [act, child] : root->children) {
        if (child.second){
            visits.push_back(child.second->visits);
            actions.push_back(act);
        }
    }

    int s_v[361] = {0};
    for (int i = 0; i < actions.size(); i++) {
        s_v[actions[i]] = visits[i];
    }

    printBoard(
        std::cout,
        19,
        [&](auto &os, int x, int y) {
            int _count = s_v[y * 19 + x];
            os << std::setw(4) << _count;
        },
        4);

    int action    = 0;
    int max_visit = 0;
    for (int i = 0; i < actions.size(); i++) {
        if (visits[i] > max_visit) {
            max_visit = visits[i];
            action    = actions[i];
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto using_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - startTime).count();
    for (auto act : actions)
        std::cout << act << " ";
    std::cout << std::endl;
    std::cout << "[INFO TIME ]: " << using_ms << std::endl;
    std::cout << "[INFO R.COUNT]: " << root->visits << std::endl;
//    std::cout << "[INFO C.COUNT]: " << root->children[action]->visits << std::endl;
    std::cout << "[INFO NPS]: " << root->visits * 1000 / using_ms << std::endl;
    std::cout << "[INFO ACTIONS]: " << board->available_moves.size() << std::endl;
    std::cout << "[INFO PIECES]: " << board->numPieces << std::endl;

    std::cout << "[INFO R.WinRate]: " << (root->win_rate + 1.0f) * 50 << std::endl;
//    std::cout << "[INFO C.WinRate]: " << (root->children[action]->win_rate + 1.0f) * 50
//              << std::endl;
    std::cout << "[INFO Count]: " << num_count << std::endl;

    m_search_info.search_counts  = root->visits;
    m_search_info.search_time_ms = using_ms;
    m_search_info.num_pieces     = board->numPieces - 1;
    m_search_info.nps            = root->visits * 1000 / using_ms;
    m_search_info.win_rate       = (root->win_rate + 1.0f) * 50;
    ;

    node_count = root->visits;
    win_rate   = (root->win_rate + 1.0f) * 50;
    return action;
}