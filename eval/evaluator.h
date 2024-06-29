#pragma once

#include <algorithm>
#include <cmath>

namespace Evaluation {

inline constexpr float ScalingFactor = 200.0f;

enum Color { BLACK, WHITE, WALL, EMPTY, COLOR_NB, SIDE_NB = 2 };

enum Value {
    VALUE_ZERO     = 0,
    VALUE_DRAW     = 0,
    VALUE_MATE     = 30000,
    VALUE_INFINITE = 30001,
    VALUE_NONE     = -30002,

    VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - 500,
    VALUE_MATED_IN_MAX_PLY = -VALUE_MATE + 500,

    VALUE_EVAL_MAX = 20000,
    VALUE_EVAL_MIN = -20000
};

constexpr Value operator+(Value d1, Value d2)
{
    return Value(int(d1) + int(d2));
}
constexpr Value operator-(Value d1, Value d2)
{
    return Value(int(d1) - int(d2));
}
constexpr Value operator-(Value d)
{
    return Value(-int(d));
}
inline Value &operator+=(Value &d1, Value d2)
{
    return d1 = d1 + d2;
}
inline Value &operator-=(Value &d1, Value d2)
{
    return d1 = d1 - d2;
}
constexpr Value operator*(int i, Value d)
{
    return Value(i * int(d));
}
constexpr Value operator*(Value d, int i)
{
    return Value(int(d) * i);
}
constexpr Value operator/(Value d, int i)
{
    return Value(int(d) / i);
}
constexpr int operator/(Value d1, Value d2)
{
    return int(d1) / int(d2);
}
inline Value &operator*=(Value &d, int i)
{
    return d = Value(int(d) * i);
}
inline Value &operator/=(Value &d, int i)
{
    return d = Value(int(d) / i);
}
constexpr Value operator+(Value v, int i)
{
    return Value(int(v) + i);
}
constexpr Value operator-(Value v, int i)
{
    return Value(int(v) - i);
}
inline Value &operator+=(Value &v, int i)
{
    return v = v + i;
}
inline Value &operator-=(Value &v, int i)
{
    return v = v - i;
}

/// Construct a value for mate in N ply
constexpr Value mate_in(int ply)
{
    return Value(int(VALUE_MATE) - ply);
}

/// Construct a value for being mated in N ply
constexpr Value mated_in(int ply)
{
    return Value(int(-VALUE_MATE) + ply);
}

/// Get number of steps to mate from value and current ply
constexpr int mate_step(Value v, int ply)
{
    return VALUE_MATE - ply - (v < 0 ? -v : v);
}

constexpr int searchValueToStoredValue(Value value, int searchPly)
{
    return value == VALUE_NONE              ? VALUE_NONE
           : value > VALUE_MATE_IN_MAX_PLY  ? value + searchPly
           : value < VALUE_MATED_IN_MAX_PLY ? value - searchPly
                                            : value;
}

constexpr Value storedValueToSearchValue(int storedValue, int searchPly)
{
    return storedValue == VALUE_NONE              ? VALUE_NONE
           : storedValue > VALUE_MATE_IN_MAX_PLY  ? Value(storedValue - searchPly)
           : storedValue < VALUE_MATED_IN_MAX_PLY ? Value(storedValue + searchPly)
                                                  : Value(storedValue);
}

/// Converts a evaluation value to winning rate (in [0, 1]) using ScalingFactor.
template <bool Strict = true>
inline float valueToWinRate(Value eval)
{
    if (eval >= (Strict ? VALUE_MATE_IN_MAX_PLY : VALUE_EVAL_MAX))
        return 1.0;
    if (eval <= (Strict ? VALUE_MATED_IN_MAX_PLY : VALUE_EVAL_MIN))
        return 0.0;
    return 1.0 / (1.0 + ::expf(-float(eval) / ScalingFactor));
}

/// Converts a winning rate in [0, 1] to a evaluation value using ScalingFactor.
inline Value winRateToValue(float winrate)
{
    if (winrate > 0.999995f)
        return VALUE_EVAL_MAX;
    else if (winrate < 0.000005f)
        return VALUE_EVAL_MIN;
    else {
        Value v = Value(ScalingFactor * ::logf(winrate / (1.0f - winrate)));
        return std::clamp(v, VALUE_EVAL_MIN, VALUE_EVAL_MAX);
    }
}

/// ValueType is a container for value (and a optional draw rate).
/// Draw rate value less than 0.0 means no draw rate is contained.
class ValueType
{
public:
    explicit ValueType(int value) : val(value) {}
    explicit ValueType(float winLogits,
                       float lossLogits,
                       float drawLogits,
                       bool  applySoftmax = true)
        : winProb(winLogits)
        , lossProb(lossLogits)
        , drawProb(drawLogits)
    {
        if (applySoftmax) {
            float maxValue = std::max(std::max(winLogits, lossLogits), drawLogits);
            winProb        = std::exp(winLogits - maxValue);
            lossProb       = std::exp(lossLogits - maxValue);
            drawProb       = std::exp(drawLogits - maxValue);
            float invSum   = 1.0f / (winProb + lossProb + drawProb);
            winProb *= invSum;
            lossProb *= invSum;
            drawProb *= invSum;
        }

        val = winRateToValue(winningRate());
    }
    bool  hasWinLossRate() const { return winProb >= 0.0f && lossProb >= 0.0f; }
    bool  hasDrawRate() const { return drawProb >= 0.0f; }
    float win() const { return winProb; }
    float loss() const { return lossProb; }
    float draw() const { return drawProb; }
    float winLossRate() const { return winProb - lossProb; }
    float winningRate() const { return (winLossRate() + 1) * 0.5f; }
    int   value() const { return val; }

private:
    int   val      = VALUE_NONE;
    float winProb  = -1.0f;
    float lossProb = -1.0f;
    float drawProb = -1.0f;
};

/// PolicyBuffer is a container for float policy values on board.
class PolicyBuffer
{
public:
    static constexpr int MAX_MOVES = 484;

    PolicyBuffer(int boardSize) : PolicyBuffer(boardSize, boardSize) {}
    PolicyBuffer(int boardWidth, int boardHeight)
        : boardWidth(boardWidth)
        , bufferSize(boardWidth * boardHeight)
    {
        std::fill_n(computeFlag, bufferSize, false);
    }

    float &operator()(int index) { return policy[index]; }
    float &operator()(int x, int y) { return policy[boardWidth * y + x]; }
    float  operator()(int x, int y) const { return policy[boardWidth * y + x]; }
    void   setComputeFlag(int index, bool enabled = true) { computeFlag[index] = enabled; }
    void   setComputeFlag(int x, int y, bool enabled = true)
    {
        computeFlag[boardWidth * y + x] = enabled;
    }
    bool getComputeFlag(int index) const { return computeFlag[index]; }
    bool getComputeFlag(int x, int y) const { return computeFlag[boardWidth * y + x]; }

    /// Applies softmax to all computed policy.
    void applySoftmax()
    {
        // Find max computed policy
        float maxPolicy = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < bufferSize; i++) {
            if (computeFlag[i] && policy[i] > maxPolicy)
                maxPolicy = policy[i];
        }

        // Apply exponent function and sum
        float sumPolicy = 0;
        for (size_t i = 0; i < bufferSize; i++) {
            if (computeFlag[i])
                sumPolicy += policy[i] = std::exp(policy[i] - maxPolicy);
            else
                policy[i] = 0;
        }

        // Divide sum policy to normalize
        float invSumPolicy = 1 / sumPolicy;
        for (size_t i = 0; i < bufferSize; i++) {
            policy[i] *= invSumPolicy;
        }
    }

private:
    int   boardWidth;
    int   bufferSize;
    bool  computeFlag[MAX_MOVES];
    float policy[MAX_MOVES];
};

/// Evaluator is the base class for evaluation plugins.
/// It provides overridable hook over board move/undo update, and interface for doing value
/// evaluation and policy evaluation. Different evaluation implementation may inherit from
/// this class to replace the default classical evaluation builtin the board.
class Evaluator
{
public:
    /// Constructor sets the board size and rule of the evaluator.
    /// Default behaviour supports all size and rule. If the evaluator does
    /// not support this rule or this board size, it throws an exception.
    Evaluator(int boardSize) : boardSize(boardSize) {}
    virtual ~Evaluator() = default;

    /// Resets the evaluator state to empty board.
    virtual void initEmptyBoard() = 0;
    /// Update hook called when put a new stone on board.
    virtual void move(Color side, int x, int y) {};
    /// Update hook called when rollback a stone on board.
    virtual void undo(Color side, int x, int y) {};

    /// Evaluates value for current side to move.
    virtual ValueType evaluateValue(Color side) = 0;
    /// Evaluates policy for current side to move.
    virtual void evaluatePolicy(Color side, PolicyBuffer &policyBuffer) = 0;

    const int boardSize;
};

}  // namespace Evaluation