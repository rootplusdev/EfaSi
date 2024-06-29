#pragma once

#include "evaluator.h"
#include "simdops.h"

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix9 {

using namespace Evaluation;

// constexpr uint32_t ArchHashBase    = 0x247e6c70;
constexpr int      ShapeNum        = 442503;
constexpr int      FeatureDim      = 128;
constexpr int      PolicyDim       = 64;
constexpr int      ValueDim        = 128;
constexpr int      FeatDWConvDim   = 64;
constexpr int      PolicyPWConvDim = 16;
constexpr int      NumHeadBucket   = 1;

struct alignas(simd::NativeAlignment) Mix9Weight
{
    // 1  mapping layer
    int16_t mapping[2][ShapeNum][FeatureDim];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
        int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
        int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)
                                             * (PolicyDim * 2)];
        int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];

        // 4  int Group MLP (layer 1,2)
        int8_t  value_corner_weight[ValueDim * FeatureDim];
        int32_t value_corner_bias[ValueDim];
        int8_t  value_edge_weight[ValueDim * FeatureDim];
        int32_t value_edge_bias[ValueDim];
        int8_t  value_center_weight[ValueDim * FeatureDim];
        int32_t value_center_bias[ValueDim];
        int8_t  value_quad_weight[ValueDim * ValueDim];
        int32_t value_quad_bias[ValueDim];

        // 5  int MLP (layer 1,2,3)
        int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
        int32_t value_l1_bias[ValueDim];
        int8_t  value_l2_weight[ValueDim * ValueDim];
        int32_t value_l2_bias[ValueDim];
        int8_t  value_l3_weight[4 * ValueDim];
        int32_t value_l3_bias[4];

        // 6  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

class Mix9Accumulator
{
public:
    struct alignas(simd::NativeAlignment) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];
    };

    Mix9Accumulator(int boardSize);
    ~Mix9Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix9Weight &w);
    /// Incremental update mix6 network state.
    void move(const Mix9Weight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix9Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix9Weight &w, PolicyBuffer &policyBuffer);

private:
    friend class Mix8Evaluator;
    struct ChangeNum
    {
        uint16_t inner, outer;
    };
    //=============================================================
    // Network states

    /// int feature sum of the full board
    ValueSumType *valueSumTable;          // [H*W+1, FeatureDim] (aligned)
    ChangeNum    *versionChangeNumTable;  // [H*W+1] (unaligned) num inner changes and outer changes
    uint16_t     *versionInnerIndexTable;  // [H*W+1, H*W] (unaligned)
    uint16_t     *versionOuterIndexTable;  // [H*W+1, (H+2)*(W+2)] (unaligned)
    /// Index table to convert line shape to map feature
    std::array<uint32_t, 4> *indexTable;  // [N_inner, 4] (unaligned)
    /// Sumed map feature of four directions
    std::array<int16_t, FeatureDim> *mapSum;  // [N_inner, FeatureDim] (aligned)
    /// Map feature after depth wise conv
    std::array<int16_t, FeatDWConvDim> *mapConv;  // [N_outer, DWConvDim] (aligned)

    //=============================================================
    int    boardSize;
    int    outerBoardSize;  // (boardSize + 2)
    int    currentVersion;
    int8_t groupIndex[32];

    void initIndexTable();
    int  getBucketIndex() { return 0; }
};

class Mix9Evaluator : public Evaluator
{
public:
    Mix9Evaluator(int                   boardSize,
                  std::filesystem::path blackWeightPath,
                  std::filesystem::path whiteWeightPath);
    ~Mix9Evaluator();

    void initEmptyBoard();
    void move(Color side, int x, int y) { addCache(side, x, y, false); }
    void undo(Color side, int x, int y) { addCache(side, x, y, true); }

    ValueType evaluateValue(Color side);
    void      evaluatePolicy(Color side, PolicyBuffer &policyBuffer);

private:
    struct MoveCache
    {
        Color  oldColor, newColor;
        int8_t x, y;

        friend bool isContraryMove(MoveCache a, MoveCache b)
        {
            bool isSameCoord = a.x == b.x && a.y == b.y;
            bool isContrary  = a.oldColor == b.newColor && a.newColor == b.oldColor;
            return isSameCoord && isContrary;
        }
    };

    /// Clear all caches to sync accumulator state with current board state.
    void clearCache(Color side);
    /// Record new board action, but not update accumulator instantly.
    void addCache(Color side, int x, int y, bool isUndo);

    Mix9Weight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Mix9Accumulator> accumulator[2];
    std::vector<MoveCache>           moveCache[2];
};

}  // namespace Evaluation::mix9
