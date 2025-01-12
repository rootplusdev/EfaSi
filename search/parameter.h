#pragma once

#include <cstdint>

namespace Search::MCTS {

constexpr float MaxNewVisitsProp = 0.275f;

constexpr float CpuctExploration     = 0.40f;
constexpr float CpuctExplorationLog  = 0.75f;
constexpr float CpuctExplorationBase = 336;
constexpr float CpuctParentVisitBias = 0.1f;

constexpr float CpuctUtilityStdevScale     = 0.035f;
constexpr float CpuctUtilityVarPrior       = 0.18f;
constexpr float CpuctUtilityVarPriorWeight = 2.14f;

constexpr float FpuReductionMax     = 0.075f;
constexpr float FpuLossProp         = 0.001f;
constexpr float RootFpuReductionMax = 0.075f;
constexpr float RootFpuLossProp     = 0.0036f;
constexpr float FpuUtilityBlendPow  = 1.73f;

constexpr uint32_t MinTranspositionSkipVisits = 12;

constexpr bool  UseLCBForBestmoveSelection = true;
constexpr float LCBStdevs                  = 5.0f;
constexpr float LCBMinVisitProp            = 0.12f;

constexpr float PolicyTemperature = 1.0f;

}  // namespace Search::MCTS
