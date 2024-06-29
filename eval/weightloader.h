#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <istream>
#include <mutex>

namespace Evaluation {

/// Base class for a weight loader.
/// @tparam WeightType The type of evaluator weight.
template <typename WeightType_>
struct WeightLoader
{
    typedef WeightType_ WeightType;

    /// Load and construct a weight type from the given input stream.
    /// @return Weight pointer if load succeeded, otherwise nullptr.
    virtual std::unique_ptr<WeightType> load(std::istream &is) = 0;

    /// Whether this weight loader needs a binary stream.
    /// Default behaviour is true.
    virtual bool needsBinaryStream() const { return true; }
};

/// Weight loader for binary Plain Old Data.
template <typename WeightType>
struct BinaryPODWeightLoader : WeightLoader<WeightType>
{
    std::unique_ptr<WeightType> load(std::istream &is) override
    {
        auto weight = std::make_unique<WeightType>();
        is.read(reinterpret_cast<char *>(weight.get()), sizeof(WeightType));
        if (is && is.peek() == std::ios::traits_type::eof())
            return std::move(weight);
        else
            return nullptr;
    }
};

/// @brief WeightRegistry is the global manager for loaded weights.
/// Usually each evaluator loads weight from file on its own, however in most case all
/// evaluator loads the same weight and it is very memory comsuming to have multiple
/// weight instance in memory. Weight Registry helps to reuse loaded weight when it is
/// applicable, by holding a pool of all loaded weight.
template <typename WeightType>
class WeightRegistry
{
public:
    using Loader = WeightLoader<WeightType>;

    /// Loads weight from the given file path, using the loader.
    /// If the weight already exists in registry, it reuse the loaded weight.
    /// @return Weight pointer, or nullptr if load failed.
    WeightType *loadWeightFromFile(std::filesystem::path filepath, Loader &loader);

    /// Unloads a loaded weight.
    void unloadWeight(WeightType *weight);

private:
    struct LoadedWeight
    {
        std::filesystem::path       filepath;
        std::unique_ptr<WeightType> weight;
        size_t                      refCount;
    };

    std::vector<LoadedWeight> pool;
    std::mutex                poolMutex;
};

}  // namespace Evaluation

template <typename WeightType>
inline WeightType *
Evaluation::WeightRegistry<WeightType>::loadWeightFromFile(std::filesystem::path filepath,
                                                           Loader               &loader)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    // Find weights in loaded weight pool
    for (auto &w : pool) {
        if (w.filepath == filepath) {
            w.refCount++;
            return w.weight.get();
        }
    }

    // If not found, load from file
    std::ios_base::openmode mode = std::ios::in;
    if (loader.needsBinaryStream())
        mode = mode | std::ios::binary;
    std::ifstream fileStream(filepath, mode);

    if (!fileStream.is_open())
        return nullptr;

    // Load weight using weight loader
    auto weight = loader.load(fileStream);

    // If load succeeded, add to pool
    if (weight) {
        pool.push_back({filepath, std::move(weight), 1});
        return pool.back().weight.get();
    }
    else
        return nullptr;
}

template <typename WeightType>
inline void Evaluation::WeightRegistry<WeightType>::unloadWeight(WeightType *weight)
{
    std::lock_guard<std::mutex> lock(poolMutex);

    for (size_t i = 0; i < pool.size(); i++) {
        if (pool[i].weight.get() == weight) {
            if (--pool[i].refCount == 0)
                pool.erase(pool.begin() + i);
            return;
        }
    }
}
