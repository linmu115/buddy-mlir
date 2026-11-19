//===----------------------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/pipeline/Stage.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace {
#include "stage-config.inc"

constexpr bool first = config.startLayer == 0;
constexpr bool last = config.endLayer == config.totalLayers;
constexpr size_t layers = config.endLayer - config.startLayer;
constexpr size_t kvElements = kvHeads * config.context * headDim;
constexpr size_t outputWidth = last ? config.vocabSize : config.activationWidth;

using PositionRef = StridedMemRefType<int64_t, 1>;

template <typename T, size_t Rank>
StridedMemRefType<T, Rank> descriptor(T *data,
                                      std::array<int64_t, Rank> shape) {
  StridedMemRefType<T, Rank> ref{};
  ref.basePtr = ref.data = data;
  int64_t stride = 1;
  for (size_t axis = Rank; axis > 0; --axis) {
    ref.sizes[axis - 1] = shape[axis - 1];
    ref.strides[axis - 1] = stride;
    stride *= shape[axis - 1];
  }
  return ref;
}

#ifdef BUDDY_STAGE_W4G32

constexpr StageConfig makeW4G32Config() {
  StageConfig cfg = config;
  cfg.prefillChunk = prefillChunk;
  return cfg;
}
constexpr StageConfig stageConfig = makeW4G32Config();

#ifndef BUDDY_STAGE_HAS_HUGEPAGES
constexpr bool useHugePages = false;
#endif

constexpr size_t prefillOutputRows = last ? 1 : prefillChunk;

using WeightF32Ref = StridedMemRefType<float, 1>;
using WeightI8Ref = StridedMemRefType<int8_t, 1>;
using KVRef = StridedMemRefType<float, 4>;
using OutputRef = StridedMemRefType<float, 3>;

using InputElement = std::conditional_t<first, int64_t, float>;
constexpr size_t inputRank = first ? 2 : 3;
using InputRef = StridedMemRefType<InputElement, inputRank>;

struct DecodeLayer {
  PositionRef position;
  std::array<KVRef, 2> kv;
};
struct ForwardResult {
  std::array<DecodeLayer, layers> layer;
  OutputRef output;
};
static_assert(sizeof(DecodeLayer) == sizeof(PositionRef) + 2 * sizeof(KVRef));
static_assert(offsetof(ForwardResult, output) == layers * sizeof(DecodeLayer));

extern "C" void _mlir_ciface_forward_prefill(ForwardResult *, WeightF32Ref *,
                                             WeightI8Ref *, InputRef *,
                                             BUDDY_STAGE_DECODE_ARGS);
extern "C" void _mlir_ciface_forward_decode(ForwardResult *, WeightF32Ref *,
                                            WeightI8Ref *, InputRef *,
                                            BUDDY_STAGE_DECODE_ARGS);
#undef BUDDY_STAGE_DECODE_ARGS

extern "C" void buddy_arena_reset(void) __attribute__((weak));
extern "C" void buddy_set_prefill_logits(int32_t) __attribute__((weak));

InputRef inputDescriptor(void *data, int rows) {
  auto *elements = static_cast<InputElement *>(data);
  std::array<int64_t, inputRank> shape{1, rows};
  if constexpr (!first)
    shape.back() = stageConfig.activationWidth;
  return descriptor(elements, shape);
}

template <size_t I>
auto *layerArgument(PositionRef &position, std::array<KVRef, 2 * layers> &kv) {
  if constexpr (I % 3 == 0)
    return &position;
  else
    return &kv[2 * (I / 3) + (I % 3) - 1];
}

template <size_t... I>
void callPrefill(ForwardResult &result, WeightF32Ref &f32W, WeightI8Ref &i8W,
                 InputRef &input, PositionRef &position,
                 std::array<KVRef, 2 * layers> &kv,
                 std::index_sequence<I...>) {
  _mlir_ciface_forward_prefill(&result, &f32W, &i8W, &input,
                               layerArgument<I>(position, kv)...);
}

template <size_t... I>
void callDecode(ForwardResult &result, WeightF32Ref &f32W, WeightI8Ref &i8W,
                InputRef &input, PositionRef &position,
                std::array<KVRef, 2 * layers> &kv,
                std::index_sequence<I...>) {
  _mlir_ciface_forward_decode(&result, &f32W, &i8W, &input,
                              layerArgument<I>(position, kv)...);
}

// When BuddyArena is enabled, MLIR temporary allocations live in the arena and
// are reclaimed by buddy_arena_reset() rather than std::free().
class ReturnedBuffers {
public:
  ~ReturnedBuffers() {
    if constexpr (!useArena) {
      for (auto *allocation : allocations)
        std::free(allocation);
    }
  }
  void add(void *allocation) {
    if (allocation && borrowed.count(allocation) == 0)
      allocations.insert(allocation);
  }
  std::unordered_set<void *> borrowed;

private:
  std::unordered_set<void *> allocations;
};

void adviseHugePages(void *p, size_t n) {
#ifdef MADV_HUGEPAGE
  const uintptr_t huge = uintptr_t(2) << 20;
  uintptr_t a = (reinterpret_cast<uintptr_t>(p) + huge - 1) & ~(huge - 1);
  uintptr_t e = (reinterpret_cast<uintptr_t>(p) + n) & ~(huge - 1);
  if (e > a)
    madvise(reinterpret_cast<void *>(a), e - a, MADV_HUGEPAGE);
#else
  (void)p;
  (void)n;
#endif
}

template <typename T>
std::vector<T> loadBlob(const std::filesystem::path &path, size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open weight file " + path.string());
  const std::streamsize expectedBytes =
      static_cast<std::streamsize>(count * sizeof(T));
  const std::streamoff actualBytes = file.tellg();
  if (actualBytes < 0)
    throw std::runtime_error("cannot determine weight size for " +
                             path.string());
  if (actualBytes != expectedBytes)
    throw std::runtime_error("weight byte count mismatch for " + path.string() +
                             ": expected " + std::to_string(expectedBytes) +
                             ", got " + std::to_string(actualBytes));
  std::vector<T> data(count);
  if constexpr (useHugePages)
    adviseHugePages(data.data(), count * sizeof(T));
  if (!file.seekg(0) ||
      !file.read(reinterpret_cast<char *>(data.data()), expectedBytes))
    throw std::runtime_error("cannot read weight file " + path.string());
  return data;
}

class DeepSeekW4G32Stage final : public Stage {
public:
  DeepSeekW4G32Stage(const std::filesystem::path &f32Path,
                     const std::filesystem::path &i8Path)
      : f32Weights(loadBlob<float>(f32Path, f32WeightElements)),
        i8Weights(loadBlob<int8_t>(i8Path, i8WeightElements)),
        kvStorage(2 * layers * kvElements, 0.0f),
        output(prefillOutputRows * outputWidth, 0.0f) {
    f32Ref = descriptor(
        f32Weights.data(),
        std::array<int64_t, 1>{static_cast<int64_t>(f32WeightElements)});
    i8Ref = descriptor(
        i8Weights.data(),
        std::array<int64_t, 1>{static_cast<int64_t>(i8WeightElements)});
    for (size_t i = 0; i < kv.size(); ++i)
      kv[i] = descriptor(
          kvStorage.data() + i * kvElements,
          std::array<int64_t, 4>{1, static_cast<int64_t>(kvHeads),
                                 stageConfig.context,
                                 static_cast<int64_t>(headDim)});
  }

  const StageConfig &configuration() const override { return stageConfig; }

  const uint16_t *prefill(void *) override {
    throw std::logic_error(
        "w4g32 stage uses f32 chunked prefill; call prefillChunk()");
  }

  const uint16_t *decode(void *, int) override {
    throw std::logic_error(
        "w4g32 stage uses f32 activations; call decodeData()");
  }

  const void *prefillData(void *input) override {
    return prefillChunk(input, 0, true);
  }

  const void *prefillChunk(void *input, int position,
                           bool computeLogits) override {
    if constexpr (useArena) {
      if (buddy_arena_reset)
        buddy_arena_reset();
    }
    if constexpr (last) {
      if (buddy_set_prefill_logits)
        buddy_set_prefill_logits(computeLogits ? 1 : 0);
    }
    auto data = inputDescriptor(input, ::prefillChunk);
    int64_t index = position;
    auto pos = descriptor(&index, std::array<int64_t, 1>{1});
    ForwardResult result{};
    callPrefill(result, f32Ref, i8Ref, data, pos, kv,
                std::make_index_sequence<3 * layers>{});
    if constexpr (last) {
      if (!computeLogits && buddy_set_prefill_logits)
        buddy_set_prefill_logits(1);
    }
    return finalizeResult(result, data, pos, prefillOutputRows,
                          !last || computeLogits);
  }

  const void *decodeData(void *input, int position) override {
    if constexpr (useArena) {
      if (buddy_arena_reset)
        buddy_arena_reset();
    }
    auto data = inputDescriptor(input, 1);
    int64_t index = position;
    auto pos = descriptor(&index, std::array<int64_t, 1>{1});
    ForwardResult result{};
    callDecode(result, f32Ref, i8Ref, data, pos, kv,
               std::make_index_sequence<3 * layers>{});
    return finalizeResult(result, data, pos, 1, /*copyOutput=*/true);
  }

private:
  const float *finalizeResult(const ForwardResult &result, const InputRef &data,
                              const PositionRef &pos, size_t rows,
                              bool copyOutput) {
    ReturnedBuffers returned;
    returned.borrowed = {data.basePtr, f32Ref.basePtr, i8Ref.basePtr,
                         pos.basePtr};
    for (const auto &ref : kv)
      returned.borrowed.insert(ref.basePtr);
    for (size_t i = 0; i < layers; ++i) {
      returned.add(result.layer[i].position.basePtr);
      updateKV(2 * i, result.layer[i].kv[0], returned);
      updateKV(2 * i + 1, result.layer[i].kv[1], returned);
    }
    if (copyOutput)
      std::copy_n(result.output.data + result.output.offset, rows * outputWidth,
                  output.data());
    returned.add(result.output.basePtr);
    return output.data();
  }

  void updateKV(size_t i, const KVRef &ref, ReturnedBuffers &returned) {
    auto *data = ref.data + ref.offset;
    if (data != kv[i].data)
      std::copy_n(data, kvElements, kv[i].data);
    returned.add(ref.basePtr);
  }

  std::vector<float> f32Weights;
  std::vector<int8_t> i8Weights;
  std::vector<float> kvStorage;
  std::vector<float> output;
  WeightF32Ref f32Ref{};
  WeightI8Ref i8Ref{};
  std::array<KVRef, 2 * layers> kv{};
};

#else // FP16 stage

using WeightRef = StridedMemRefType<uint16_t, 1>;
using KVRef = StridedMemRefType<uint16_t, 4>;
using OutputRef = StridedMemRefType<uint16_t, 3>;

using InputElement = std::conditional_t<first, int64_t, uint16_t>;
constexpr size_t inputRank = first ? 2 : 3;
using InputRef = StridedMemRefType<InputElement, inputRank>;

struct PrefillResult {
  std::array<KVRef, 2 * layers> kv;
  OutputRef output;
};
struct DecodeLayer {
  PositionRef position;
  std::array<KVRef, 2> kv;
};
struct DecodeResult {
  std::array<DecodeLayer, layers> layer;
  OutputRef output;
};
static_assert(sizeof(DecodeLayer) == sizeof(PositionRef) + 2 * sizeof(KVRef));
static_assert(offsetof(PrefillResult, output) == 2 * layers * sizeof(KVRef));
static_assert(offsetof(DecodeResult, output) == layers * sizeof(DecodeLayer));

extern "C" void _mlir_ciface_forward_prefill(PrefillResult *, WeightRef *,
                                             InputRef *);
extern "C" void _mlir_ciface_forward_decode(DecodeResult *, WeightRef *,
                                            InputRef *,
                                            BUDDY_STAGE_DECODE_ARGS);
#undef BUDDY_STAGE_DECODE_ARGS

InputRef inputDescriptor(void *data, int rows) {
  auto *elements = static_cast<InputElement *>(data);
  std::array<int64_t, inputRank> shape{1, rows};
  if constexpr (!first)
    shape.back() = config.activationWidth;
  return descriptor(elements, shape);
}

template <size_t I>
auto *decodeArgument(PositionRef &position, std::array<KVRef, 2 * layers> &kv) {
  if constexpr (I % 3 == 0)
    return &position;
  else
    return &kv[2 * (I / 3) + (I % 3) - 1];
}

template <size_t... I>
void callDecode(DecodeResult &result, WeightRef &weights, InputRef &input,
                PositionRef &position, std::array<KVRef, 2 * layers> &kv,
                std::index_sequence<I...>) {
  _mlir_ciface_forward_decode(&result, &weights, &input,
                              decodeArgument<I>(position, kv)...);
}

// Bufferization can return an input descriptor or a newly allocated memref.
// Keep input storage owned by the caller and free each new allocation once.
class ReturnedBuffers {
public:
  ~ReturnedBuffers() {
    for (auto *allocation : allocations)
      std::free(allocation);
  }
  void add(void *allocation) {
    if (allocation && borrowed.count(allocation) == 0)
      allocations.insert(allocation);
  }
  std::unordered_set<void *> borrowed;

private:
  std::unordered_set<void *> allocations;
};

std::vector<uint16_t> loadWeights(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open weight file " + path.string());

  constexpr std::streamsize expectedBytes = weightElements * sizeof(uint16_t);
  const std::streamoff actualBytes = file.tellg();
  if (actualBytes < 0)
    throw std::runtime_error("cannot determine weight size for " + path.string());
  if (actualBytes != expectedBytes)
    throw std::runtime_error("weight byte count mismatch for " + path.string() +
                             ": expected " + std::to_string(expectedBytes) +
                             ", got " + std::to_string(actualBytes));

  std::vector<uint16_t> weights(weightElements);
  if (!file.seekg(0) ||
      !file.read(reinterpret_cast<char *>(weights.data()), expectedBytes))
    throw std::runtime_error("cannot read weight file " + path.string());
  return weights;
}

class DeepSeekStage final : public Stage {
public:
  explicit DeepSeekStage(const std::filesystem::path &weightPath)
      : weights(loadWeights(weightPath)), kvStorage(2 * layers * kvElements),
        output(config.context * outputWidth) {
    weightRef =
        descriptor(weights.data(), std::array<int64_t, 1>{weightElements});
    for (size_t i = 0; i < kv.size(); ++i)
      kv[i] = descriptor(
          kvStorage.data() + i * kvElements,
          std::array<int64_t, 4>{1, kvHeads, config.context, headDim});
  }

  const StageConfig &configuration() const override { return config; }

  const uint16_t *prefill(void *input) override {
    auto data = inputDescriptor(input, config.context);
    PrefillResult result{};
    _mlir_ciface_forward_prefill(&result, &weightRef, &data);
    ReturnedBuffers returned;
    returned.borrowed = {data.basePtr, weightRef.basePtr};
    for (size_t i = 0; i < kv.size(); ++i)
      updateKV(i, result.kv[i], returned);
    return storeOutput(result.output, config.context, returned);
  }

  const uint16_t *decode(void *input, int position) override {
    auto data = inputDescriptor(input, 1);
    int64_t index = position;
    auto pos = descriptor(&index, std::array<int64_t, 1>{1});
    DecodeResult result{};
    callDecode(result, weightRef, data, pos, kv,
               std::make_index_sequence<3 * layers>{});
    ReturnedBuffers returned;
    returned.borrowed = {data.basePtr, weightRef.basePtr, pos.basePtr};
    for (const auto &ref : kv)
      returned.borrowed.insert(ref.basePtr);
    for (size_t i = 0; i < layers; ++i) {
      returned.add(result.layer[i].position.basePtr);
      updateKV(2 * i, result.layer[i].kv[0], returned);
      updateKV(2 * i + 1, result.layer[i].kv[1], returned);
    }
    return storeOutput(result.output, 1, returned);
  }

private:
  void updateKV(size_t i, const KVRef &ref, ReturnedBuffers &returned) {
    auto *data = ref.data + ref.offset;
    if (data != kv[i].data)
      std::copy_n(data, kvElements, kv[i].data);
    returned.add(ref.basePtr);
  }

  const uint16_t *storeOutput(const OutputRef &ref, int rows,
                              ReturnedBuffers &returned) {
    std::copy_n(ref.data + ref.offset, rows * outputWidth, output.data());
    returned.add(ref.basePtr);
    return output.data();
  }

  std::vector<uint16_t> weights;
  std::vector<uint16_t> kvStorage;
  std::vector<uint16_t> output;
  WeightRef weightRef{};
  std::array<KVRef, 2 * layers> kv{};
};

#endif // BUDDY_STAGE_W4G32
} // namespace

extern "C" Stage *
buddy_create_pipeline_stage_v1(const char *resolvedWeightPath) {
  if (!resolvedWeightPath || !*resolvedWeightPath)
    throw std::invalid_argument("resolvedWeightPath must be non-empty");
#ifdef BUDDY_STAGE_W4G32
  const std::filesystem::path p(resolvedWeightPath);
  const std::filesystem::path dir =
      std::filesystem::is_directory(p) ? p : p.parent_path();
  return new DeepSeekW4G32Stage(dir / f32WeightFile, dir / i8WeightFile);
#else
  return new DeepSeekStage(resolvedWeightPath);
#endif
}

extern "C" Stage *
buddy_create_pipeline_stage_v2(const char *const *resolvedWeightPaths,
                               size_t weightCount) {
#ifdef BUDDY_STAGE_W4G32
  if (!resolvedWeightPaths || weightCount != 2)
    throw std::invalid_argument(
        "w4g32 stage requires exactly 2 resolved weight paths");
  if (!resolvedWeightPaths[0] || !*resolvedWeightPaths[0] ||
      !resolvedWeightPaths[1] || !*resolvedWeightPaths[1])
    throw std::invalid_argument("w4g32 weight paths must be non-empty");
  return new DeepSeekW4G32Stage(resolvedWeightPaths[0], resolvedWeightPaths[1]);
#else
  if (!resolvedWeightPaths || weightCount != 1)
    throw std::invalid_argument(
        "FP16 stage requires exactly 1 resolved weight path");
  return buddy_create_pipeline_stage_v1(resolvedWeightPaths[0]);
#endif
}

extern "C" void buddy_destroy_pipeline_stage_v1(Stage *stage) { delete stage; }

extern "C" Stage *createPipelineStage(const char *directory) {
  if (!directory || !*directory)
    throw std::invalid_argument("stage directory must be non-empty");
#ifdef BUDDY_STAGE_W4G32
  const std::filesystem::path dir(directory);
  return new DeepSeekW4G32Stage(dir / f32WeightFile, dir / i8WeightFile);
#else
  const auto path = std::filesystem::path(directory) / weightFile;
  return buddy_create_pipeline_stage_v1(path.c_str());
#endif
}
