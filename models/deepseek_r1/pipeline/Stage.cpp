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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
using WeightRef = StridedMemRefType<uint16_t, 1>;
using PositionRef = StridedMemRefType<int64_t, 1>;
using KVRef = StridedMemRefType<uint16_t, 4>;
using OutputRef = StridedMemRefType<uint16_t, 3>;
#include "stage-config.inc"

constexpr bool first = config.startLayer == 0;
constexpr bool last = config.endLayer == config.totalLayers;
constexpr size_t layers = config.endLayer - config.startLayer;
constexpr size_t kvElements = kvHeads * config.context * headDim;
constexpr size_t outputWidth = last ? config.vocabSize : config.activationWidth;

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
} // namespace

extern "C" Stage *
buddy_create_pipeline_stage_v1(const char *resolvedWeightPath) {
  if (!resolvedWeightPath || !*resolvedWeightPath)
    throw std::invalid_argument("resolvedWeightPath must be non-empty");
  return new DeepSeekStage(resolvedWeightPath);
}

extern "C" void buddy_destroy_pipeline_stage_v1(Stage *stage) { delete stage; }

extern "C" Stage *createPipelineStage(const char *directory) {
  if (!directory || !*directory)
    throw std::invalid_argument("stage directory must be non-empty");
  const auto path = std::filesystem::path(directory) / weightFile;
  return buddy_create_pipeline_stage_v1(path.c_str());
}
