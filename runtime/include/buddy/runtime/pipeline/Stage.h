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

#ifndef BUDDY_RUNTIME_PIPELINE_STAGE_H
#define BUDDY_RUNTIME_PIPELINE_STAGE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

enum class ActivationType { F16, F32 };

struct StageConfig {
  int startLayer;
  int endLayer;
  int totalLayers;
  int context;
  int activationWidth;
  int vocabSize;
  int eosId;
  ActivationType dtype;
  int prefillChunk = 0;
};

class Stage {
public:
  virtual ~Stage() = default;
  virtual const StageConfig &configuration() const = 0;
  // Input contains token IDs on the first stage and contiguous FP16 hidden
  // states elsewhere. Prefill uses context rows. Decode uses one row.
  // The result is valid until the next invocation.
  virtual const uint16_t *prefill(void *input) = 0;
  virtual const uint16_t *decode(void *input, int position) = 0;

  // Dtype-neutral and chunked-prefill entry points. Default implementations
  // preserve full source compatibility for existing FP16 Stage subclasses.
  virtual const void *prefillData(void *input) { return prefill(input); }
  virtual const void *decodeData(void *input, int position) {
    return decode(input, position);
  }
  // Chunked prefill (used when configuration().prefillChunk > 0): runs a
  // prefill chunk of `prefillChunk` rows starting at KV cache `position`.
  // On the last stage, `computeLogits` controls whether the final lm_head
  // projection runs for the chunk's last row (via buddy_set_prefill_logits).
  virtual const void *prefillChunk(void *input, int position,
                                   bool computeLogits) {
    (void)input;
    (void)position;
    (void)computeLogits;
    throw std::logic_error("prefillChunk is not supported by this stage");
  }
};

extern "C" {
Stage *createPipelineStage(const char *directory);
Stage *buddy_create_pipeline_stage_v1(const char *resolvedWeightPath);
Stage *buddy_create_pipeline_stage_v2(const char *const *resolvedWeightPaths,
                                      size_t weightCount);
void buddy_destroy_pipeline_stage_v1(Stage *stage);
}

using CreateStage = decltype(&createPipelineStage);
using CreateStageV1 = decltype(&buddy_create_pipeline_stage_v1);
using CreateStageV2 = decltype(&buddy_create_pipeline_stage_v2);
using DestroyStageV1 = decltype(&buddy_destroy_pipeline_stage_v1);

#endif
