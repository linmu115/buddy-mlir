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

#ifndef BUDDY_DEEPSEEK_PIPELINE_STAGE_H
#define BUDDY_DEEPSEEK_PIPELINE_STAGE_H

#include <cstdint>

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
};

using CreateStage = Stage *(*)(const char *directory);

#endif
