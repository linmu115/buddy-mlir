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
//

#ifndef BUDDY_PIPELINE_PROMPT_FRAME_PLAN_H
#define BUDDY_PIPELINE_PROMPT_FRAME_PLAN_H

#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/mpi/ActivationProtocol.h"
#include <algorithm>

namespace buddy::runtime::pipeline {
struct PromptFrame {
  ActivationHeader header;
  bool prefill;
  bool needLogits;
  int validRows;
};

// Indexable so RX can look ahead without advancing the compute cursor.
class PromptFramePlan {
public:
  PromptFramePlan(int prompt, int chunk, int context, int version,
                  bool appendOnly = false)
      : prompt(prompt), chunk(chunk), context(context), version(version),
        appendOnly(appendOnly) {
    if (prompt < 1 || prompt > context || chunk < 1 || chunk > context)
      throw std::invalid_argument("invalid chunked prompt dimensions");
  }
  int prefillTokens() const {
    return !appendOnly && prompt < chunk ? prompt - 1 : prompt;
  }
  int size() const {
    if (!appendOnly && prompt < chunk)
      return prompt == 1 ? 1 : 2;
    return 1 + (prompt - 1) / chunk;
  }
  PromptFrame operator[](int i) const {
    const bool final = i == size() - 1;
    if (!appendOnly && final && prompt < chunk)
      return {makeChunkDecodeHeader(i, prompt - 1, context, version), false,
              true, 1};
    const int pos = !appendOnly && final ? prompt - chunk : i * chunk;
    return {makeChunkPrefillHeader(i, pos, chunk, context, version), true,
            final, appendOnly ? std::min(chunk, prompt - pos) : chunk};
  }

private:
  int prompt, chunk, context, version;
  bool appendOnly;
};
inline const void *runPromptFrame(Stage &stage, void *input,
                                  const PromptFrame &frame, bool appendOnly) {
  if (!frame.prefill)
    return stage.decodeData(input, frame.header.pos);
  if (appendOnly)
    return stage.prefillValidRows(input, frame.header.pos, frame.needLogits,
                                  frame.validRows);
  return stage.prefillChunk(input, frame.header.pos, frame.needLogits);
}
} // namespace buddy::runtime::pipeline
#endif
