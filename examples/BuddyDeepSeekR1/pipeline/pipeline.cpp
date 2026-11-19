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

#include "BlockingMPI.h"

#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/KVCacheManager.h"
#include "buddy/runtime/llm/Sampler.h"
#include "buddy/runtime/pipeline/Stage.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr int activationTag = 0;
constexpr int tokenTag = 1;

class StageLibrary {
public:
  explicit StageLibrary(const std::filesystem::path &directory) {
    const auto path = directory / "stage.so";
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error("cannot load " + path.string() + ": " +
                               dlerror());
  }
  ~StageLibrary() { dlclose(handle); }
  StageLibrary(const StageLibrary &) = delete;
  StageLibrary &operator=(const StageLibrary &) = delete;

  std::unique_ptr<Stage> create(const char *directory) const {
    dlerror();
    auto createStage =
        reinterpret_cast<CreateStage>(dlsym(handle, "createPipelineStage"));
    if (const char *error = dlerror())
      throw std::runtime_error(std::string("createPipelineStage: ") + error);
    if (!createStage)
      throw std::runtime_error("createPipelineStage symbol is null");
    std::unique_ptr<Stage> stage(createStage(directory));
    if (!stage)
      throw std::runtime_error("createPipelineStage returned null");
    return stage;
  }

private:
  void *handle;
};

void checkStages(const StageConfig &config, int ranks) {
  const std::array fields{
      config.startLayer,      config.endLayer,
      config.totalLayers,     config.context,
      config.activationWidth, config.vocabSize,
      config.eosId,           static_cast<int>(config.dtype)};
  std::vector<int> all(ranks * fields.size());
  mpiCheck(MPI_Allgather(fields.data(), fields.size(), MPI_INT, all.data(),
                         fields.size(), MPI_INT, MPI_COMM_WORLD));
  int nextLayer = 0;
  for (int rank = 0; rank < ranks; ++rank) {
    const int *stage = all.data() + rank * fields.size();
    if (stage[0] != nextLayer || stage[1] <= stage[0] ||
        !std::equal(fields.begin() + 2, fields.end(), stage + 2))
      throw std::runtime_error("ranks have incompatible stage configurations");
    nextLayer = stage[1];
  }
  if (nextLayer != config.totalLayers || config.dtype != ActivationType::F16)
    throw std::runtime_error("pipeline requires all layers in FP16");
}

void generate(int count, const std::string &prompt, const char *vocab,
              const std::string &logitsFile, Stage &stage, int rank,
              int ranks) {
  const auto &config = stage.configuration();
  checkStages(config, ranks);
  const bool first = rank == 0;
  const bool last = rank == ranks - 1;
  BlockingActivation activation(config, MPI_COMM_WORLD);

  int promptRows = 0;
  std::vector<int64_t> tokenInput;
  if (first) {
    if (prompt.empty() || prompt.front() == ' ')
      throw std::runtime_error("prompt must start with a non-space character");
    buddy::Text<size_t, 2> text(prompt);
    // Space is mapped to a two-byte character. Allow the chat markers too
    // so the tokenizer can report a too-long prompt before the model runs.
    text.tokenizeDeepSeekR1(
        vocab, std::max<size_t>(config.context, 2 * prompt.size() + 6));
    if (text.getTokenCnt() > static_cast<size_t>(config.context))
      throw std::runtime_error("tokenized prompt exceeds model context");
    promptRows = static_cast<int>(text.getTokenCnt());
    tokenInput.assign(text.getData(), text.getData() + config.context);
  }
  mpiCheck(MPI_Bcast(&promptRows, 1, MPI_INT, 0, MPI_COMM_WORLD));
  if (count <= 0 || count - 1 > config.context - promptRows)
    throw std::runtime_error("generation length exceeds remaining context");

  std::vector<uint16_t> hidden;
  if (!first)
    hidden.resize(static_cast<size_t>(config.context) * config.activationWidth);
  std::vector<float> logits;
  std::ofstream raw;
  if (last) {
    logits.resize(config.vocabSize);
    if (logitsFile != "-") {
      raw.open(logitsFile, std::ios::binary);
      if (!raw)
        throw std::runtime_error("cannot open " + logitsFile);
    }
  }
  buddy::Sampler sampler(buddy::SamplerConfig{});
  std::vector<int> generated;
  for (int step = 0; step < count; ++step) {
    const int rows = step == 0 ? config.context : 1;
    void *input = tokenInput.data();
    if (!first) {
      const int received = activation.receive(hidden.data(), config.context,
                                              rank - 1, activationTag);
      if (received != rows)
        throw std::runtime_error("activation rows differ from graph schedule");
      input = hidden.data();
    }
    const uint16_t *output = step == 0
                                 ? stage.prefill(input)
                                 : stage.decode(input, promptRows + step - 1);
    int token = 0;
    if (last) {
      if (step == 0)
        output += static_cast<size_t>(promptRows - 1) * config.vocabSize;
      if (raw.is_open() && !raw.write(reinterpret_cast<const char *>(output),
                                      logits.size() * sizeof(uint16_t)))
        throw std::runtime_error("error writing " + logitsFile);
      std::transform(output, output + logits.size(), logits.begin(),
                     buddy::kvcache::detail::halfToFloat);
      token = sampler.sample(logits.data(), logits.size(), generated);
      generated.push_back(token);
      if (!first)
        mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, tokenTag, MPI_COMM_WORLD));
    } else {
      activation.send(output, rows, rank + 1, activationTag);
      if (first)
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, ranks - 1, tokenTag,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    int stop = token == config.eosId;
    if (ranks > 2)
      mpiCheck(MPI_Bcast(&stop, 1, MPI_INT, ranks - 1, MPI_COMM_WORLD));
    if (stop)
      break;
    if (first)
      tokenInput.front() = token;
  }

  if (last) {
    if (raw.is_open() && !raw.flush())
      throw std::runtime_error("error flushing " + logitsFile);
    std::cout << "{\"prompt_tokens\":" << promptRows << ",\"token_ids\":[";
    for (size_t i = 0; i < generated.size(); ++i)
      std::cout << (i ? "," : "") << generated[i];
    std::cout << "]}\n";
  }
}
} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return 1;
  int rank = 0;
  try {
    int ranks = 0;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (argc != 5 + ranks)
      throw std::runtime_error("usage: pipeline VOCAB MAX_NEW_TOKENS PROMPT "
                               "LOGITS_FILE_OR_DASH STAGE_DIR...");
    {
      StageLibrary library(std::filesystem::absolute(argv[5 + rank]));
      auto stage = library.create(argv[5 + rank]);
      generate(std::stoi(argv[2]), argv[3], argv[1], argv[4], *stage, rank,
               ranks);
    }
  } catch (const std::exception &err) {
    std::cerr << "rank " << rank << ": " << err.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
