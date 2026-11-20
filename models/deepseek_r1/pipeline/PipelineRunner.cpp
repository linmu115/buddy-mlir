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

#include "PipelineRunner.h"
#include "DistributedGenerate.h"

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/StageManifest.h"
#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <dlfcn.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace buddy::runtime {
namespace {

void requireActiveMPI() {
  int initialized = 0;
  int finalized = 0;
  mpiCheck(MPI_Initialized(&initialized));
  if (!initialized)
    throw std::runtime_error(
        "PipelineRunner requires an active MPI session (MPI_Init not called)");
  mpiCheck(MPI_Finalized(&finalized));
  if (finalized)
    throw std::runtime_error("PipelineRunner requires an active MPI session "
                             "(MPI_Finalize already called)");
}

void requireExternalWeights(const ModelManifest &m, const StageManifest &spec) {
  if (m.soPath.empty())
    throw std::runtime_error("missing HostSharedLib code object (soPath)");
  if (spec.activationType == "f16") {
    if (m.weightPaths.size() != 1)
      throw std::runtime_error(
          "field constants (external weights) must contain exactly 1 weight "
          "for f16 stage (got " +
          std::to_string(m.weightPaths.size()) + ")");
  } else {
    if (m.weightPaths.empty() || m.weightPaths.size() > 2)
      throw std::runtime_error(
          "field constants (external weights) must contain 1 or 2 weights "
          "for f32 stage (got " +
          std::to_string(m.weightPaths.size()) + ")");
  }
  for (size_t i = 0; i < m.weightPaths.size(); ++i) {
    if (m.weightPaths[i].empty())
      throw std::runtime_error("field constants[" + std::to_string(i) +
                               "] (external weight path) must be non-empty");
  }
}

struct StageDeleter {
  DestroyStageV1 destroy = nullptr;
  void operator()(Stage *stage) const {
    if (stage && destroy)
      destroy(stage);
  }
};

using StagePtr = std::unique_ptr<Stage, StageDeleter>;

class StageLibrary {
public:
  explicit StageLibrary(const std::string &soPath) {
    if (soPath.empty())
      throw std::runtime_error("field soPath must be non-empty");
    dlerror();
    handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char *err = dlerror();
      throw std::runtime_error("cannot load stage library " + soPath + ": " +
                               (err ? err : "unknown error"));
    }
  }

  ~StageLibrary() {
    if (handle)
      dlclose(handle);
  }

  StageLibrary(const StageLibrary &) = delete;
  StageLibrary &operator=(const StageLibrary &) = delete;

  StagePtr create(const std::vector<std::string> &weightPaths) const {
    dlerror();
    auto *destroySym = dlsym(handle, "buddy_destroy_pipeline_stage_v1");
    if (const char *err = dlerror())
      throw std::runtime_error(
          std::string("missing buddy_destroy_pipeline_stage_v1: ") + err);
    if (!destroySym)
      throw std::runtime_error(
          "buddy_destroy_pipeline_stage_v1 symbol is null");
    auto destroyFn = reinterpret_cast<DestroyStageV1>(destroySym);

    if (weightPaths.size() > 1) {
      dlerror();
      auto *createV2Sym = dlsym(handle, "buddy_create_pipeline_stage_v2");
      if (!dlerror() && createV2Sym) {
        std::vector<const char *> ptrs;
        ptrs.reserve(weightPaths.size());
        for (const auto &p : weightPaths)
          ptrs.push_back(p.c_str());
        auto createV2Fn = reinterpret_cast<CreateStageV2>(createV2Sym);
        Stage *raw = createV2Fn(ptrs.data(), ptrs.size());
        if (!raw)
          throw std::runtime_error(
              "buddy_create_pipeline_stage_v2 returned null");
        return StagePtr(raw, StageDeleter{destroyFn});
      }
    }

    dlerror();
    auto *createSym = dlsym(handle, "buddy_create_pipeline_stage_v1");
    if (const char *err = dlerror())
      throw std::runtime_error(
          std::string("missing buddy_create_pipeline_stage_v1: ") + err);
    if (!createSym)
      throw std::runtime_error("buddy_create_pipeline_stage_v1 symbol is null");

    auto createFn = reinterpret_cast<CreateStageV1>(createSym);
    Stage *raw = createFn(weightPaths.at(0).c_str());
    if (!raw)
      throw std::runtime_error("buddy_create_pipeline_stage_v1 returned null");
    return StagePtr(raw, StageDeleter{destroyFn});
  }

private:
  void *handle = nullptr;
};

void checkLocalConfig(const StageManifest &spec, const StageConfig &config) {
  if (spec.startLayer != config.startLayer)
    throw std::runtime_error(
        "local config mismatch on field pp_start_layer: manifest has " +
        std::to_string(spec.startLayer) + ", stage library reports " +
        std::to_string(config.startLayer));
  if (spec.endLayer != config.endLayer)
    throw std::runtime_error(
        "local config mismatch on field pp_end_layer: manifest has " +
        std::to_string(spec.endLayer) + ", stage library reports " +
        std::to_string(config.endLayer));
  if (spec.totalLayers != config.totalLayers)
    throw std::runtime_error(
        "local config mismatch on field pp_total_layers: manifest has " +
        std::to_string(spec.totalLayers) + ", stage library reports " +
        std::to_string(config.totalLayers));
  if (spec.activationWidth != config.activationWidth)
    throw std::runtime_error(
        "local config mismatch on field pp_activation_width: manifest has " +
        std::to_string(spec.activationWidth) + ", stage library reports " +
        std::to_string(config.activationWidth));

  const std::string libType = pipeline::activationTypeName(config.dtype);
  if (spec.activationType != libType ||
      (config.dtype != ActivationType::F16 &&
       config.dtype != ActivationType::F32))
    throw std::runtime_error(
        "local config mismatch on field pp_activation_type: manifest has '" +
        spec.activationType + "', stage library reports '" + libType + "'");

  if (config.context <= 0)
    throw std::runtime_error("stage library reports non-positive field context "
                             "(" +
                             std::to_string(config.context) + ")");
  if (config.vocabSize <= 0)
    throw std::runtime_error(
        "stage library reports non-positive field vocabSize (" +
        std::to_string(config.vocabSize) + ")");
  if (config.prefillChunk < 0 || config.prefillChunk > config.context)
    throw std::runtime_error(
        "stage library reports out-of-range field prefillChunk (" +
        std::to_string(config.prefillChunk) + ")");
  if (spec.prefillChunk > 0 && spec.prefillChunk != config.prefillChunk)
    throw std::runtime_error(
        "local config mismatch on field pp_prefill_chunk: manifest has " +
        std::to_string(spec.prefillChunk) + ", stage library reports " +
        std::to_string(config.prefillChunk));
}

void validateRunConfig(const RunConfig &cfg) {
  if (cfg.raxPath.empty())
    throw std::runtime_error("field raxPath must be non-empty");
  if (cfg.batchSize != 0 && cfg.batchSize != 1)
    throw std::runtime_error("field batchSize must be 0 or 1 in pipeline mode "
                             "(got " +
                             std::to_string(cfg.batchSize) + ")");
  if (cfg.interactive)
    throw std::runtime_error("field interactive is not supported in pipeline "
                             "mode");
  if (cfg.promptLength != 0)
    throw std::runtime_error("field promptLength is not supported in pipeline "
                             "mode");
  if (!cfg.prompts.empty())
    throw std::runtime_error("field prompts (batch prompt-file) is not "
                             "supported in pipeline mode");
  if (!cfg.perplexityIdsPath.empty())
    throw std::runtime_error("field perplexityIdsPath is not supported in "
                             "pipeline mode");
  if (!cfg.chatTemplatePath.empty())
    throw std::runtime_error("field chatTemplatePath is not supported in "
                             "pipeline mode");
  if (cfg.maxNewTokens < 0)
    throw std::runtime_error("field maxNewTokens must be non-negative (got " +
                             std::to_string(cfg.maxNewTokens) + ")");
}

int parseOptionalPositiveEnvInt(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return 0;
  std::string value(raw);
  int parsed = 0;
  const char *end = value.data() + value.size();
  const auto res = std::from_chars(value.data(), end, parsed);
  if (res.ec != std::errc{} || res.ptr != end || parsed <= 0)
    throw std::runtime_error(std::string("invalid test hook ") + name + "='" +
                             value + "' (expected positive integer)");
  return parsed;
}

std::string optionalEnvString(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return {};
  return std::string(raw);
}

pipeline::TokenizerKind resolveTokenizerKind(const std::string &name) {
  if (name.empty() || name == "deepseek_r1")
    return pipeline::TokenizerKind::DeepSeekR1;
  if (name == "qwen3")
    return pipeline::TokenizerKind::Qwen3;
  throw std::runtime_error("unsupported stage tokenizer '" + name +
                           "' (expected 'deepseek_r1' or 'qwen3')");
}

void appendUniqueStopToken(std::vector<int> &stopTokenIds, int tokenId) {
  if (std::find(stopTokenIds.begin(), stopTokenIds.end(), tokenId) ==
      stopTokenIds.end())
    stopTokenIds.push_back(tokenId);
}

[[noreturn]] void abortPipelineJob(MPI_Comm comm, int rank,
                                   const std::string &raxPath,
                                   const std::string &modelName,
                                   const std::string &detail) {
  std::ostringstream os;
  os << "rank " << rank << " [package="
     << (raxPath.empty() ? "<unset>" : raxPath);
  if (!modelName.empty())
    os << ", model=" << modelName;
  os << "]: " << detail << '\n';
  const std::string msg = os.str();
  std::cerr.write(msg.data(), static_cast<std::streamsize>(msg.size()));
  std::cerr.flush();
  // Give the MPI process manager a brief window to drain redirected stderr
  // pipes before MPI_Abort tears down the job.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  MPI_Abort(comm, 1);
  std::_Exit(1);
}

} // namespace

void PipelineRunner::run(const RunConfig &cfg) {
  requireActiveMPI();
  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  int size = 0;
  mpiCheck(MPI_Comm_rank(comm, &rank));
  mpiCheck(MPI_Comm_size(comm, &size));

  std::string modelName;
  try {
    validateRunConfig(cfg);
    auto m = ModelManifest::loadFromRax(cfg.raxPath);
    modelName = m.modelName;

    auto spec = parseStageV1(m.moduleAttrs);
    requireExternalWeights(m, spec);

    StageLibrary lib(m.soPath);
    auto stage = lib.create(m.weightPaths);
    checkLocalConfig(spec, stage->configuration());

    pipeline::StageCheckOptions checkOpts;
    checkOpts.modelName = m.modelName;
    checkOpts.requireModelName = true;
    pipeline::checkAllStages(comm, rank, size, stage->configuration(),
                             checkOpts);

    pipeline::DistributedGenerateOptions genOpts;
    genOpts.prompt = cfg.prompt;
    genOpts.vocabPath = m.vocabPath;
    genOpts.tokenizer = resolveTokenizerKind(spec.tokenizer);
    genOpts.stopTokenIds = spec.stopTokenIds;
    if (genOpts.tokenizer == pipeline::TokenizerKind::Qwen3) {
      // Match whole-model Qwen3VLMoeRunner stop tokens: <|im_end|> and
      // <|endoftext|>.
      appendUniqueStopToken(genOpts.stopTokenIds, 151645);
      appendUniqueStopToken(genOpts.stopTokenIds, 151643);
    }
    genOpts.tokenLimitMode = pipeline::TokenLimitMode::TotalSequenceLength;
    genOpts.maxTokens = cfg.maxNewTokens;
    genOpts.allowZeroTokenLimit = true;
    genOpts.samplerConfig = cfg.samplerConfig;
    genOpts.streamJsonl = cfg.streamJsonl;
    // Test-only hooks (do not change public RunConfig).
    genOpts.logitsDumpPath = optionalEnvString("BUDDY_PP_DUMP_LOGITS");
    genOpts.testEosAtStep = parseOptionalPositiveEnvInt("BUDDY_PP_TEST_EOS_STEP");
    genOpts.traceStagePrefix = optionalEnvString("BUDDY_PP_TRACE_STAGE");

    pipeline::generateDistributed(genOpts, *stage, comm, rank, size);
  } catch (const std::exception &err) {
    abortPipelineJob(comm, rank, cfg.raxPath, modelName, err.what());
  } catch (...) {
    abortPipelineJob(comm, rank, cfg.raxPath, modelName, "unknown exception");
  }
}

} // namespace buddy::runtime
