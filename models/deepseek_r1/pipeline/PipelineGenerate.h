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

#pragma once
#include "DistributedGenerate.h"
#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/pipeline/StageManifest.h"
namespace buddy::runtime {
inline void checkLocalConfig(const StageManifest &spec,
                             const StageConfig &config) {
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
  if (spec.activationType != libType || (config.dtype != ActivationType::F16 &&
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

inline int parseOptionalPositiveEnvInt(const char *name) {
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

inline std::string optionalEnvString(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0')
    return {};
  return std::string(raw);
}

inline pipeline::TokenizerKind resolveTokenizerKind(const std::string &name) {
  if (name.empty() || name == "deepseek_r1")
    return pipeline::TokenizerKind::DeepSeekR1;
  if (name == "qwen3")
    return pipeline::TokenizerKind::Qwen3;
  throw std::runtime_error("unsupported stage tokenizer '" + name +
                           "' (expected 'deepseek_r1' or 'qwen3')");
}

inline void appendUniqueStopToken(std::vector<int> &stopTokenIds, int tokenId) {
  if (std::find(stopTokenIds.begin(), stopTokenIds.end(), tokenId) ==
      stopTokenIds.end())
    stopTokenIds.push_back(tokenId);
}

inline void runDistributedStage(Stage &stage, const StageManifest &spec,
                                const std::string &modelName,
                                const std::string &vocabPath,
                                const RunConfig &cfg, MPI_Comm comm, int rank,
                                int size) {
  checkLocalConfig(spec, stage.configuration());

  pipeline::StageCheckOptions checkOpts;
  checkOpts.modelName = modelName;
  checkOpts.requireModelName = true;
  checkOpts.wireDtype = spec.wireDtype;
  checkOpts.transport = spec.transport;
  pipeline::checkAllStages(comm, rank, size, stage.configuration(), checkOpts);

  pipeline::DistributedGenerateOptions genOpts;
  genOpts.transport = spec.transport;
  genOpts.wireDtype = spec.wireDtype;
  if (rank == 0) {
    const auto wire =
        resolveWireSpec(stage.configuration().dtype, spec.wireDtype);
    std::cerr << "pipeline transport="
              << (spec.transport == pipeline::Transport::Async ? "async"
                                                               : "blocking")
              << " requested_wire="
              << (spec.wireDtype == pipeline::WireDtype::F16 ? "f16" : "native")
              << " effective_wire="
              << (size == 1 ? "none" : pipeline::activationTypeName(wire.dtype))
              << " version=" << wire.version << '\n';
  }
  genOpts.prompt = cfg.prompt;
  genOpts.vocabPath = vocabPath;
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

  if (rank == 0 && std::getenv("BUDDY_PP_OBSERVE"))
    std::fprintf(stderr, "pp_ready rank=0\n");
  pipeline::generateDistributed(genOpts, stage, comm, rank, size);
  if (std::getenv("BUDDY_PP_OBSERVE")) {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
      if (line.rfind("VmHWM:", 0) == 0)
        std::fprintf(stderr, "pp_memory rank=%d %s\n", rank, line.c_str());
  }
}
} // namespace buddy::runtime
