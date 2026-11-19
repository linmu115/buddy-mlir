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
// Shared pipeline-parallel stage validation and distributed generation loop
// used by both PipelineRunner (RAX runner plugin) and the standalone MPI
// example (examples/BuddyDeepSeekR1/pipeline/pipeline.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MODELS_DEEPSEEK_R1_PIPELINE_DISTRIBUTED_GENERATE_H
#define BUDDY_MODELS_DEEPSEEK_R1_PIPELINE_DISTRIBUTED_GENERATE_H

#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/KVCacheManager.h"
#include "buddy/runtime/llm/Sampler.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy::runtime::pipeline {

inline const char *activationTypeName(ActivationType dtype) {
  switch (dtype) {
  case ActivationType::F16:
    return "f16";
  case ActivationType::F32:
    return "f32";
  }
  return "unknown";
}

struct StageCheckOptions {
  std::string modelName;
  bool requireModelName = false;
};

struct DistributedGenerateOptions {
  std::string prompt;
  std::string vocabPath;
  /// 0 maps to the context-allowed limit when allowZeroMaxNewTokens is true.
  int maxNewTokens = 0;
  bool allowZeroMaxNewTokens = true;
  buddy::SamplerConfig samplerConfig{};
  /// Optional output file for sampled-row FP16 logits (empty or "-" disables).
  std::string logitsDumpPath;
  bool streamJsonl = false;
  /// Test-only hook: 1-based step index whose sampled token is treated as EOS.
  /// 0 disables the hook.
  int testEosAtStep = 0;
  /// Test-only hook: when non-empty, appends stage prefill/decode calls to
  /// "<traceStagePrefix>.rank<rank>.log".
  std::string traceStagePrefix;
};

inline void checkAllStages(MPI_Comm comm, int rank, int ranks,
                           const StageConfig &config,
                           const StageCheckOptions &opts = {}) {
  if (ranks <= 0 || rank < 0 || rank >= ranks)
    throw std::runtime_error("invalid MPI rank/size (" + std::to_string(rank) +
                             "/" + std::to_string(ranks) + ")");

  const std::array<int, 8> fields{
      config.startLayer,      config.endLayer,
      config.totalLayers,     config.context,
      config.activationWidth, config.vocabSize,
      config.eosId,           static_cast<int>(config.dtype)};
  constexpr int kFieldCount = static_cast<int>(fields.size());
  std::vector<int> all(static_cast<size_t>(ranks) * kFieldCount);
  mpiCheck(MPI_Allgather(fields.data(), kFieldCount, MPI_INT, all.data(),
                         kFieldCount, MPI_INT, comm));

  if (opts.requireModelName) {
    if (opts.modelName.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("field model_name exceeds MPI message size");
    const int localLen = static_cast<int>(opts.modelName.size());
    std::vector<int> lengths(static_cast<size_t>(ranks), 0);
    mpiCheck(MPI_Allgather(&localLen, 1, MPI_INT, lengths.data(), 1, MPI_INT,
                           comm));

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    int64_t totalChars = 0;
    for (int r = 0; r < ranks; ++r) {
      if (lengths[r] < 0)
        throw std::runtime_error("field model_name has negative length on rank " +
                                 std::to_string(r));
      if (totalChars > std::numeric_limits<int>::max() - lengths[r])
        throw std::runtime_error("field model_name aggregate length overflows");
      displs[r] = static_cast<int>(totalChars);
      totalChars += lengths[r];
    }

    std::vector<char> gathered(static_cast<size_t>(totalChars));
    const char *sendPtr = opts.modelName.empty() ? "" : opts.modelName.data();
    char *recvPtr = gathered.empty() ? nullptr : gathered.data();
    mpiCheck(MPI_Allgatherv(sendPtr, localLen, MPI_CHAR, recvPtr,
                            lengths.data(), displs.data(), MPI_CHAR, comm));

    std::vector<std::string> modelNames(static_cast<size_t>(ranks));
    for (int r = 0; r < ranks; ++r) {
      modelNames[r].assign(gathered.data() + displs[r],
                           static_cast<size_t>(lengths[r]));
    }
    for (int r = 0; r < ranks; ++r) {
      if (modelNames[r].empty())
        throw std::runtime_error("field model_name is empty on rank " +
                                 std::to_string(r));
      if (modelNames[r] != modelNames[0])
        throw std::runtime_error(
            "field model_name mismatch: rank " + std::to_string(r) + " has '" +
            modelNames[r] + "', expected '" + modelNames[0] +
            "' (from rank 0)");
    }
  }

  const int refTotalLayers = all[2];
  const int refContext = all[3];
  const int refWidth = all[4];
  const int refVocab = all[5];
  const int refEos = all[6];
  const int refDtype = all[7];

  if (refTotalLayers <= 0)
    throw std::runtime_error("field totalLayers must be positive (rank 0 has " +
                             std::to_string(refTotalLayers) + ")");
  if (refContext <= 0)
    throw std::runtime_error("field context must be positive (rank 0 has " +
                             std::to_string(refContext) + ")");
  if (refWidth <= 0)
    throw std::runtime_error(
        "field activationWidth must be positive (rank 0 has " +
        std::to_string(refWidth) + ")");
  if (refVocab <= 0)
    throw std::runtime_error("field vocabSize must be positive (rank 0 has " +
                             std::to_string(refVocab) + ")");
  if (refDtype != static_cast<int>(ActivationType::F16))
    throw std::runtime_error("field dtype must be f16 (rank 0 has " +
                             std::string(activationTypeName(
                                 static_cast<ActivationType>(refDtype))) +
                             ")");

  int expectedStart = 0;
  for (int r = 0; r < ranks; ++r) {
    const int *stage = all.data() + static_cast<size_t>(r) * kFieldCount;
    const int startLayer = stage[0];
    const int endLayer = stage[1];
    const int totalLayers = stage[2];
    const int context = stage[3];
    const int width = stage[4];
    const int vocab = stage[5];
    const int eos = stage[6];
    const int dtype = stage[7];

    if (totalLayers != refTotalLayers)
      throw std::runtime_error("field totalLayers mismatch: rank " +
                               std::to_string(r) + " has " +
                               std::to_string(totalLayers) + ", expected " +
                               std::to_string(refTotalLayers) +
                               " (from rank 0)");
    if (context != refContext)
      throw std::runtime_error("field context mismatch: rank " +
                               std::to_string(r) + " has " +
                               std::to_string(context) + ", expected " +
                               std::to_string(refContext) + " (from rank 0)");
    if (width != refWidth)
      throw std::runtime_error("field activationWidth mismatch: rank " +
                               std::to_string(r) + " has " +
                               std::to_string(width) + ", expected " +
                               std::to_string(refWidth) + " (from rank 0)");
    if (vocab != refVocab)
      throw std::runtime_error("field vocabSize mismatch: rank " +
                               std::to_string(r) + " has " +
                               std::to_string(vocab) + ", expected " +
                               std::to_string(refVocab) + " (from rank 0)");
    if (eos != refEos)
      throw std::runtime_error("field eosId mismatch: rank " +
                               std::to_string(r) + " has " +
                               std::to_string(eos) + ", expected " +
                               std::to_string(refEos) + " (from rank 0)");
    if (dtype != refDtype || dtype != static_cast<int>(ActivationType::F16))
      throw std::runtime_error(
          "field dtype mismatch: rank " + std::to_string(r) + " has " +
          std::string(activationTypeName(static_cast<ActivationType>(dtype))) +
          ", expected f16");

    if (endLayer <= startLayer)
      throw std::runtime_error("field startLayer/endLayer invalid on rank " +
                               std::to_string(r) + ": [" +
                               std::to_string(startLayer) + ", " +
                               std::to_string(endLayer) + ")");

    if (r == 0 && startLayer != 0)
      throw std::runtime_error("field startLayer invalid on first rank 0: "
                               "expected 0, got " +
                               std::to_string(startLayer));
    if (r > 0) {
      if (startLayer < expectedStart)
        throw std::runtime_error("field startLayer/endLayer overlap: rank " +
                                 std::to_string(r) + " startLayer=" +
                                 std::to_string(startLayer) +
                                 " overlaps rank " + std::to_string(r - 1) +
                                 " endLayer=" + std::to_string(expectedStart));
      if (startLayer > expectedStart)
        throw std::runtime_error("field startLayer/endLayer gap (missing "
                                 "layers): rank " +
                                 std::to_string(r - 1) + " endLayer=" +
                                 std::to_string(expectedStart) + " < rank " +
                                 std::to_string(r) + " startLayer=" +
                                 std::to_string(startLayer));
    }
    expectedStart = endLayer;
  }

  if (expectedStart != refTotalLayers) {
    if (expectedStart < refTotalLayers)
      throw std::runtime_error(
          "field endLayer/totalLayers gap (missing trailing layers): last rank " +
          std::to_string(ranks - 1) + " endLayer=" +
          std::to_string(expectedStart) + " < totalLayers=" +
          std::to_string(refTotalLayers));
    throw std::runtime_error(
        "field endLayer/totalLayers exceeds model: last rank " +
        std::to_string(ranks - 1) + " endLayer=" +
        std::to_string(expectedStart) + " > totalLayers=" +
        std::to_string(refTotalLayers));
  }
}

namespace detail {
inline void traceStageCall(const std::string &prefix, int rank,
                           const char *phase, int step, int pos) {
  if (prefix.empty())
    return;
  const std::string path = prefix + ".rank" + std::to_string(rank) + ".log";
  std::ofstream out(path, std::ios::app);
  if (!out)
    throw std::runtime_error("cannot open stage trace file " + path);
  out << "rank=" << rank << " phase=" << phase << " step=" << step
      << " pos=" << pos << '\n';
  out.flush();
}
} // namespace detail

inline void generateDistributed(const DistributedGenerateOptions &opts,
                                Stage &stage, MPI_Comm comm, int rank,
                                int ranks) {
  const auto &config = stage.configuration();
  const bool first = (rank == 0);
  const bool last = (rank == ranks - 1);

  if (opts.maxNewTokens < 0)
    throw std::runtime_error("field maxNewTokens must be non-negative (got " +
                             std::to_string(opts.maxNewTokens) + ")");
  if (opts.testEosAtStep < 0)
    throw std::runtime_error("field testEosAtStep must be non-negative (got " +
                             std::to_string(opts.testEosAtStep) + ")");

  BlockingActivation activation(config, comm);

  int promptRows = 0;
  int resolvedMaxNewTokens = 0;
  std::vector<int64_t> tokenInput;
  if (first) {
    if (opts.vocabPath.empty())
      throw std::runtime_error("field vocab_uri missing on first stage");
    if (opts.prompt.empty() || opts.prompt.front() == ' ')
      throw std::runtime_error(
          "field prompt must be non-empty and start with a non-space character");
    buddy::Text<size_t, 2> text(opts.prompt);
    // Space is mapped to a two-byte character. Allow the chat markers too
    // so the tokenizer can report a too-long prompt before the model runs.
    text.tokenizeDeepSeekR1(
        opts.vocabPath,
        std::max<size_t>(static_cast<size_t>(config.context),
                         2 * opts.prompt.size() + 6));
    if (text.getTokenCnt() == 0)
      throw std::runtime_error("field prompt produced zero tokens");
    if (text.getTokenCnt() > static_cast<size_t>(config.context))
      throw std::runtime_error("tokenized prompt (" +
                               std::to_string(text.getTokenCnt()) +
                               ") exceeds model context (" +
                               std::to_string(config.context) + ")");
    promptRows = static_cast<int>(text.getTokenCnt());
    tokenInput.assign(text.getData(), text.getData() + config.context);

    const int maxAllowed = config.context - promptRows + 1;
    if (opts.maxNewTokens == 0) {
      if (!opts.allowZeroMaxNewTokens)
        throw std::runtime_error("generation length must be positive");
      resolvedMaxNewTokens = maxAllowed;
    } else if (opts.maxNewTokens > maxAllowed) {
      throw std::runtime_error(
          "field maxNewTokens (" + std::to_string(opts.maxNewTokens) +
          ") exceeds remaining context limit (" + std::to_string(maxAllowed) +
          ", context=" + std::to_string(config.context) +
          ", promptRows=" + std::to_string(promptRows) + ")");
    } else {
      resolvedMaxNewTokens = opts.maxNewTokens;
    }
  }

  std::array<int, 2> broadcastMeta{promptRows, resolvedMaxNewTokens};
  mpiCheck(MPI_Bcast(broadcastMeta.data(), 2, MPI_INT, 0, comm));
  promptRows = broadcastMeta[0];
  resolvedMaxNewTokens = broadcastMeta[1];

  if (promptRows <= 0 || promptRows > config.context)
    throw std::runtime_error("broadcast promptRows out of context range (" +
                             std::to_string(promptRows) + ")");
  const int maxAllowed = config.context - promptRows + 1;
  if (resolvedMaxNewTokens <= 0 || resolvedMaxNewTokens > maxAllowed)
    throw std::runtime_error("generation length (" +
                             std::to_string(resolvedMaxNewTokens) +
                             ") exceeds remaining context limit (" +
                             std::to_string(maxAllowed) + ")");

  std::vector<uint16_t> hidden;
  if (!first)
    hidden.resize(static_cast<size_t>(config.context) * config.activationWidth);

  std::vector<float> logits;
  std::ofstream raw;
  if (last) {
    logits.resize(static_cast<size_t>(config.vocabSize));
    if (!opts.logitsDumpPath.empty() && opts.logitsDumpPath != "-") {
      raw.open(opts.logitsDumpPath, std::ios::binary | std::ios::trunc);
      if (!raw)
        throw std::runtime_error("cannot open " + opts.logitsDumpPath);
    }
  }

  buddy::Sampler sampler(opts.samplerConfig);
  std::vector<int> generated;
  generated.reserve(static_cast<size_t>(resolvedMaxNewTokens));

  for (int step = 0; step < resolvedMaxNewTokens; ++step) {
    const ActivationHeader expected =
        makeActivationHeader(step, promptRows, config.context);
    ActivationHeader header = expected;
    void *input = tokenInput.data();
    if (!first) {
      header = activation.receive(hidden.data(), hidden.size(), rank - 1,
                                  expected);
      input = hidden.data();
    }

    if (!opts.traceStagePrefix.empty())
      detail::traceStageCall(opts.traceStagePrefix, rank,
                             header.step == 0 ? "prefill" : "decode",
                             header.step, header.pos);

    const uint16_t *output =
        header.step == 0 ? stage.prefill(input)
                         : stage.decode(input, header.pos);

    int token = 0;
    int stop = 0;
    if (last) {
      if (step == 0)
        output += static_cast<size_t>(promptRows - 1) * config.vocabSize;
      if (raw.is_open() &&
          !raw.write(reinterpret_cast<const char *>(output),
                     static_cast<std::streamsize>(logits.size() *
                                                  sizeof(uint16_t))))
        throw std::runtime_error("error writing " + opts.logitsDumpPath);
      std::transform(output, output + logits.size(), logits.begin(),
                     buddy::kvcache::detail::halfToFloat);
      token = sampler.sample(logits.data(), logits.size(), generated);
      generated.push_back(token);
      stop = (token == config.eosId ||
              (opts.testEosAtStep > 0 && step + 1 == opts.testEosAtStep))
                 ? 1
                 : 0;
      if (opts.streamJsonl) {
        std::cout << "{\"event\":\"token\",\"phase\":\""
                  << (step == 0 ? "prefill" : "decode")
                  << "\",\"iteration\":" << step << ",\"user\":0,\"token_id\":"
                  << token << ",\"text\":\"\",\"stopped\":"
                  << (stop ? "true" : "false") << "}\n"
                  << std::flush;
      }
      if (!first)
        mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, kTokenTag, comm));
    } else {
      activation.send(header, output, rank + 1);
      if (first)
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, ranks - 1, kTokenTag, comm,
                          MPI_STATUS_IGNORE));
    }

    if (ranks > 1)
      mpiCheck(MPI_Bcast(&stop, 1, MPI_INT, ranks - 1, comm));
    if (stop)
      break;
    if (first)
      tokenInput.front() = token;
  }

  if (last) {
    if (raw.is_open() && !raw.flush())
      throw std::runtime_error("error flushing " + opts.logitsDumpPath);
    if (opts.streamJsonl) {
      std::cout << "{\"event\":\"done\",\"user\":0,\"text\":\"\"}\n"
                << std::flush;
    } else {
      std::cout << "{\"prompt_tokens\":" << promptRows << ",\"token_ids\":[";
      for (size_t i = 0; i < generated.size(); ++i)
        std::cout << (i ? "," : "") << generated[i];
      std::cout << "]}\n" << std::flush;
    }
  }
}

} // namespace buddy::runtime::pipeline

#endif // BUDDY_MODELS_DEEPSEEK_R1_PIPELINE_DISTRIBUTED_GENERATE_H
