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

#include "PromptFramePlan.h"
#include "buddy/runtime/pipeline/mpi/AsyncMPI.h"
#include <memory>
#include <optional>

#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/KVCacheManager.h"
#include "buddy/runtime/llm/Sampler.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
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
  WireDtype wireDtype = WireDtype::Native;
  Transport transport = Transport::Blocking;
};

enum class TokenLimitMode {
  /// Match whole-model TextGeneration.cpp (`--max-tokens N`):
  /// prefill always emits 1 token (unless EOS), followed by
  /// `max(0, maxTokens - promptRows)` decode steps, producing up to
  /// `max(1, maxTokens - promptRows + 1)` tokens for `1 <= N <= context`.
  /// `0` maps to the remaining stage context (`context - promptRows + 1`) when
  /// `allowZeroTokenLimit` is true. Values `> context` are rejected because
  /// pipeline stages use a fixed-size KV cache without overflow eviction.
  TotalSequenceLength,
  /// Direct upper bound on newly generated tokens (including the first token
  /// from prefill). Used by the standalone `pipeline` example
  /// (`MAX_NEW_TOKENS`).
  NewTokens,
};

enum class TokenizerKind {
  DeepSeekR1,
  Qwen3,
  Qwen35,
};

struct DistributedGenerateOptions {
  std::function<std::vector<int64_t>(const std::string &, bool)> chatEncode;
  std::string prompt;
  std::string vocabPath;
  TokenizerKind tokenizer = TokenizerKind::DeepSeekR1;
  /// Optional additional stop token IDs beyond `config.eosId`.
  std::vector<int> stopTokenIds;
  /// Semantics applied to `maxTokens` when rank 0 resolves the generated-token
  /// count before broadcasting to all ranks.
  TokenLimitMode tokenLimitMode = TokenLimitMode::TotalSequenceLength;
  /// Token limit interpreted according to `tokenLimitMode`:
  /// - TotalSequenceLength: total sequence length N (matching whole-model
  ///   `--max-tokens N`); 0 fills remaining stage context when
  ///   `allowZeroTokenLimit` is true.
  /// - NewTokens: exact upper bound on newly generated tokens; 0 fills
  ///   remaining stage context when `allowZeroTokenLimit` is true.
  int maxTokens = 0;
  bool allowZeroTokenLimit = true;
  buddy::SamplerConfig samplerConfig{};
  /// Optional output file for sampled-row logits (FP16 or FP32 matching stage
  /// dtype; empty or "-" disables).
  std::string logitsDumpPath;
  bool streamJsonl = false;
  /// Test-only hook: 1-based step index whose sampled token is treated as EOS.
  /// 0 disables the hook.
  int testEosAtStep = 0;
  /// Test-only hook: when non-empty, appends stage prefill/decode calls to
  /// "<traceStagePrefix>.rank<rank>.log".
  std::string traceStagePrefix;
  Transport transport = Transport::Blocking;
  WireDtype wireDtype = WireDtype::Native;
  bool appendOnlyPrefill = false;
};

inline bool isStopToken(int token, int eosId,
                        const std::vector<int> &stopTokenIds) {
  return token == eosId || std::find(stopTokenIds.begin(), stopTokenIds.end(),
                                     token) != stopTokenIds.end();
}

inline int tokenizePrompt(const std::string &prompt,
                          const std::string &vocabPath, int context,
                          TokenizerKind tokenizer,
                          std::vector<int64_t> &tokenInput,
                          buddy::Text<size_t, 2> *text = nullptr) {
  if (context <= 0)
    throw std::runtime_error("field context must be positive (got " +
                             std::to_string(context) + ")");
  if (vocabPath.empty())
    throw std::runtime_error("field vocab_uri missing on first stage");
  if (tokenizer != TokenizerKind::DeepSeekR1 && prompt.empty())
    throw std::runtime_error("field prompt must be non-empty");
  if (tokenizer == TokenizerKind::DeepSeekR1 &&
      (prompt.empty() || prompt.front() == ' '))
    throw std::runtime_error("field prompt must be non-empty and start with "
                             "a non-space character");

  std::optional<buddy::Text<size_t, 2>> local;
  if (!text) {
    local.emplace("");
    local->loadVocab(vocabPath);
    text = &*local;
  }

  // Allow for the byte-level space mapping and chat markers. The tokenizer
  // can then report the true prompt length instead of truncating to context.
  const size_t capacity = std::max<size_t>(context, 2 * prompt.size() + 8);
  if (tokenizer == TokenizerKind::Qwen3)
    text->tokenizeQwen3WithLoadedVocab(prompt, capacity);
  else if (tokenizer == TokenizerKind::Qwen35)
    text->tokenizeQwen35WithLoadedVocab(prompt, capacity);
  else
    text->tokenizeDeepSeekR1WithLoadedVocab(prompt, capacity);

  if (text->getTokenCnt() == 0)
    throw std::runtime_error("field prompt produced zero tokens");
  if (text->getTokenCnt() > static_cast<size_t>(context))
    throw std::runtime_error(
        "tokenized prompt (" + std::to_string(text->getTokenCnt()) +
        ") exceeds model context (" + std::to_string(context) + ")");
  const int promptRows = static_cast<int>(text->getTokenCnt());
  tokenInput.assign(text->getData(), text->getData() + context);
  return promptRows;
}

inline int encodePrompt(const DistributedGenerateOptions &opts,
                        const StageConfig &config, const std::string &prompt,
                        std::vector<int64_t> &ids,
                        buddy::Text<size_t, 2> *legacy = nullptr,
                        bool structured = false) {
  if (!opts.chatEncode)
    return tokenizePrompt(prompt, opts.vocabPath, config.context,
                          opts.tokenizer, ids, legacy);
  ids = opts.chatEncode(prompt, structured);
  if (ids.empty() || ids.size() > size_t(config.context))
    throw std::invalid_argument("tokenized chat is outside the model context");
  for (auto id : ids)
    if (id < 0 || id >= config.vocabSize)
      throw std::invalid_argument("BPE id outside vocabulary");
  int rows = static_cast<int>(ids.size());
  ids.resize(config.context, 0);
  return rows;
}

/// Convert the user-facing token limit into the internal generated-token count
/// (including the first token sampled from prefill) on rank 0 before broadcast.
inline int resolveGeneratedTokenLimit(int maxTokens, TokenLimitMode mode,
                                      bool allowZero, int promptRows,
                                      int context) {
  if (context <= 0)
    throw std::runtime_error("field context must be positive (got " +
                             std::to_string(context) + ")");
  if (promptRows <= 0 || promptRows > context)
    throw std::runtime_error("promptRows (" + std::to_string(promptRows) +
                             ") out of context range (1.." +
                             std::to_string(context) + ")");
  if (maxTokens < 0)
    throw std::runtime_error("field maxTokens must be non-negative (got " +
                             std::to_string(maxTokens) + ")");

  const int maxAllowedNewTokens = context - promptRows + 1;
  if (maxTokens == 0) {
    if (!allowZero)
      throw std::runtime_error("generation length must be positive");
    return maxAllowedNewTokens;
  }

  if (mode == TokenLimitMode::NewTokens) {
    if (maxTokens > maxAllowedNewTokens)
      throw std::runtime_error(
          "field maxNewTokens (" + std::to_string(maxTokens) +
          ") exceeds remaining context limit (" +
          std::to_string(maxAllowedNewTokens) +
          ", context=" + std::to_string(context) +
          ", promptRows=" + std::to_string(promptRows) + ")");
    return maxTokens;
  }

  // TokenLimitMode::TotalSequenceLength (matches
  // runtime/llm/TextGeneration.cpp):
  //   1 <= N <= promptRows     -> 1 generated token (prefill only, 0 decode
  //   steps) promptRows < N <= context -> N - promptRows + 1 generated tokens
  //   N > context              -> exceeds static pipeline KV cache context.
  if (maxTokens > context)
    throw std::runtime_error(
        "field maxTokens (" + std::to_string(maxTokens) +
        ") exceeds pipeline stage context (" + std::to_string(context) +
        ", promptRows=" + std::to_string(promptRows) + ")");
  if (maxTokens <= promptRows)
    return 1;
  return maxTokens - promptRows + 1;
}

inline void checkAllStages(MPI_Comm comm, int rank, int ranks,
                           const StageConfig &config,
                           const StageCheckOptions &opts = {}) {
  if (ranks <= 0 || rank < 0 || rank >= ranks)
    throw std::runtime_error("invalid MPI rank/size (" + std::to_string(rank) +
                             "/" + std::to_string(ranks) + ")");

  const auto wire = resolveWireSpec(config.dtype, opts.wireDtype);
  const std::array<int, 13> fields{config.startLayer,
                                   config.endLayer,
                                   config.totalLayers,
                                   config.context,
                                   config.activationWidth,
                                   config.vocabSize,
                                   config.eosId,
                                   static_cast<int>(config.dtype),
                                   config.prefillChunk,
                                   static_cast<int>(opts.transport),
                                   static_cast<int>(opts.wireDtype),
                                   static_cast<int>(wire.dtype),
                                   wire.version};
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
    mpiCheck(
        MPI_Allgather(&localLen, 1, MPI_INT, lengths.data(), 1, MPI_INT, comm));

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    int64_t totalChars = 0;
    for (int r = 0; r < ranks; ++r) {
      if (lengths[r] < 0)
        throw std::runtime_error(
            "field model_name has negative length on rank " +
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
        throw std::runtime_error("field model_name mismatch: rank " +
                                 std::to_string(r) + " has '" + modelNames[r] +
                                 "', expected '" + modelNames[0] +
                                 "' (from rank 0)");
    }
  }

  const int refTotalLayers = all[2];
  const int refContext = all[3];
  const int refWidth = all[4];
  const int refVocab = all[5];
  const int refEos = all[6];
  const int refDtype = all[7];
  const int refPrefillChunk = all[8];

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
  if (refDtype != static_cast<int>(ActivationType::F16) &&
      refDtype != static_cast<int>(ActivationType::F32))
    throw std::runtime_error(
        "field dtype must be f16 or f32 (rank 0 has " +
        std::string(activationTypeName(static_cast<ActivationType>(refDtype))) +
        ")");
  if (refPrefillChunk < 0 || refPrefillChunk > refContext)
    throw std::runtime_error(
        "field prefillChunk out of context range (rank 0 has " +
        std::to_string(refPrefillChunk) + ")");

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
    const int prefillChunk = stage[8];

    if (totalLayers != refTotalLayers)
      throw std::runtime_error(
          "field totalLayers mismatch: rank " + std::to_string(r) + " has " +
          std::to_string(totalLayers) + ", expected " +
          std::to_string(refTotalLayers) + " (from rank 0)");
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
    if (dtype != refDtype)
      throw std::runtime_error(
          "field dtype mismatch: rank " + std::to_string(r) + " has " +
          std::string(activationTypeName(static_cast<ActivationType>(dtype))) +
          ", expected " +
          std::string(
              activationTypeName(static_cast<ActivationType>(refDtype))));
    if (prefillChunk != refPrefillChunk)
      throw std::runtime_error(
          "field prefillChunk mismatch: rank " + std::to_string(r) + " has " +
          std::to_string(prefillChunk) + ", expected " +
          std::to_string(refPrefillChunk) + " (from rank 0)");

    auto fieldValue = [](int field, int value) -> std::string {
      if (field == 9)
        return value == static_cast<int>(Transport::Blocking) ? "blocking"
                                                              : "async";
      if (field == 10)
        return value == static_cast<int>(WireDtype::Native) ? "native" : "f16";
      if (field == 11)
        return activationTypeName(static_cast<ActivationType>(value));
      return std::to_string(value);
    };
    constexpr const char *wireFields[] = {"pp_transport", "pp_wire_dtype",
                                          "effective wire dtype",
                                          "wire version"};
    for (int i = 9; i < kFieldCount; ++i) {
      if (stage[i] != all[i])
        throw std::runtime_error(
            std::string("field ") + wireFields[i - 9] + " mismatch: rank " +
            std::to_string(r) + " has " + fieldValue(i, stage[i]) +
            ", expected " + fieldValue(i, all[i]) + " (from rank 0)");
    }
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
                                 std::to_string(r) +
                                 " startLayer=" + std::to_string(startLayer) +
                                 " overlaps rank " + std::to_string(r - 1) +
                                 " endLayer=" + std::to_string(expectedStart));
      if (startLayer > expectedStart)
        throw std::runtime_error("field startLayer/endLayer gap (missing "
                                 "layers): rank " +
                                 std::to_string(r - 1) +
                                 " endLayer=" + std::to_string(expectedStart) +
                                 " < rank " + std::to_string(r) +
                                 " startLayer=" + std::to_string(startLayer));
    }
    expectedStart = endLayer;
  }

  if (expectedStart != refTotalLayers) {
    if (expectedStart < refTotalLayers)
      throw std::runtime_error(
          "field endLayer/totalLayers gap (missing trailing layers): last "
          "rank " +
          std::to_string(ranks - 1) +
          " endLayer=" + std::to_string(expectedStart) +
          " < totalLayers=" + std::to_string(refTotalLayers));
    throw std::runtime_error(
        "field endLayer/totalLayers exceeds model: last rank " +
        std::to_string(ranks - 1) +
        " endLayer=" + std::to_string(expectedStart) +
        " > totalLayers=" + std::to_string(refTotalLayers));
  }
}

namespace detail {
inline void traceStageCall(const std::string &prefix, int rank,
                           const char *phase, const ActivationHeader &h,
                           bool needLogits) {
  if (prefix.empty())
    return;
  const std::string path = prefix + ".rank" + std::to_string(rank) + ".log";
  std::ofstream out(path, std::ios::app);
  if (!out)
    throw std::runtime_error("cannot open stage trace file " + path);
  out << "rank=" << rank << " phase=" << phase << " step=" << h.step
      << " pos=" << h.pos << " rows=" << h.rows << " logits=" << needLogits
      << '\n';
  out.flush();
}
struct BatchPrefillState {
  int token = 0;
  int stop = 0;
  int64_t ttftUs = 0;
  std::optional<buddy::Sampler> sampler;
  std::ofstream raw;
};
} // namespace detail

// Capture the first sample and sampler state for batch decode. Both callers
// use the same PromptFramePlan, chunk transport and first-token sampling.
template <bool CapturePrefill = false>
inline void generateTokenized(const DistributedGenerateOptions &opts,
                              Stage &stage, MPI_Comm comm, int rank, int ranks,
                              int promptRows, int resolvedNewTokens,
                              std::vector<int64_t> &tokenInput,
                              AsyncActivation *async,
                              detail::BatchPrefillState *prefill = nullptr) {
  const auto &config = stage.configuration();
  const bool first = rank == 0, last = rank == ranks - 1;
  const int version = resolveWireSpec(config.dtype, opts.wireDtype).version;
  BlockingActivation activation(config, comm, opts.wireDtype);

  int capacity = 0;
  std::vector<float> hiddenF32;
  std::vector<uint16_t> hiddenF16;
  void *hidden = nullptr;
  if (!first && !async) {
    if (!computeElementCount(config.prefillChunk ? config.prefillChunk
                                                 : config.context,
                             config.activationWidth, capacity))
      throw std::length_error("activation shape exceeds MPI element count");
    if (config.dtype == ActivationType::F32) {
      hiddenF32.resize(size_t(capacity) + 15);
      void *base = hiddenF32.data();
      size_t space = hiddenF32.size() * sizeof(float);
      hidden = std::align(64, size_t(capacity) * sizeof(float), base, space);
    } else {
      hiddenF16.resize(capacity);
      hidden = hiddenF16.data();
    }
  }

  auto receive = [&](const ActivationHeader &h, bool preposted) -> void * {
    if (async) {
      if (!preposted)
        async->postReceive(h, rank - 1);
      return async->finishReceive(h, rank - 1);
    }
    FrameInterval interval(rank, h.step, "recv-wait");
    activation.receive(hidden, capacity, rank - 1, h);
    return hidden;
  };
  // Software-progress MPI providers may leave a large send idle throughout
  // the next stage computation. Opt in to completing chunk sends on the
  // two-rank path while retaining preposted receives.
  const char *completion = std::getenv("BUDDY_PP_COMPLETE_PREFILL_SEND");
  const bool completePrefill = ranks == 2 && config.prefillChunk > 0 &&
                               completion && std::string(completion) == "1";
  auto send = [&](const ActivationHeader &h, const void *output) {
    if (async) {
      async->postSend(h, output, rank + 1, completePrefill && h.rows > 1);
    } else {
      FrameInterval interval(rank, h.step, "send-wait");
      activation.send(h, output, rank + 1);
    }
  };

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
  generated.reserve(static_cast<size_t>(resolvedNewTokens));

  auto finishStepOnLast = [&](const void *rawOut, int genStep, int &token,
                              int &stop) {
    if (config.dtype == ActivationType::F16) {
      const auto *f16 = static_cast<const uint16_t *>(rawOut);
      if (raw.is_open() && !raw.write(reinterpret_cast<const char *>(f16),
                                      static_cast<std::streamsize>(
                                          logits.size() * sizeof(uint16_t))))
        throw std::runtime_error("error writing " + opts.logitsDumpPath);
      std::transform(f16, f16 + logits.size(), logits.begin(),
                     buddy::kvcache::detail::halfToFloat);
    } else {
      const auto *f32 = static_cast<const float *>(rawOut);
      if (raw.is_open() && !raw.write(reinterpret_cast<const char *>(f32),
                                      static_cast<std::streamsize>(
                                          logits.size() * sizeof(float))))
        throw std::runtime_error("error writing " + opts.logitsDumpPath);
      std::copy_n(f32, logits.size(), logits.begin());
    }
    token = sampler.sample(logits.data(), logits.size(), generated);
    generated.push_back(token);
    stop = (isStopToken(token, config.eosId, opts.stopTokenIds) ||
            (opts.testEosAtStep > 0 && genStep + 1 == opts.testEosAtStep))
               ? 1
               : 0;
    if (opts.streamJsonl) {
      std::cout << "{\"event\":\"token\",\"phase\":\""
                << (genStep == 0 ? "prefill" : "decode")
                << "\",\"iteration\":" << genStep
                << ",\"user\":0,\"token_id\":" << token
                << ",\"text\":\"\",\"stopped\":" << (stop ? "true" : "false")
                << "}\n"
                << std::flush;
    }
    if (!first)
      mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, kTokenTag, comm));
  };

  if (config.prefillChunk > 0) {
    // Chunked prefill + decode (matches ModelSession::_emit_chunked_prefill).
    const int C = config.prefillChunk;
    const PromptFramePlan plan(promptRows, C, config.context, version,
                               opts.appendOnlyPrefill);
    const void *prefillFinalOut = nullptr;
    std::vector<int64_t> chunkTokens(first && plan.prefillTokens() ? C : 0);
    if (!first && async)
      for (int i = 0; i < std::min(2, plan.size()); ++i)
        async->postReceive(plan[i].header, rank - 1);

    const bool observe = first && std::getenv("BUDDY_PP_OBSERVE");
    auto firstFrame = (observe || (CapturePrefill && first))
                          ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
    for (int i = 0; i < plan.size(); ++i) {
      const auto frame = plan[i];
      const auto &header = frame.header;
      void *input;
      if (!first) {
        input = receive(header, true);
      } else if (frame.prefill) {
        for (int r = 0; r < C; ++r)
          chunkTokens[r] =
              tokenInput[std::min(header.pos + r, plan.prefillTokens() - 1)];
        input = chunkTokens.data();
      } else {
        input = &tokenInput[header.pos];
      }

      detail::traceStageCall(opts.traceStagePrefix, rank,
                             frame.prefill ? "prefill" : "decode", header,
                             frame.needLogits);
      const void *output;
      {
        FrameInterval interval(rank, header.step, "compute");
        output = runPromptFrame(stage, input, frame, opts.appendOnlyPrefill);
      }
      if (!last) {
        send(header, output);
      } else if (frame.needLogits) {
        prefillFinalOut = output;
      }
      if (!first && async && i < plan.size() - 2)
        async->postReceive(plan[i + 2].header, rank - 1);
    }

    int frameStep = plan.size();
    int token = 0;
    int stop = 0;
    if (last) {
      finishStepOnLast(prefillFinalOut, /*genStep=*/0, token, stop);
    } else if (first) {
      mpiCheck(MPI_Recv(&token, 1, MPI_INT, ranks - 1, kTokenTag, comm,
                        MPI_STATUS_IGNORE));
    }
    if (observe || (CapturePrefill && first)) {
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - firstFrame)
                    .count();
      if constexpr (CapturePrefill)
        prefill->ttftUs = us;
      else
        std::fprintf(stderr, "pp_first rank=0 ttft_us=%lld\n", (long long)us);
    }
    if (ranks > 1)
      mpiCheck(MPI_Bcast(&stop, 1, MPI_INT, ranks - 1, comm));
    if constexpr (CapturePrefill) {
      prefill->token = token;
      prefill->stop = stop;
      if (last) {
        prefill->sampler.emplace(std::move(sampler));
        prefill->raw = std::move(raw);
      }
      return;
    }

    if (!stop) {
      int64_t nextToken = token;
      for (int genStep = 1; genStep < resolvedNewTokens; ++genStep) {
        const int pos = promptRows + (genStep - 1);
        const auto header =
            makeChunkDecodeHeader(frameStep, pos, config.context, version);
        void *input = first ? &nextToken : receive(header, false);
        detail::traceStageCall(opts.traceStagePrefix, rank, "decode", header,
                               true);
        const void *output;
        {
          FrameInterval interval(rank, header.step, "compute");
          output = stage.decodeData(input, header.pos);
        }

        token = 0;
        stop = 0;
        if (last) {
          finishStepOnLast(output, genStep, token, stop);
        } else {
          send(header, output);
          if (first)
            mpiCheck(MPI_Recv(&token, 1, MPI_INT, ranks - 1, kTokenTag, comm,
                              MPI_STATUS_IGNORE));
        }

        if (ranks > 1)
          mpiCheck(MPI_Bcast(&stop, 1, MPI_INT, ranks - 1, comm));
        if (stop)
          break;
        if (first)
          nextToken = token;
        ++frameStep;
      }
    }
  } else {
    for (int step = 0; step < resolvedNewTokens; ++step) {
      const auto header =
          makeActivationHeader(step, promptRows, config.context, version);
      void *input = first ? tokenInput.data() : receive(header, false);
      detail::traceStageCall(opts.traceStagePrefix, rank,
                             step == 0 ? "prefill" : "decode", header, true);
      const void *output;
      {
        FrameInterval interval(rank, step, "compute");
        if (config.dtype == ActivationType::F16) {
          output = header.step == 0 ? stage.prefill(input)
                                    : stage.decode(input, header.pos);
        } else {
          output = header.step == 0 ? stage.prefillData(input)
                                    : stage.decodeData(input, header.pos);
        }
      }
      int token = 0;
      int stop = 0;
      if (last) {
        const void *rowOut = output;
        if (step == 0) {
          const size_t elemOffset =
              static_cast<size_t>(promptRows - 1) * config.vocabSize;
          if (config.dtype == ActivationType::F16)
            rowOut = static_cast<const uint16_t *>(output) + elemOffset;
          else
            rowOut = static_cast<const float *>(output) + elemOffset;
        }
        finishStepOnLast(rowOut, step, token, stop);
      } else {
        send(header, output);
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

template <bool CapturePrefill = false>
inline void generatePrepared(const DistributedGenerateOptions &opts,
                             Stage &stage, MPI_Comm comm, int rank, int ranks,
                             int promptRows, int resolvedNewTokens,
                             std::vector<int64_t> &tokenInput,
                             detail::BatchPrefillState *prefill = nullptr) {
  const auto &config = stage.configuration();
  std::optional<AsyncActivation> async;
  if (ranks > 1 && opts.transport == Transport::Async)
    async.emplace(config, comm, opts.wireDtype, rank > 0, rank < ranks - 1,
                  config.prefillChunk ? 2 : 1);
  try {
    generateTokenized<CapturePrefill>(opts, stage, comm, rank, ranks,
                                      promptRows, resolvedNewTokens, tokenInput,
                                      async ? &*async : nullptr, prefill);
    if (async)
      async->drain();
  } catch (const std::exception &e) {
    if (!async)
      throw;
    std::cerr << "rank " << rank << " pipeline error: " << e.what()
              << std::endl;
    MPI_Abort(comm, 1);
    std::abort();
  } catch (...) {
    if (!async)
      throw;
    std::cerr << "rank " << rank << " pipeline error: unknown exception"
              << std::endl;
    MPI_Abort(comm, 1);
    std::abort();
  }
}

inline void generateDistributed(const DistributedGenerateOptions &opts,
                                Stage &stage, MPI_Comm comm, int rank,
                                int ranks) {
  const auto &config = stage.configuration();
  const bool first = (rank == 0);

  if (opts.maxTokens < 0)
    throw std::runtime_error("field maxTokens must be non-negative (got " +
                             std::to_string(opts.maxTokens) + ")");
  if (opts.testEosAtStep < 0)
    throw std::runtime_error("field testEosAtStep must be non-negative (got " +
                             std::to_string(opts.testEosAtStep) + ")");

  int promptRows = 0;
  int resolvedNewTokens = 0;
  std::vector<int64_t> tokenInput;
  if (first) {
    promptRows = encodePrompt(opts, config, opts.prompt, tokenInput);

    if (std::getenv("BUDDY_PP_OBSERVE")) {
      std::ostringstream ids;
      ids << "pp_prompt ids=[";
      for (int i = 0; i < promptRows; ++i)
        ids << (i ? "," : "") << tokenInput[i];
      ids << "]\n";
      std::fputs(ids.str().c_str(), stderr);
    }
    resolvedNewTokens = resolveGeneratedTokenLimit(
        opts.maxTokens, opts.tokenLimitMode, opts.allowZeroTokenLimit,
        promptRows, config.context);
  }

  std::array<int, 2> broadcastMeta{promptRows, resolvedNewTokens};
  mpiCheck(MPI_Bcast(broadcastMeta.data(), 2, MPI_INT, 0, comm));
  promptRows = broadcastMeta[0];
  resolvedNewTokens = broadcastMeta[1];

  if (promptRows <= 0 || promptRows > config.context)
    throw std::runtime_error("broadcast promptRows out of context range (" +
                             std::to_string(promptRows) + ")");
  const int maxAllowed = config.context - promptRows + 1;
  if (resolvedNewTokens <= 0 || resolvedNewTokens > maxAllowed)
    throw std::runtime_error("generation length (" +
                             std::to_string(resolvedNewTokens) +
                             ") exceeds remaining context limit (" +
                             std::to_string(maxAllowed) + ")");

  generatePrepared(opts, stage, comm, rank, ranks, promptRows,
                   resolvedNewTokens, tokenInput);
}

inline constexpr size_t kMaxBatchStreams = 8;

namespace detail {
using BatchClock = std::chrono::steady_clock;
enum class BatchActKind : uint32_t {
  Decode = 1,
  Shutdown = 2,
  PrefillStart = 3,
  PrefillChunk = 4
};
inline constexpr uint32_t kBatchActMagic = 0x424A3437;
inline constexpr int kTagBatchAct = 3;
inline constexpr int kTagBatchCtrl = 4;
struct BatchActHeader {
  uint32_t magic;
  BatchActKind kind;
  int32_t slot;
  int32_t pos;
  uint32_t wireDtype;
  uint32_t payloadBytes;
  int32_t rows;
  int32_t limit;
};
struct BatchCtrlReply {
  int32_t slot;
  int32_t token;
  int32_t stop;
  int32_t reserved;
};
static_assert(sizeof(BatchActHeader) == 32);
static_assert(sizeof(BatchCtrlReply) == 16);

struct BatchStream {
  BatchPrefillState prefill;
  std::vector<int> generated;
  int promptRows = 0;
  int limit = 0;
  int pos = 0;
  int64_t token = 0;
  bool done = false;
  bool inFlight = false;
  int64_t decodeUs = 0;
};

// Each MPI buffer stays owned by its slot until its request completes.
// The typed backing vectors also give encode/decode a correctly typed payload
// immediately after the header, without aliasing an array of bytes as floats.
template <typename WireElement> class BatchLink {
  static constexpr bool Half = std::is_same_v<WireElement, uint16_t>;
  static constexpr auto Dtype =
      static_cast<uint32_t>(Half ? ActivationType::F16 : ActivationType::F32);
  static constexpr size_t HeaderElements =
      sizeof(BatchActHeader) / sizeof(WireElement);
  const size_t width, capacity;
  const int frameBytes;
  MPI_Comm comm;
  int rank, last;
  std::vector<std::vector<WireElement>> sendSlots;
  std::vector<MPI_Request> requests;
  std::vector<BatchCtrlReply> ctrlSlots;
  std::vector<WireElement> recvBuffer;
  std::vector<float> decoded;

  static int checkedFrameBytes(size_t width) {
    if (width > (std::numeric_limits<int>::max() - sizeof(BatchActHeader)) /
                    sizeof(WireElement))
      throw std::length_error("batch activation exceeds MPI message size");
    return static_cast<int>(sizeof(BatchActHeader) +
                            width * sizeof(WireElement));
  }

  static void requireSize(MPI_Status &status, int expected) {
    int received;
    mpiCheck(MPI_Get_count(&status, MPI_BYTE, &received));
    if (received != expected)
      throw std::runtime_error("invalid batch MPI message size");
  }

public:
  BatchLink(int activationWidth, size_t slots, MPI_Comm comm, int rank,
            int last, int chunkRows = 1)
      : width(activationWidth), capacity(width * chunkRows),
        frameBytes(checkedFrameBytes(capacity)), comm(comm), rank(rank),
        last(last), sendSlots(rank < last ? slots : 0),
        requests(slots, MPI_REQUEST_NULL), ctrlSlots(rank == last ? slots : 0),
        recvBuffer(rank > 0 ? HeaderElements + capacity : 0),
        decoded(rank > 0 && Half ? capacity : 0) {
    for (auto &buffer : sendSlots)
      buffer.resize(HeaderElements + capacity);
  }

  void sendActivation(int slot, int pos, const void *output, int rows = 1,
                      BatchActKind kind = BatchActKind::Decode) {
    const size_t count = width * rows;
    if (rows < 1 || count > capacity)
      throw std::logic_error("batch send exceeds activation capacity");
    mpiCheck(MPI_Wait(&requests[slot], MPI_STATUS_IGNORE));
    auto &buffer = sendSlots[slot];
    const BatchActHeader header{
        kBatchActMagic, kind,
        slot,           pos,
        Dtype,          uint32_t(count * sizeof(WireElement)),
        rows,           0};
    std::memcpy(buffer.data(), &header, sizeof(header));
    auto *payload = buffer.data() + HeaderElements;
    const auto *floats = static_cast<const float *>(output);
    if constexpr (Half)
      encodeInto(floats, count, payload, count);
    else
      std::copy_n(floats, count, payload);
    mpiCheck(MPI_Isend(buffer.data(), checkedFrameBytes(count), MPI_BYTE,
                       rank + 1, kTagBatchAct, comm, &requests[slot]));
  }

  void startPrefill(int slot, int promptRows, int limit, int chunks) {
    if (rank < last) {
      const BatchActHeader header{kBatchActMagic, BatchActKind::PrefillStart,
                                  slot,           promptRows,
                                  Dtype,          0,
                                  chunks,         limit};
      mpiCheck(MPI_Send(&header, sizeof(header), MPI_BYTE, rank + 1,
                        kTagBatchAct, comm));
    }
  }

  BatchActHeader receiveActivation(size_t slots) {
    MPI_Status status;
    mpiCheck(MPI_Recv(recvBuffer.data(), frameBytes, MPI_BYTE, rank - 1,
                      kTagBatchAct, comm, &status));
    int bytes;
    mpiCheck(MPI_Get_count(&status, MPI_BYTE, &bytes));
    if (bytes < static_cast<int>(sizeof(BatchActHeader)))
      throw std::runtime_error("truncated batch activation header");
    BatchActHeader header;
    std::memcpy(&header, recvBuffer.data(), sizeof(header));
    if (header.magic != kBatchActMagic || header.wireDtype != Dtype)
      throw std::runtime_error("invalid batch activation header");
    if (header.kind != BatchActKind::Shutdown &&
        (header.slot < 0 || size_t(header.slot) >= slots))
      throw std::runtime_error("invalid batch activation slot");
    if (header.kind == BatchActKind::Shutdown ||
        header.kind == BatchActKind::PrefillStart) {
      if (bytes != static_cast<int>(sizeof(header)) || header.payloadBytes != 0)
        throw std::runtime_error("invalid batch control frame");
    } else {
      const size_t count = width * header.rows;
      if (header.rows < 1 || count > capacity ||
          (header.kind != BatchActKind::PrefillChunk &&
           (header.kind != BatchActKind::Decode || header.rows != 1)) ||
          bytes != checkedFrameBytes(count) ||
          header.payloadBytes != count * sizeof(WireElement) || header.limit)
        throw std::runtime_error("invalid batch activation frame");
    }
    return header;
  }

  void *activationData(int rows = 1) {
    auto *payload = recvBuffer.data() + HeaderElements;
    if constexpr (Half) {
      const size_t count = width * rows;
      decodeInto(payload, count, decoded.data(), count);
      return decoded.data();
    } else {
      // TemplateStage copies unaligned input into its aligned scratch buffer.
      return payload;
    }
  }

  void sendReply(const BatchCtrlReply &reply) {
    const int slot = reply.slot;
    mpiCheck(MPI_Wait(&requests[slot], MPI_STATUS_IGNORE));
    ctrlSlots[slot] = reply;
    mpiCheck(MPI_Isend(&ctrlSlots[slot], sizeof(reply), MPI_BYTE, 0,
                       kTagBatchCtrl, comm, &requests[slot]));
  }

  bool replyAvailable() {
    int ready;
    mpiCheck(MPI_Iprobe(last, kTagBatchCtrl, comm, &ready, MPI_STATUS_IGNORE));
    return ready != 0;
  }

  BatchCtrlReply receiveReply() {
    BatchCtrlReply reply;
    MPI_Status status;
    mpiCheck(MPI_Recv(&reply, sizeof(reply), MPI_BYTE, last, kTagBatchCtrl,
                      comm, &status));
    requireSize(status, sizeof(reply));
    return reply;
  }

  void drain() {
    mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                         MPI_STATUSES_IGNORE));
  }

  void drainAndShutdown() {
    drain();
    if (rank < last) {
      const BatchActHeader header{
          kBatchActMagic, BatchActKind::Shutdown, -1, 0, Dtype, 0, 0, 0};
      mpiCheck(MPI_Send(&header, sizeof(header), MPI_BYTE, rank + 1,
                        kTagBatchAct, comm));
    }
  }
};

inline BatchCtrlReply sampleBatchToken(const DistributedGenerateOptions &opts,
                                       const StageConfig &config, int slot,
                                       BatchStream &stream,
                                       const void *output) {
  auto &raw = stream.prefill.raw;
  if (raw.is_open() &&
      !raw.write(static_cast<const char *>(output),
                 static_cast<std::streamsize>(config.vocabSize) *
                     sizeof(float)))
    throw std::runtime_error("error writing batch logits");
  const int token = stream.prefill.sampler->sample(
      static_cast<const float *>(output), config.vocabSize, stream.generated);
  stream.generated.push_back(token);
  const size_t count = stream.generated.size();
  const bool stop =
      count == size_t(stream.limit) ||
      isStopToken(token, config.eosId, opts.stopTokenIds) ||
      (opts.testEosAtStep > 0 && count == size_t(opts.testEosAtStep));
  return {slot, token, stop, 0};
}

inline const void *decodeBatchSlot(const DistributedGenerateOptions &opts,
                                   Stage &stage, int rank, int slot,
                                   const BatchStream &stream, void *input) {
  stage.setActiveSlot(slot);
  if (!opts.traceStagePrefix.empty()) {
    const auto &config = stage.configuration();
    const auto header = makeChunkDecodeHeader(
        stream.pos - stream.promptRows + 1, stream.pos, config.context,
        resolveWireSpec(config.dtype, opts.wireDtype).version);
    traceStageCall(opts.traceStagePrefix + ".slot" + std::to_string(slot), rank,
                   "decode", header, true);
  }
  return stage.decodeData(input, stream.pos);
}

[[noreturn]] inline void abortBatch(MPI_Comm comm, int rank, const char *phase,
                                    const char *what) {
  // Abort before destroying buffers which MPI may still own.
  std::fprintf(stderr, "rank %d %s: %s\n", rank, phase, what);
  std::fflush(stderr);
  MPI_Abort(comm, 1);
  std::abort();
}

inline int64_t batchElapsedUs(BatchClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             BatchClock::now() - start)
      .count();
}

template <typename WireElement>
int64_t decodeBatch(const DistributedGenerateOptions &opts, Stage &stage,
                    MPI_Comm comm, int rank, int ranks,
                    std::vector<BatchStream> &streams) {
  const auto &config = stage.configuration();
  const int last = ranks - 1;
  BatchLink<WireElement> link(config.activationWidth, streams.size(), comm,
                              rank, last);
  try {
    if (rank != 0) {
      for (;;) {
        const auto frame = link.receiveActivation(streams.size());
        if (frame.kind == BatchActKind::Shutdown)
          break;
        auto &stream = streams[frame.slot];
        if (frame.kind != BatchActKind::Decode || stream.done ||
            frame.pos != stream.pos || stream.pos >= config.context)
          throw std::runtime_error("unexpected batch slot position");
        const auto *output = decodeBatchSlot(opts, stage, rank, frame.slot,
                                             stream, link.activationData());
        if (rank == last) {
          const auto reply =
              sampleBatchToken(opts, config, frame.slot, stream, output);
          stream.done = reply.stop;
          link.sendReply(reply);
        } else {
          link.sendActivation(frame.slot, stream.pos, output);
        }
        ++stream.pos;
      }
      link.drainAndShutdown();
      return 0;
    }

    std::deque<int> ready;
    for (size_t slot = 0; slot < streams.size(); ++slot)
      if (!streams[slot].done)
        ready.push_back(static_cast<int>(slot));
    int active = static_cast<int>(ready.size());
    int inFlight = 0;
    const auto start = BatchClock::now();
    auto complete = [&](const BatchCtrlReply &reply) {
      if (reply.slot < 0 || size_t(reply.slot) >= streams.size() ||
          !streams[reply.slot].inFlight || reply.token < 0 ||
          reply.token >= config.vocabSize ||
          (reply.stop != 0 && reply.stop != 1) || reply.reserved)
        throw std::runtime_error("unexpected batch control reply");
      auto &stream = streams[reply.slot];
      if (ranks > 1)
        stream.generated.push_back(reply.token);
      stream.token = reply.token;
      ++stream.pos;
      stream.inFlight = false;
      --inFlight;
      if (reply.stop) {
        stream.done = true;
        stream.decodeUs = batchElapsedUs(start);
        --active;
      } else {
        ready.push_back(reply.slot);
      }
    };

    while (active > 0) {
      while (inFlight > 0 && link.replyAvailable())
        complete(link.receiveReply());
      if (ready.empty() && active > 0) {
        if (inFlight == 0)
          throw std::logic_error("active batch streams have no work");
        complete(link.receiveReply());
      }
      if (ready.empty())
        continue;

      const int slot = ready.front();
      ready.pop_front();
      auto &stream = streams[slot];
      const void *output =
          decodeBatchSlot(opts, stage, rank, slot, stream, &stream.token);
      stream.inFlight = true;
      ++inFlight;
      if (ranks == 1)
        complete(sampleBatchToken(opts, config, slot, stream, output));
      else
        link.sendActivation(slot, stream.pos, output);
    }
    const auto elapsed = batchElapsedUs(start);
    link.drainAndShutdown();
    return elapsed;
  } catch (const std::exception &e) {
    abortBatch(comm, rank, "batch decode", e.what());
  } catch (...) {
    abortBatch(comm, rank, "batch decode", "unknown exception");
  }
}

inline void printBatchResult(const std::vector<BatchStream> &streams,
                             int64_t decodeUs) {
  size_t totalSteps = 0;
  for (const auto &stream : streams)
    totalSteps += stream.generated.size() - 1;
  auto rate = [](size_t tokens, int64_t us) {
    return us > 0 ? 1e6 * tokens / us : 0.0;
  };
  std::ostringstream result;
  result.precision(12);
  result << "{\"decode_wall_us\":" << decodeUs
         << ",\"aggregate_decode_tokens\":" << totalSteps
         << ",\"aggregate_decode_tok_s\":" << rate(totalSteps, decodeUs)
         << ",\"streams\":[";
  for (size_t slot = 0; slot < streams.size(); ++slot) {
    const auto &stream = streams[slot];
    const size_t steps = stream.generated.size() - 1;
    result << (slot ? "," : "") << "{\"slot\":" << slot
           << ",\"prompt_tokens\":" << stream.promptRows
           << ",\"ttft_us\":" << stream.prefill.ttftUs
           << ",\"decode_steps\":" << steps
           << ",\"stream_decode_us\":" << stream.decodeUs
           << ",\"stream_decode_tok_s\":" << rate(steps, stream.decodeUs)
           << ",\"token_ids\":[";
    for (size_t i = 0; i < stream.generated.size(); ++i)
      result << (i ? "," : "") << stream.generated[i];
    result << "]}";
  }
  std::cout << result.str() << "]}\n" << std::flush;
}
} // namespace detail

inline void generateDistributedBatch(const DistributedGenerateOptions &opts,
                                     const std::vector<std::string> &prompts,
                                     Stage &stage, MPI_Comm comm, int rank,
                                     int ranks) {
  const auto &config = stage.configuration();
  if (prompts.size() < 2 || prompts.size() > kMaxBatchStreams ||
      prompts.size() > size_t(stage.maxSlots()))
    throw std::invalid_argument("invalid batch stream count for stage");
  if (config.dtype != ActivationType::F32 || config.prefillChunk <= 0)
    throw std::invalid_argument("batch decode requires a chunked f32 stage");

  std::vector<detail::BatchStream> streams(prompts.size());
  for (size_t slot = 0; slot < streams.size(); ++slot) {
    auto &stream = streams[slot];
    std::vector<int64_t> input;
    std::array<int, 2> meta{};
    if (rank == 0) {
      meta[0] = encodePrompt(opts, config, prompts[slot], input);
      meta[1] = resolveGeneratedTokenLimit(opts.maxTokens, opts.tokenLimitMode,
                                           opts.allowZeroTokenLimit, meta[0],
                                           config.context);
    }
    mpiCheck(MPI_Bcast(meta.data(), 2, MPI_INT, 0, comm));
    stream.promptRows = stream.pos = meta[0];
    stream.limit = meta[1];

    auto prefillOpts = opts;
    if (!opts.traceStagePrefix.empty())
      prefillOpts.traceStagePrefix += ".slot" + std::to_string(slot);
    if (!opts.logitsDumpPath.empty() && opts.logitsDumpPath != "-")
      prefillOpts.logitsDumpPath += ".slot" + std::to_string(slot);

    stage.clearSlot(static_cast<int>(slot));
    stage.setActiveSlot(static_cast<int>(slot));
    generatePrepared<true>(prefillOpts, stage, comm, rank, ranks,
                           stream.promptRows, 1, input, &stream.prefill);
    stream.done = stream.prefill.stop || stream.limit == 1;
    if (rank == 0 || rank == ranks - 1) {
      stream.token = stream.prefill.token;
      stream.generated.reserve(stream.limit);
      stream.generated.push_back(stream.prefill.token);
    }
  }
  const auto decodeUs =
      ranks > 1 && opts.wireDtype == WireDtype::F16
          ? detail::decodeBatch<uint16_t>(opts, stage, comm, rank, ranks,
                                          streams)
          : detail::decodeBatch<float>(opts, stage, comm, rank, ranks, streams);
  for (auto &stream : streams) {
    auto &raw = stream.prefill.raw;
    if (raw.is_open() && !raw.flush())
      throw std::runtime_error("error flushing batch logits");
  }
  if (rank == 0)
    detail::printBatchResult(streams, decodeUs);
}

} // namespace buddy::runtime::pipeline

#endif // BUDDY_MODELS_DEEPSEEK_R1_PIPELINE_DISTRIBUTED_GENERATE_H
