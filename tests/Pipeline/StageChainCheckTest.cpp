//===- StageChainCheckTest.cpp - Pipeline stage-chain MPI unit tests ------===//
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

#include "DistributedGenerate.h"

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using buddy::runtime::pipeline::checkAllStages;
using buddy::runtime::pipeline::StageCheckOptions;

class FakeStage final : public Stage {
public:
  explicit FakeStage(StageConfig cfg) : cfg(cfg) {}

  const StageConfig &configuration() const override { return cfg; }
  const uint16_t *prefill(void *) override { return nullptr; }
  const uint16_t *decode(void *, int) override { return nullptr; }

private:
  StageConfig cfg;
};

StageConfig makeStageConfig(int startLayer, int endLayer, int totalLayers = 28,
                            int context = 64, int activationWidth = 1536,
                            int vocabSize = 151936, int eosId = 151643,
                            ActivationType dtype = ActivationType::F16,
                            int prefillChunk = 0) {
  StageConfig cfg{};
  cfg.startLayer = startLayer;
  cfg.endLayer = endLayer;
  cfg.totalLayers = totalLayers;
  cfg.context = context;
  cfg.activationWidth = activationWidth;
  cfg.vocabSize = vocabSize;
  cfg.eosId = eosId;
  cfg.dtype = dtype;
  cfg.prefillChunk = prefillChunk;
  return cfg;
}

StageCheckOptions makeCheckOpts(const std::string &modelName,
                                bool requireModelName = true) {
  StageCheckOptions opts;
  opts.modelName = modelName;
  opts.requireModelName = requireModelName;
  return opts;
}

void expectFailure(int rank, int ranks, const StageConfig &cfg,
                   const StageCheckOptions &opts,
                   const std::string &expectedSubstr,
                   const std::string &caseName) {
  FakeStage stage(cfg);
  try {
    checkAllStages(MPI_COMM_WORLD, rank, ranks, stage.configuration(), opts);
  } catch (const std::runtime_error &err) {
    const std::string message = err.what();
    if (message.find(expectedSubstr) == std::string::npos) {
      throw std::runtime_error(caseName + ": diagnostic '" + message +
                               "' did not contain '" + expectedSubstr + "'");
    }
    return;
  }
  throw std::runtime_error(caseName + ": expected checkAllStages to throw");
}

void runValid1Rank(int rank, int ranks) {
  if (ranks != 1)
    throw std::runtime_error("valid-1rank requires 1 rank");
  FakeStage stage(makeStageConfig(0, 28));
  checkAllStages(MPI_COMM_WORLD, rank, ranks, stage.configuration(),
                 makeCheckOpts("deepseek_r1_f16", true));
  std::cout << "PASS stage-chain valid-1rank\n";
}

void runValid2Rank(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("valid-2rank requires 2 ranks");
  FakeStage stage(rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(14, 28));
  checkAllStages(MPI_COMM_WORLD, rank, ranks, stage.configuration(),
                 makeCheckOpts("deepseek_r1_f16", true));
  checkAllStages(MPI_COMM_WORLD, rank, ranks, stage.configuration(),
                 makeCheckOpts("", false));
  // Valid 2-rank f32 + chunked prefill (w4g32).
  FakeStage f32Stage(
      rank == 0 ? makeStageConfig(0, 14, 28, 1024, 1536, 151936, 151643,
                                  ActivationType::F32, 64)
                : makeStageConfig(14, 28, 28, 1024, 1536, 151936, 151643,
                                  ActivationType::F32, 64));
  checkAllStages(MPI_COMM_WORLD, rank, ranks, f32Stage.configuration(),
                 makeCheckOpts("deepseek_r1_w4g32_prefill_chunk64", true));
  if (rank == 0)
    std::cout << "PASS stage-chain valid-2rank\n";
}

void runValid3Rank(int rank, int ranks) {
  if (ranks != 3)
    throw std::runtime_error("valid-3rank requires 3 ranks");
  StageConfig cfg = makeStageConfig(0, 7);
  if (rank == 1)
    cfg = makeStageConfig(7, 14);
  else if (rank == 2)
    cfg = makeStageConfig(14, 28);
  FakeStage stage(cfg);
  checkAllStages(MPI_COMM_WORLD, rank, ranks, stage.configuration(),
                 makeCheckOpts("deepseek_r1_f16", true));
  if (rank == 0)
    std::cout << "PASS stage-chain valid-3rank\n";
}

void runOverlap2Rank(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("overlap-2rank requires 2 ranks");
  const StageConfig cfg =
      rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(7, 14);
  expectFailure(rank, ranks, cfg, makeCheckOpts("deepseek_r1_f16", true),
                "overlap", "overlap-2rank");
  if (rank == 0)
    std::cout << "PASS stage-chain overlap-2rank\n";
}

void runOverlap3Rank(int rank, int ranks) {
  if (ranks != 3)
    throw std::runtime_error("overlap-3rank requires 3 ranks");
  StageConfig cfg = makeStageConfig(0, 14);
  if (rank == 1)
    cfg = makeStageConfig(7, 14);
  else if (rank == 2)
    cfg = makeStageConfig(14, 28);
  expectFailure(rank, ranks, cfg, makeCheckOpts("deepseek_r1_f16", true),
                "overlap", "overlap-3rank");
  if (rank == 0)
    std::cout << "PASS stage-chain overlap-3rank\n";
}

void runGap2Rank(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("gap-2rank requires 2 ranks");
  const auto opts = makeCheckOpts("deepseek_r1_f16", true);

  // Middle gap: [0, 7) + [14, 28)
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 7) : makeStageConfig(14, 28),
                opts, "gap (missing layers)", "gap-middle-0-7-14-28");

  // Middle gap: [0, 14) + [21, 28)
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(21, 28),
                opts, "gap (missing layers)", "gap-middle-0-14-21-28");

  // Trailing gap: [0, 7) + [7, 21) out of 28
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 7) : makeStageConfig(7, 21),
                opts, "missing trailing layers", "gap-trailing");

  // Reversed order / non-zero first start: [14, 28) + [0, 14)
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(14, 28) : makeStageConfig(0, 14),
                opts, "startLayer invalid on first rank 0", "reversed-start");

  // End exceeds totalLayers: [0, 14) + [14, 30) out of 28
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(14, 30),
                opts, "exceeds model", "exceeds-total-layers");

  if (rank == 0)
    std::cout << "PASS stage-chain gap-2rank\n";
}

void runModelNameMismatch(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("model-name-mismatch requires 2 ranks");
  const StageConfig cfg =
      rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(14, 28);

  expectFailure(
      rank, ranks, cfg,
      makeCheckOpts(rank == 0 ? "deepseek_r1_f16" : "other_model_f16", true),
      "model_name mismatch", "model-name-mismatch");

  expectFailure(rank, ranks, cfg,
                makeCheckOpts(rank == 0 ? "deepseek_r1_f16" : "", true),
                "model_name is empty", "model-name-empty");

  if (rank == 0)
    std::cout << "PASS stage-chain model-name-mismatch\n";
}

void runFieldMismatch(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("field-mismatch requires 2 ranks");
  const auto opts = makeCheckOpts("deepseek_r1_f16", true);

  const auto cfg = rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(14, 28);
  auto policy = opts;
  using namespace buddy::runtime::pipeline;
  if (rank == 1)
    policy.wireDtype = WireDtype::F16;
  expectFailure(rank, ranks, cfg, policy, "pp_wire_dtype mismatch: rank 1",
                "wire");
  policy.wireDtype = WireDtype::Native;
  if (rank == 1)
    policy.transport = Transport::Async;
  expectFailure(rank, ranks, cfg, policy, "pp_transport mismatch: rank 1",
                "transport");
  policy.transport = Transport::Async;
  expectFailure(rank, ranks, cfg, policy, "async is not implemented", "async");

  // totalLayers mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28)
                          : makeStageConfig(14, 28, 32),
                opts, "totalLayers mismatch", "totalLayers-mismatch");

  // context mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28, 64)
                          : makeStageConfig(14, 28, 28, 128),
                opts, "context mismatch", "context-mismatch");

  // activationWidth mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28, 64, 1536)
                          : makeStageConfig(14, 28, 28, 64, 2048),
                opts, "activationWidth mismatch", "activationWidth-mismatch");

  // vocabSize mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28, 64, 1536, 151936)
                          : makeStageConfig(14, 28, 28, 64, 1536, 32000),
                opts, "vocabSize mismatch", "vocabSize-mismatch");

  // eosId mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28, 64, 1536, 151936, 151643)
                          : makeStageConfig(14, 28, 28, 64, 1536, 151936, 2),
                opts, "eosId mismatch", "eosId-mismatch");

  // dtype mismatch
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14, 28, 64, 1536, 151936, 151643,
                                            ActivationType::F16)
                          : makeStageConfig(14, 28, 28, 64, 1536, 151936,
                                            151643, ActivationType::F32),
                opts, "dtype mismatch", "dtype-mismatch");

  // prefillChunk mismatch
  expectFailure(
      rank, ranks,
      rank == 0 ? makeStageConfig(0, 14, 28, 1024, 1536, 151936, 151643,
                                  ActivationType::F32, 64)
                : makeStageConfig(14, 28, 28, 1024, 1536, 151936, 151643,
                                  ActivationType::F32, 32),
      opts, "prefillChunk mismatch", "prefillChunk-mismatch");

  // Local inverted/empty interval on rank 1
  expectFailure(rank, ranks,
                rank == 0 ? makeStageConfig(0, 14) : makeStageConfig(14, 14),
                opts, "startLayer/endLayer invalid", "empty-local-interval");

  if (rank == 0)
    std::cout << "PASS stage-chain field-mismatch\n";
}

void runTokenLimit(int rank, int ranks) {
  if (ranks != 1)
    throw std::runtime_error("token-limit requires 1 rank");
  using buddy::runtime::pipeline::resolveGeneratedTokenLimit;
  using buddy::runtime::pipeline::TokenLimitMode;

  auto expectEq = [](int actual, int expected, const char *label) {
    if (actual != expected) {
      throw std::runtime_error(std::string(label) + ": expected " +
                               std::to_string(expected) + ", got " +
                               std::to_string(actual));
    }
  };
  auto expectThrow = [](int maxTokens, TokenLimitMode mode, bool allowZero,
                        int promptRows, int context,
                        const std::string &expectedSubstr, const char *label) {
    try {
      (void)resolveGeneratedTokenLimit(maxTokens, mode, allowZero, promptRows,
                                       context);
    } catch (const std::runtime_error &err) {
      const std::string msg = err.what();
      if (msg.find(expectedSubstr) == std::string::npos) {
        throw std::runtime_error(std::string(label) + ": diagnostic '" + msg +
                                 "' did not contain '" + expectedSubstr + "'");
      }
      return;
    }
    throw std::runtime_error(std::string(label) + ": expected throw");
  };

  // TotalSequenceLength mode (matches runtime/llm/TextGeneration.cpp):
  //   1 <= N <= P   -> 1 generated token (prefill only)
  //   P < N <= C    -> N - P + 1 generated tokens
  //   N == 0        -> C - P + 1 (if allowZero)
  //   N > C         -> error (static pipeline KV cache)
  constexpr int kCtx = 64;
  expectEq(resolveGeneratedTokenLimit(56, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           32, "total-m1-56");
  expectEq(resolveGeneratedTokenLimit(26, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           2, "total-p-plus-1");
  expectEq(resolveGeneratedTokenLimit(25, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           1, "total-eq-p");
  expectEq(resolveGeneratedTokenLimit(10, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           1, "total-lt-p");
  expectEq(resolveGeneratedTokenLimit(1, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           1, "total-1");
  expectEq(resolveGeneratedTokenLimit(64, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           40, "total-eq-ctx");
  expectEq(resolveGeneratedTokenLimit(0, TokenLimitMode::TotalSequenceLength,
                                      true, 25, kCtx),
           40, "total-zero");
  expectEq(resolveGeneratedTokenLimit(64, TokenLimitMode::TotalSequenceLength,
                                      true, 64, kCtx),
           1, "total-p-eq-ctx");
  expectEq(resolveGeneratedTokenLimit(0, TokenLimitMode::TotalSequenceLength,
                                      true, 64, kCtx),
           1, "total-p-eq-ctx-zero");
  expectEq(resolveGeneratedTokenLimit(10, TokenLimitMode::TotalSequenceLength,
                                      true, 7, kCtx),
           4, "total-short-prompt");

  expectThrow(65, TokenLimitMode::TotalSequenceLength, true, 25, kCtx,
              "exceeds pipeline stage context", "total-gt-ctx");
  expectThrow(4096, TokenLimitMode::TotalSequenceLength, true, 25, kCtx,
              "exceeds pipeline stage context", "total-default-gt-ctx");
  expectThrow(-1, TokenLimitMode::TotalSequenceLength, true, 25, kCtx,
              "must be non-negative", "total-negative");
  expectThrow(0, TokenLimitMode::TotalSequenceLength, false, 25, kCtx,
              "generation length must be positive", "total-zero-disallowed");

  // NewTokens mode (used by standalone pipeline example):
  expectEq(resolveGeneratedTokenLimit(32, TokenLimitMode::NewTokens, false, 25,
                                      kCtx),
           32, "new-32");
  expectEq(resolveGeneratedTokenLimit(40, TokenLimitMode::NewTokens, false, 25,
                                      kCtx),
           40, "new-max-allowed");
  expectEq(resolveGeneratedTokenLimit(0, TokenLimitMode::NewTokens, true, 25,
                                      kCtx),
           40, "new-zero-allowed");
  expectThrow(41, TokenLimitMode::NewTokens, false, 25, kCtx,
              "exceeds remaining context limit", "new-gt-max-allowed");
  expectThrow(0, TokenLimitMode::NewTokens, false, 25, kCtx,
              "generation length must be positive", "new-zero-disallowed");

  if (rank == 0)
    std::cout << "PASS stage-chain token-limit\n";
}

class FakeChunkedF32Stage final : public Stage {
public:
  FakeChunkedF32Stage(StageConfig cfg, std::vector<int> stepWinningTokens)
      : cfg(cfg), winners(std::move(stepWinningTokens)),
        logits(static_cast<size_t>(cfg.vocabSize), -1.0f) {}

  const StageConfig &configuration() const override { return cfg; }
  const uint16_t *prefill(void *) override {
    throw std::logic_error("unexpected f16 prefill");
  }
  const uint16_t *decode(void *, int) override {
    throw std::logic_error("unexpected f16 decode");
  }

  const void *prefillChunk(void *, int, bool computeLogits) override {
    ++prefillCalls;
    if (computeLogits)
      return emitLogitsForCurrentStep();
    return logits.data();
  }

  const void *decodeData(void *, int) override {
    ++decodeCalls;
    return emitLogitsForCurrentStep();
  }

  int prefillCalls = 0;
  int decodeCalls = 0;
  int emittedSteps = 0;
  std::vector<float> dumpedExpected;

private:
  const float *emitLogitsForCurrentStep() {
    std::fill(logits.begin(), logits.end(), -1.25f);
    const int winner = winners.at(static_cast<size_t>(emittedSteps));
    logits[static_cast<size_t>(winner)] = 7.5f + static_cast<float>(emittedSteps);
    dumpedExpected.insert(dumpedExpected.end(), logits.begin(), logits.end());
    ++emittedSteps;
    return logits.data();
  }

  StageConfig cfg;
  std::vector<int> winners;
  std::vector<float> logits;
};

void runTokenizerStop(int rank, int ranks, const std::string &qwen3Vocab,
                      const std::string &r1Vocab,
                      const std::string &dumpPath) {
  if (ranks != 1)
    throw std::runtime_error("tokenizer-stop requires 1 rank");
  using buddy::runtime::pipeline::DistributedGenerateOptions;
  using buddy::runtime::pipeline::generateDistributed;
  using buddy::runtime::pipeline::isStopToken;
  using buddy::runtime::pipeline::tokenizePrompt;
  using buddy::runtime::pipeline::TokenizerKind;

  // 1. Qwen3 prompt tokenization must match whole-model Text::tokenizeQwen3
  //    bit-for-bit across normal, leading-space, and UTF-8 prompts.
  constexpr int kCtx = 64;
  for (const std::string &prompt :
       {std::string("Hello"), std::string(" Hello leading space"),
        std::string("你好 world")}) {
    std::vector<int64_t> ppTokens;
    const int ppRows = tokenizePrompt(prompt, qwen3Vocab, kCtx,
                                      TokenizerKind::Qwen3, ppTokens);

    buddy::Text<size_t, 2> wholeText(prompt);
    wholeText.tokenizeQwen3(qwen3Vocab, kCtx);
    if (static_cast<size_t>(ppRows) != wholeText.getTokenCnt())
      throw std::runtime_error("Qwen3 prompt token count mismatch for '" +
                               prompt + "'");
    if (ppTokens.size() != static_cast<size_t>(kCtx))
      throw std::runtime_error("Qwen3 tokenInput size mismatch");
    for (int i = 0; i < kCtx; ++i) {
      if (ppTokens[static_cast<size_t>(i)] !=
          static_cast<int64_t>(wholeText.getData()[i]))
        throw std::runtime_error("Qwen3 token ID mismatch at index " +
                                 std::to_string(i));
    }
  }

  // Over-context prompt must be rejected cleanly rather than truncated.
  {
    std::vector<int64_t> ignored;
    bool threw = false;
    try {
      (void)tokenizePrompt("Hello world from Qwen3", qwen3Vocab, 8,
                           TokenizerKind::Qwen3, ignored);
    } catch (const std::runtime_error &err) {
      threw = std::string(err.what()).find("exceeds model context") !=
              std::string::npos;
    }
    if (!threw)
      throw std::runtime_error(
          "Qwen3 tokenizePrompt did not reject over-context prompt");
  }

  // DeepSeek-R1 default behavior preserved: leading space still rejected,
  // regular prompt works.
  {
    std::vector<int64_t> ignored;
    bool threw = false;
    try {
      (void)tokenizePrompt(" Hello", r1Vocab, kCtx, TokenizerKind::DeepSeekR1,
                           ignored);
    } catch (const std::runtime_error &err) {
      threw = std::string(err.what()).find("non-space") != std::string::npos;
    }
    if (!threw)
      throw std::runtime_error(
          "DeepSeekR1 tokenizePrompt did not reject leading space");
    const int r1Rows =
        tokenizePrompt("Hello", r1Vocab, kCtx, TokenizerKind::DeepSeekR1,
                       ignored);
    if (r1Rows <= 0)
      throw std::runtime_error("DeepSeekR1 tokenizePrompt produced 0 tokens");
  }

  // 2. Stop token helper checks.
  if (!isStopToken(151645, 151645, {}))
    throw std::runtime_error("isStopToken failed for eosId");
  if (isStopToken(151643, 151645, {}))
    throw std::runtime_error("isStopToken false positive for empty extras");
  if (!isStopToken(151643, 151645, {151645, 151643}))
    throw std::runtime_error("isStopToken failed for secondary stop token");

  // 3. End-to-end DistributedGenerate with Qwen3 tokenizer, F32 chunked stage,
  //    secondary stop token (151643 != eosId=151645), and FP32 logits dump.
  constexpr int kVocab = 151936;
  StageConfig cfg =
      makeStageConfig(/*startLayer=*/0, /*endLayer=*/4, /*totalLayers=*/4,
                      /*context=*/32, /*activationWidth=*/16,
                      /*vocabSize=*/kVocab, /*eosId=*/151645,
                      ActivationType::F32, /*prefillChunk=*/4);
  FakeChunkedF32Stage stage(cfg, {42, 151643, 99});

  DistributedGenerateOptions opts;
  opts.prompt = "Hello";
  opts.vocabPath = qwen3Vocab;
  opts.tokenizer = TokenizerKind::Qwen3;
  opts.stopTokenIds = {151645, 151643};
  opts.tokenLimitMode =
      buddy::runtime::pipeline::TokenLimitMode::TotalSequenceLength;
  opts.maxTokens = 20;
  opts.samplerConfig.temperature = 0.0f;
  opts.logitsDumpPath = dumpPath;

  generateDistributed(opts, stage, MPI_COMM_WORLD, rank, ranks);

  // "Hello" tokenizes to 9 Qwen3 tokens (> chunk 4): prefill emits step 0 (42),
  // decode step 1 emits 151643 (secondary stop token) and must stop before
  // step 2 (99).
  if (stage.emittedSteps != 2)
    throw std::runtime_error("expected early stop at 2 emitted steps, got " +
                             std::to_string(stage.emittedSteps));

  std::ifstream in(dumpPath, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("cannot open dumped f32 logits: " + dumpPath);
  const std::streamoff size = in.tellg();
  const std::streamoff expectedSize =
      static_cast<std::streamoff>(stage.dumpedExpected.size() * sizeof(float));
  if (size != expectedSize)
    throw std::runtime_error("dumped f32 logits size mismatch: got " +
                             std::to_string(size) + ", expected " +
                             std::to_string(expectedSize));
  std::vector<float> actual(stage.dumpedExpected.size());
  in.seekg(0);
  if (!in.read(reinterpret_cast<char *>(actual.data()), expectedSize))
    throw std::runtime_error("failed reading dumped f32 logits");
  if (std::memcmp(actual.data(), stage.dumpedExpected.data(),
                  static_cast<size_t>(expectedSize)) != 0)
    throw std::runtime_error("dumped f32 logits did not match bit-for-bit");

  std::cout << "PASS stage-chain tokenizer-stop\n";
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
    if (argc < 2)
      throw std::runtime_error("usage: buddy-pipeline-stage-chain-test MODE");
    const std::string mode = argv[1];
    if (mode == "valid-1rank")
      runValid1Rank(rank, ranks);
    else if (mode == "valid-2rank")
      runValid2Rank(rank, ranks);
    else if (mode == "valid-3rank")
      runValid3Rank(rank, ranks);
    else if (mode == "overlap-2rank")
      runOverlap2Rank(rank, ranks);
    else if (mode == "overlap-3rank")
      runOverlap3Rank(rank, ranks);
    else if (mode == "gap-2rank")
      runGap2Rank(rank, ranks);
    else if (mode == "model-name-mismatch")
      runModelNameMismatch(rank, ranks);
    else if (mode == "field-mismatch")
      runFieldMismatch(rank, ranks);
    else if (mode == "token-limit")
      runTokenLimit(rank, ranks);
    else if (mode == "tokenizer-stop") {
      if (argc < 5)
        throw std::runtime_error(
            "tokenizer-stop requires QWEN3_VOCAB R1_VOCAB DUMP_PATH");
      runTokenizerStop(rank, ranks, argv[2], argv[3], argv[4]);
    } else
      throw std::runtime_error("unknown mode: " + mode);
  } catch (const std::exception &err) {
    std::cerr << "rank " << rank
              << ": StageChainCheckTest FAILED: " << err.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
