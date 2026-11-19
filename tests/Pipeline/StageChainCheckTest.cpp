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
#include <iostream>
#include <stdexcept>
#include <string>

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
    else
      throw std::runtime_error("unknown mode: " + mode);
  } catch (const std::exception &err) {
    std::cerr << "rank " << rank
              << ": StageChainCheckTest FAILED: " << err.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
