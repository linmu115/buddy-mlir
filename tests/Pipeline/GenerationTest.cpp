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

#include <numeric>

namespace pp = buddy::runtime::pipeline;

// A small deterministic stage with one reusable output and position-indexed KV.
// The final stage can produce EOS through the ordinary greedy sampler.
class TestStage final : public Stage {
public:
  TestStage(int rank, int ranks, int chunk, bool eos, bool fail)
      : rank(rank), last(rank + 1 == ranks), eos(eos), fail(fail),
        cfg{rank, rank + 1, ranks, 24, 4, 4, 3, ActivationType::F32, chunk},
        kv(cfg.context), output(chunk * cfg.activationWidth) {}
  const StageConfig &configuration() const override { return cfg; }
  const uint16_t *prefill(void *) override {
    throw std::logic_error("unexpected ABI");
  }
  const uint16_t *decode(void *, int) override {
    throw std::logic_error("unexpected ABI");
  }

  const void *prefillChunk(void *in, int pos, bool logits) override {
    return forward(in, pos, cfg.prefillChunk, logits);
  }
  const void *decodeData(void *in, int pos) override {
    return forward(in, pos, 1, true);
  }

private:
  const void *forward(void *in, int pos, int rows, bool logits) {
    if (fail && rank == 1)
      throw std::runtime_error("injected stage exception");
    for (int i = 0; i < rows; ++i) {
      const int value = rank ? int(static_cast<float *>(in)[4 * i])
                             : int(static_cast<int64_t *>(in)[i]);
      kv[pos + i] = value + rank;
      std::fill_n(output.data() + 4 * i, 4, float(kv[pos + i]));
    }
    if (last && logits) {
      const int sum = std::accumulate(kv.begin(), kv.end(), 0);
      std::fill_n(output.data(), 4, -float(sum + 1));
      output[eos ? cfg.eosId : sum % 3] = float(sum + 1);
    }
    return output.data();
  }

  int rank;
  bool last, eos, fail;
  StageConfig cfg;
  std::vector<int> kv;
  std::vector<float> output;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc != 4)
      throw std::invalid_argument("expected backend, output prefix and mode");
    pp::DistributedGenerateOptions opts;
    opts.transport = std::string(argv[1]) == "async" ? pp::Transport::Async
                                                     : pp::Transport::Blocking;
    bool fail = std::string(argv[3]) == "throw";
    for (int chunk : {4, 1}) {
      for (int prompt : {1, 3, 4, 5, 8, 9}) {
        const std::string prefix = std::string(argv[2]) + "-" +
                                   std::to_string(chunk) + "-" +
                                   std::to_string(prompt);
        opts.logitsDumpPath = prefix + ".logits";
        opts.traceStagePrefix = prefix;
        // Throw with another known receive active in the window.
        const bool inject = fail && chunk == 4 && prompt == 9;
        TestStage stage(rank, ranks, chunk, prompt == 8, inject);
        std::vector<int64_t> tokens(stage.configuration().context);
        std::iota(tokens.begin(), tokens.end(), 1);
        pp::generatePrepared(opts, stage, MPI_COMM_WORLD, rank, ranks, prompt,
                             3, tokens);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return MPI_Finalize();
}
