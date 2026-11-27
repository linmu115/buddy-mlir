// Compare every logit at identical prefixes, including rollback of a rejected
// second token. The reference opens the original unpartitioned DSOs and packs.
#include "TemplateBundle.h"
#include "route_b/TensorParallel.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace buddy::runtime;
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    if (argc != 4)
      throw std::invalid_argument("expected source.json tp-directory f32|f16");
    bool half = std::string(argv[3]) == "f16";
    auto b = TensorParallel::load(argv[2], rank, size, argv[3], MPI_COMM_WORLD);
    TensorParallel tp(MPI_COMM_WORLD, rank, b, half);
    auto stage = makeTemplateStage(b, {0, b.config.totalLayers}, true, 1, true,
                                   &tp.hooks);
    if (rank != b.tpHost) {
      tp.worker(*stage);
    } else {
      auto full = TemplateBundle::load(argv[1]);
      auto ref =
          makeTemplateStage(full, {0, b.config.totalLayers}, true, 1, true);
      const auto &c = b.config;
      const int vocab = c.vocabSize;
      std::vector<int64_t> ids(c.prefillChunk, 100);
      for (int i = 0; i < c.prefillChunk; ++i)
        ids[i] = 100 + i % 4;
      double maxError = 0;
      int matches = 0, expected = 0;
      auto compare = [&](const char *phase, const float *a, const float *r,
                         int rows) {
        double maximum = 0;
        for (int i = 0; i < rows * vocab; ++i) {
          if (!std::isfinite(a[i]) || !std::isfinite(r[i]))
            throw std::runtime_error("nonfinite logit");
          maximum = std::max(maximum, std::abs(double(a[i]) - r[i]));
        }
        for (int i = 0; i < rows; ++i) {
          auto left = a + i * vocab, right = r + i * vocab;
          matches += std::max_element(left, left + vocab) - left ==
                     std::max_element(right, right + vocab) - right;
          ++expected;
        }
        maxError = std::max(maxError, maximum);
        std::cout << phase << " rows=" << rows << " max_abs_diff=" << maximum
                  << '\n';
      };
      stage->clearSlot(0);
      ref->clearSlot(0);
      auto run = [&](const char *phase, int pos, int rows, int valid) {
        auto a = stage->backbone(ids.data(), pos, rows, valid);
        auto r = ref->backbone(ids.data(), pos, rows, valid);
        for (int i = 0; i < valid; ++i)
          compare(phase, stage->linear(a + i * c.activationWidth, 1),
                  ref->linear(r + i * c.activationWidth, 1), 1);
        const bool halfEmbed = b.weights.count({"embed", -1, "f16"});
        std::vector<float> packed(size_t(rows) * c.activationWidth);
        stage->gatherShifted(ids.data(), valid, packed.data());
        auto draft =
            stage->draftHidden(packed.data(), halfEmbed, a, pos, rows, valid);
        auto oracle =
            ref->draftHidden(packed.data(), halfEmbed, r, pos, rows, valid);
        compare("predictor",
                stage->linear(draft + (valid - 1) * c.activationWidth, 1),
                ref->linear(oracle + (valid - 1) * c.activationWidth, 1), 1);
      };
      run("prefill", 0, c.prefillChunk, 7);
      run("decode", 7, 1, 1);
      run("verify_2", 8, 2, 2);
      stage->rewindOne();
      ref->rewindOne();
      ids[0] = 103;
      run("after_rewind", 9, 1, 1);
      std::cout.precision(17);
      std::cout << "wire=" << argv[3] << " max_abs_diff=" << maxError
                << " greedy=" << matches << "/" << expected << '\n';
      if (matches != expected || (!half && maxError >= 1e-4))
        throw std::runtime_error("TP parity threshold");
      tp.stop();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}
