// Compare TP logits with the unpartitioned model and every local cache byte
// with scalar replay after each possible verifier commit.
#include "TemplateBundle.h"
#include "route_b/TensorParallel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace buddy::runtime {
void compareSpeculativeSlots(const MtpStage &, int, int);
}
using namespace buddy::runtime;
static std::vector<std::string> probe(const TemplateBundle &b,
                                      const char *source, const char *wire,
                                      int rank, TpTransport mode) {
  bool half = std::string(wire) == "f16";
  TensorParallel tp(MPI_COMM_WORLD, rank, b, half, mode);
  std::vector<std::string> logits;
  auto stage = makeTemplateStage(b, {0, b.config.totalLayers}, true, 2, true, 3,
                                 true, &tp.hooks);
  if (rank != b.tpHost) {
    tp.worker(*stage);
  } else {
    auto full = TemplateBundle::load(source);
    auto ref =
        makeTemplateStage(full, {0, b.config.totalLayers}, true, 1, true, 3);
    const auto &c = b.config;
    const int vocab = c.vocabSize;
    std::vector<int64_t> ids(c.prefillChunk, 100);
    for (int i = 0; i < c.prefillChunk; ++i)
      ids[i] = 100 + i % 4;
    double maxError = 0;
    int matches = 0, expected = 0;
    auto compare = [&](const char *phase, const float *a, const float *r,
                       int rows) {
      logits.push_back(contentDigest(a, size_t(rows) * vocab * sizeof(float)));
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
    stage->commitPrefix(8, 1, 2);
    ref->commitPrefix(8, 1, 2);
    ids[0] = 103;
    run("after_rewind", 9, 1, 1);
    int pos = 10;
    for (int rows : {3, 4}) {
      for (int kept = 1; kept <= rows; ++kept) {
        run("verify", pos, rows, rows);
        stage->commitPrefix(pos, kept, rows);
        ref->commitPrefix(pos, kept, rows);
        pos += kept;
        run("after_commit", pos++, 1, 1);
      }
    }
    std::cout.precision(17);
    std::cout << "wire=" << wire << " max_abs_diff=" << maxError
              << " greedy=" << matches << "/" << expected << '\n';
    if (matches != expected || (!half && maxError >= 1e-4))
      throw std::runtime_error("TP parity threshold");
    tp.stop();
  }

  const auto &cfg = stage->configuration();
  std::vector<int64_t> prompt(cfg.prefillChunk, 100);
  constexpr int position = 7;
  const std::array<int64_t, 4> inputs{101, 102, 103, 100};
  for (int rows : {2, 3, 4}) {
    for (int kept = 1; kept <= rows; ++kept) {
      if (rank == b.tpHost) {
        for (int slot = 0; slot < 2; ++slot) {
          stage->setActiveSlot(slot);
          stage->clearSlot(slot);
          stage->backbone(prompt.data(), 0, cfg.prefillChunk, position);
        }
        stage->setActiveSlot(0);
        auto ids = inputs;
        for (int i = 0; i < kept; ++i)
          stage->backbone(&ids[i], position + i, 1, 1);
        stage->setActiveSlot(1);
        stage->backbone(ids.data(), position, rows, rows);
        stage->commitPrefix(position, kept, rows);
        tp.stop();
      } else {
        tp.worker(*stage);
      }
      compareSpeculativeSlots(*stage, 0, 1);
      std::cout << "wire=" << wire << " rank=" << rank << " rows=" << rows
                << " accepted=" << kept - 1 << " backbone_state_byte_diff=0"
                << std::endl;
    }
  }
  return logits;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    if (argc != 4 && argc != 5)
      throw std::invalid_argument(
          "expected source.json tp-directory f32|f16 [transport]");
    std::string mode = argc == 5 ? argv[4] : "mpi";
    auto transport = parseTpTransport(mode);
    auto b = TensorParallel::load(argv[2], rank, size, argv[3], MPI_COMM_WORLD,
                                  transport);
    auto baseline = probe(b, argv[1], argv[3], rank, TpTransport::Mpi);
    if (argc == 5 && probe(b, argv[1], argv[3], rank, transport) != baseline)
      throw std::runtime_error(
          "TP logit bytes differ from MPI at matching wire precision");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}
