// Drive the production stage with two independent slots. Compare every byte
// of all backbone caches, including unused KV tails, after each commit.
#include "PipelineGenerate.h"
#include "route_b/MtpGenerate.h"
#include <cstring>

namespace buddy::runtime {
void compareSpeculativeSlots(const MtpStage &, int, int);
}
using namespace buddy::runtime;
using namespace buddy::runtime::pipeline;

void check(const char *path, int rank, int ranks, bool halfWire) {
  auto bundle = TemplateBundle::load(path);
  const int layers = bundle.config.totalLayers;
  const int cut = layers / 2;
  const LayerRange range{rank == 0 ? 0 : cut,
                         ranks == 1 || rank ? layers : cut};
  auto stage = makeTemplateStage(bundle, range, false, 2, true, 3, false);
  mtp::Link link(*stage, MPI_COMM_WORLD, rank, ranks - 1,
                 bundle.weights.count({"embed", -1, "f16"}), halfWire);
  auto &cfg = stage->configuration();
  auto top = [&](const float *row) {
    int token;
    link.greedy(row, 1, &token);
    return token;
  };
  int frames = 0;
  for (int firstPrefix = 0; firstPrefix <= 3; ++firstPrefix) {
    std::vector<int64_t> history{40,  41, 42, 100, 101, 102, 103,
                                 100, 80, 40, 41,  42,  100};
    if (firstPrefix < 3)
      history[5 + firstPrefix] = 88;
    std::vector<int64_t> prompt(cfg.prefillChunk);
    std::copy(history.begin(), history.end(), prompt.begin());
    int pos = int(history.size()), current = 0;
    for (int slot = 0; slot < 2; ++slot) {
      stage->clearSlot(slot);
      stage->setActiveSlot(slot);
      auto h = link.backbone(prompt.data(), 0, cfg.prefillChunk, pos);
      current = top(h + size_t(pos - 1) * cfg.activationWidth);
    }
    history.push_back(current);
    auto lookup = mtp::lookup(history, 2, 3);
    if (lookup.count != 3)
      throw std::runtime_error("missing seeded lookup");
    int step = 0;
    while (pos + 4 <= cfg.context) {
      // First verify a genuine prompt lookup. Then alternate mixed prefixes
      // using the known 100..103 cycle of the mixed fixture.
      int k = step ? 1 + step % 3 : 3;
      int wanted = step ? (step / 3) % (k + 1) : firstPrefix;
      std::array<int64_t, 4> ids{current};
      for (int i = 0; i < k; ++i)
        ids[i + 1] =
            step ? (i == wanted ? 88 : 100 + (current - 100 + i + 1) % 4)
                 : lookup.tokens[i];
      stage->setActiveSlot(1);
      auto h = link.backbone(ids.data(), pos, k + 1, k + 1);
      mtp::Draft draft;
      draft.count = k;
      std::copy_n(ids.begin() + 1, k, draft.tokens.begin());
      const auto [accepted, correction] = link.verify(h, draft);
      if (accepted != wanted)
        throw std::runtime_error("unexpected prefix");
      const int kept = accepted + 1;
      stage->commitPrefix(pos, kept, k + 1);
      stage->setActiveSlot(0);
      for (int i = 0; i < kept; ++i) {
        auto single = link.backbone(&ids[i], pos + i, 1, 1);
        if (top(single) != (i == accepted ? correction : draft.tokens[i]))
          throw std::runtime_error("token differs");
      }
      compareSpeculativeSlots(*stage, 0, 1);
      current = correction;
      pos += kept;
      ++step;
      ++frames;
    }
    if (step < 100)
      throw std::runtime_error("short rollback trace");
  }
  std::cout << "wire=" << (halfWire ? "f16" : "f32") << " rank=" << rank
            << " frames=" << frames
            << " lookup_accept=0,1,2,3 token_diff=0 backbone_state_byte_diff=0"
            << std::endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  try {
    if (argc < 2 || argc > 3 || ranks > 2)
      throw std::invalid_argument(
          "expected bundle.json [f32|f16] under one or two ranks");
    const std::string wire = argc == 3 ? argv[2] : "f32";
    if (wire != "f32" && wire != "f16")
      throw std::invalid_argument("expected f32 or f16 wire");
    check(argv[1], rank, ranks, wire == "f16");
  } catch (const std::exception &e) {
    std::cerr << "rank=" << rank << ' ' << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
}
