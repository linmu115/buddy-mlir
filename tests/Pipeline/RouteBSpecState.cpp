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

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  auto bundle = TemplateBundle::load(argv[1]);
  auto stage = makeTemplateStage(bundle, {0, bundle.config.totalLayers}, false,
                                 2, true, 3, false);
  auto &cfg = stage->configuration();
  auto top = [&](const float *row) {
    const float *logits = stage->linear(row, 1);
    return int(std::max_element(logits, logits + cfg.vocabSize) - logits);
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
      auto h = stage->backbone(prompt.data(), 0, cfg.prefillChunk, pos);
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
      auto h = stage->backbone(ids.data(), pos, k + 1, k + 1);
      auto logits = stage->linear(h, k + 1);
      std::vector<int> targets;
      for (int i = 0; i <= k; ++i) {
        const auto *row = logits + size_t(i) * cfg.vocabSize;
        targets.push_back(
            int(std::max_element(row, row + cfg.vocabSize) - row));
        if (i < k && targets.back() != ids[i + 1])
          break;
      }
      int kept = int(targets.size());
      if (kept != wanted + 1)
        throw std::runtime_error("unexpected prefix");
      stage->commitPrefix(pos, kept, k + 1);
      stage->setActiveSlot(0);
      for (int i = 0; i < kept; ++i) {
        auto single = stage->backbone(&ids[i], pos + i, 1, 1);
        if (top(single) != targets[i])
          throw std::runtime_error("token differs");
      }
      compareSpeculativeSlots(*stage, 0, 1);
      current = targets.back();
      pos += kept;
      ++step;
      ++frames;
    }
    if (step < 100)
      throw std::runtime_error("short rollback trace");
  }
  std::cout
      << "frames=" << frames
      << " lookup_accept=0,1,2,3 token_diff=0 backbone_state_byte_diff=0\n";
}
