#include "TemplateBundle.h"
#include <array>
#include <fstream>
#include <iostream>
using namespace buddy::runtime;
int main(int argc, char **argv) {
  try {
    auto b = TemplateBundle::load(argv[1]);
    auto range = parseRanges(argv[2], b.config.totalLayers, false).at(0);
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto stage = makeTemplateStage(b, range, argc > 3);
      // Inspect Linux VMAs while the stage is alive, including the actual
      // protection flags and the absence of every unowned weight.
      std::ifstream active("/proc/self/maps");
      std::string maps((std::istreambuf_iterator<char>(active)), {});
      for (const auto &[key, weight] : b.weights) {
        auto [owner, id, bucket] = key;
        bool owned =
            owner == "rope" || (owner == "embed" && !range.first) ||
            (owner == "head" && range.second == b.config.totalLayers) ||
            (owner == "decoder" && id >= range.first && id < range.second);
        auto pos = maps.find(weight.path.string() + "\n");
        if (owned != (pos != maps.npos))
          return 6;
        if (owned &&
            maps.substr(maps.rfind('\n', pos) + 1, pos - maps.rfind('\n', pos))
                    .find(" r--p ") == maps.npos)
          return 7;
      }
      alignas(64) std::array<int64_t, 64> ids{};
      alignas(64) std::array<float, 64 * 16 + 1> hidden;
      hidden.fill(float(range.first));
      void *input = range.first
                        ? static_cast<void *>(hidden.data() + repeat % 2)
                        : ids.data();
      const float *result =
          static_cast<const float *>(stage->prefillChunk(input, 0, true));
      bool last = range.second == b.config.totalLayers;
      if (last ? result[42] != 1 : result[0] != range.second)
        return 2;
      result = static_cast<const float *>(stage->decodeData(input, 64));
      if (last ? result[43] != 1 : result[0] != range.second)
        return 3;
      result =
          static_cast<const float *>(stage->prefillChunk(input, 64, false));
      if (last && result[43] != 1)
        return 4;
    }
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
      if (line.find(b.vocab.parent_path().string() + "/") != line.npos)
        return 5;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
