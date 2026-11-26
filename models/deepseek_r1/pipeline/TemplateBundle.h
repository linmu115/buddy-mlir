#pragma once
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/StageManifest.h"
#include <filesystem>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

namespace buddy::runtime {
using LayerRange = std::pair<int, int>;
struct TemplateEntry {
  std::filesystem::path path;
  std::string layout;
  int abi;
  uint64_t bytes;
  std::string digest;
};
struct WeightEntry {
  std::filesystem::path path;
  size_t elements, bytes;
  std::string digest;
};
struct TemplateBundle {
  StageConfig config{};
  std::string id, shape, model;
  int kvHeads, headDim;
  StageManifest spec{};
  std::filesystem::path vocab;
  std::map<std::string, TemplateEntry> kernels;
  std::map<std::tuple<std::string, int, std::string>, WeightEntry> weights;
  static TemplateBundle load(const std::filesystem::path &index);
};
std::string contentDigest(const void *data, size_t bytes);
std::string hexDigest(const uint8_t *bytes);
std::vector<LayerRange> parseRanges(const std::string &text, int total,
                                    bool full);
void validateChain(const std::vector<LayerRange> &ranges, int total);
std::unique_ptr<Stage> makeTemplateStage(const TemplateBundle &, LayerRange,
                                         bool verifyWeights = false,
                                         int slots = 1);
} // namespace buddy::runtime
