#pragma once
#include "buddy/runtime/pipeline/RouteB.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/StageManifest.h"
#include <filesystem>
#include <functional>
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
  int tpRank = -1, tpHost = -1, bodyKvHeads = 0;
  std::string bodyShape, tpSource;
  std::vector<int> tpHeads;
  bool hybrid = false, hasMtp = false;
  std::vector<std::string> layerTypes;
  StageManifest spec{};
  std::filesystem::path vocab, runtime;
  std::string vocabDigest, runtimeDigest;
  std::map<std::filesystem::path, std::string> tokenizerAssets;
  void verifyVocab() const;
  std::map<std::string, TemplateEntry> kernels;
  std::map<std::tuple<std::string, int, std::string>, WeightEntry> weights;
  static TemplateBundle load(const std::filesystem::path &index);
};
std::string contentDigest(const void *data, size_t bytes);
void verifyFileDigest(const std::filesystem::path &path,
                      const std::string &digest, const char *role);
std::string hexDigest(const uint8_t *bytes);
std::vector<LayerRange> parseRanges(const std::string &text, int total,
                                    bool full);
void validateChain(const std::vector<LayerRange> &ranges, int total);
class MtpStage : public Stage {
public:
  virtual const float *backbone(void *input, int pos, int rows, int valid) = 0;
  virtual void commitPrefix(int pos, int kept, int rows) = 0;
  virtual void gatherShifted(const int64_t *ids, int rows, void *packed) = 0;
  virtual const float *draftHidden(const void *packed, bool half,
                                   const float *hidden, int pos, int rows,
                                   int valid) = 0;
  virtual const float *linear(const float *hidden, int rows) = 0;
};
struct TensorHooks {
  BJ47Reduce reduce;
  void *context;
  std::function<void(const float *, const int64_t *, int, int, int, int)>
      dispatch;
  std::function<void(int)> clear;
  std::function<void(int, int, int, int)> commit;
};
std::unique_ptr<MtpStage> makeTemplateStage(const TemplateBundle &, LayerRange,
                                            bool verifyWeights = false,
                                            int slots = 1,
                                            bool enableMtp = false,
                                            int maxDraft = 1,
                                            bool usePredictor = true,

                                            TensorHooks *tp = nullptr);
} // namespace buddy::runtime
