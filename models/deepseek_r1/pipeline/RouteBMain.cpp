#include "PipelineGenerate.h"
#include "TemplateBundle.h"
#include <optional>
#include <thread>

using namespace buddy::runtime;
namespace {
void run(int argc, char **argv, int rank, int size) {
  RunConfig cfg;
  cfg.maxNewTokens = 0;
  std::string bundle;
  bool verify = false;
  std::optional<std::string> list, single;
  for (int i = 1; i < argc; ++i) {
    std::string opt = argv[i];
    if (opt == "--verify-weights") {
      verify = true;
      continue;
    }
    if (++i == argc)
      throw std::invalid_argument("missing value for " + opt);
    std::string val = argv[i];
    if (opt == "--bundle")
      bundle = val;
    else if (opt == "--ranges")
      list = val;
    else if (opt == "--stage-range")
      single = val;
    else if (opt == "--prompt")
      cfg.prompt = val;
    else if (opt == "--total-tokens") {
      auto [end, ec] = std::from_chars(val.data(), val.data() + val.size(),
                                       cfg.maxNewTokens);
      if (ec != std::errc{} || end != val.data() + val.size() ||
          cfg.maxNewTokens < 0)
        throw std::invalid_argument("invalid --total-tokens");
    } else
      throw std::invalid_argument("unknown option " + opt);
  }
  auto b = TemplateBundle::load(bundle);
  auto total = b.config.totalLayers;
  LayerRange range{0, total};
  std::string planIdentity = b.id;
  if (list && single)
    throw std::invalid_argument("conflicting range options");
  if (list) {
    auto plan = parseRanges(*list, total, true);
    if (plan.size() != size_t(size))
      throw std::invalid_argument("--ranges count differs from MPI size");
    for (auto [a, e] : plan)
      planIdentity += "," + std::to_string(a) + ":" + std::to_string(e);
    range = plan[rank];
  } else if (single)
    range = parseRanges(*single, total, false).front();
  else if (size != 1)
    throw std::invalid_argument("multiple ranks require explicit ranges");
  b.config.startLayer = b.spec.startLayer = range.first;
  b.config.endLayer = b.spec.endLayer = range.second;
  pipeline::StageCheckOptions check;
  check.modelName = planIdentity;
  check.requireModelName = true;
  pipeline::checkAllStages(MPI_COMM_WORLD, rank, size, b.config, check);
  auto stage = makeTemplateStage(b, range, verify);
  runDistributedStage(*stage, b.spec, b.model, b.vocab.string(), cfg,
                      MPI_COMM_WORLD, rank, size);
}
} // namespace
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    run(argc, argv, rank, size);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Route B rank %d: %s\n", rank, e.what());
    std::fflush(stderr);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
  return 0;
}
