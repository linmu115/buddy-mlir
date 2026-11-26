#include "PipelineGenerate.h"
#include "ResidentGenerate.h"
#include "TemplateBundle.h"
#include <optional>
#include <thread>

using namespace buddy::runtime;
namespace {
void run(int argc, char **argv, int rank, int size) {
  RunConfig cfg;
  cfg.maxNewTokens = 0;
  std::string bundle;
  std::vector<std::string> prompts;
  auto wire = pipeline::WireDtype::Native; // Route B computes in f32.
  auto transport = pipeline::Transport::Blocking;
  bool verify = false, serve = false;
  int slots = 2;
  std::optional<std::string> list, single;
  for (int i = 1; i < argc; ++i) {
    std::string opt = argv[i];
    if (opt == "--serve") {
      serve = true;
      continue;
    }
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
      prompts.push_back(std::move(val));
    else if (opt == "--wire") {
      if (val == "f32")
        wire = pipeline::WireDtype::Native;
      else if (val == "f16")
        wire = pipeline::WireDtype::F16;
      else
        throw std::invalid_argument("--wire expects f32 or f16");
    } else if (opt == "--transport") {
      if (val == "blocking")
        transport = pipeline::Transport::Blocking;
      else if (val == "async")
        transport = pipeline::Transport::Async;
      else
        throw std::invalid_argument("--transport expects blocking or async");
    } else if (opt == "--slots") {
      auto [end, ec] =
          std::from_chars(val.data(), val.data() + val.size(), slots);
      if (ec != std::errc{} || end != val.data() + val.size() || slots < 1 ||
          slots > static_cast<int>(pipeline::kMaxBatchStreams))
        throw std::invalid_argument("--slots must be in 1..8");
    } else if (opt == "--total-tokens") {
      auto [end, ec] = std::from_chars(val.data(), val.data() + val.size(),
                                       cfg.maxNewTokens);
      if (ec != std::errc{} || end != val.data() + val.size() ||
          cfg.maxNewTokens < 0)
        throw std::invalid_argument("invalid --total-tokens");
    } else
      throw std::invalid_argument("unknown option " + opt);
  }
  if (serve && !prompts.empty())
    throw std::invalid_argument("--serve cannot be combined with --prompt");
  if (!serve &&
      (prompts.empty() || prompts.size() > pipeline::kMaxBatchStreams))
    throw std::invalid_argument("expected 1 to 8 --prompt arguments");
  const int streamCount = static_cast<int>(prompts.size());
  const int slotCount = serve ? slots : streamCount;
  auto b = TemplateBundle::load(bundle);
  auto total = b.config.totalLayers;
  LayerRange range{0, total};
  std::string planIdentity =
      b.id + ",streamCount=" + std::to_string(streamCount) +
      ",serveMode=" + std::to_string(serve) + ",slots=" + std::to_string(slots);
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
  check.wireDtype = b.spec.wireDtype = wire;
  check.transport = b.spec.transport = transport;
  check.modelName = planIdentity;
  check.requireModelName = true;
  pipeline::checkAllStages(MPI_COMM_WORLD, rank, size, b.config, check);
  if (rank == 0)
    b.verifyVocab();
  auto stage = makeTemplateStage(b, range, verify, slotCount);
  if (serve) {
    checkLocalConfig(b.spec, stage->configuration());
    const auto opts = makeGenerateOptions(b.spec, b.vocab.string(), cfg,
                                          stage->configuration(), rank, size);
    pipeline::generateResident(
        opts, *stage, MPI_COMM_WORLD, rank, size, slots,
        list.value_or(single.value_or("0:" + std::to_string(total))));
    reportPipelineMemory(rank);
  } else if (streamCount == 1) {
    cfg.prompt = std::move(prompts.front());
    runDistributedStage(*stage, b.spec, b.model, b.vocab.string(), cfg,
                        MPI_COMM_WORLD, rank, size);
  } else {
    runDistributedBatch(*stage, b.spec, b.model, b.vocab.string(), cfg, prompts,
                        MPI_COMM_WORLD, rank, size);
  }
}
} // namespace
int main(int argc, char **argv) {
  int threading;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threading);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    if (threading < MPI_THREAD_FUNNELED)
      throw std::runtime_error("Route B requires MPI_THREAD_FUNNELED");
    run(argc, argv, rank, size);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Route B rank %d: %s\n", rank, e.what());
    std::fflush(stderr);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  } catch (...) {
    pipeline::detail::abortBatch(MPI_COMM_WORLD, rank, "Route B",
                                 "unknown exception");
  }
  MPI_Finalize();
  return 0;
}
