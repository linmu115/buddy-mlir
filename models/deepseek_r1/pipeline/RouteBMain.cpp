#include "PipelineGenerate.h"
#include "ResidentGenerate.h"
#include "TemplateBundle.h"
#include "route_b/ChatTokenizer.h"
#include "route_b/MtpGenerate.h"
#include "route_b/TensorParallel.h"
#include <optional>
#include <thread>

using namespace buddy::runtime;
namespace {
void run(int argc, char **argv, int rank, int size) {
  RunConfig cfg;
  cfg.maxNewTokens = 0;
  std::string bundle, tpBundle;
  std::string tpTransport = optionalEnvString("BUDDY_TP_TRANSPORT");
  if (tpTransport.empty())
    tpTransport = "mpi";
  std::string tpIface;
  const int physicalRank = rank;
  MPI_Comm generatorComm = MPI_COMM_WORLD;
  std::vector<std::string> prompts;
  auto wire = pipeline::WireDtype::Native; // Route B computes in f32.
  auto transport = pipeline::Transport::Blocking;
  bool verify = false, serve = false, mtp = false;
  pipeline::mtp::Options spec;
  bool legacyMtp = false, explicitDepth = false;
  int slots = 2;
  bool launcherInput = false;
  std::optional<bool> thinking;
  std::optional<std::string> list, single;
  for (int i = 1; i < argc; ++i) {
    std::string opt = argv[i];
    if (opt == "--enable-thinking" || opt == "--no-thinking") {
      thinking = opt == "--enable-thinking";
      continue;
    }
    if (opt == "--mtp") {
      mtp = legacyMtp = true;
      spec.mode = pipeline::mtp::Mode::Mtp;
      continue;
    }
    if (opt == "--launcher-stdin") {
      launcherInput = true;
      continue;
    }
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
    else if (opt == "--tp-bundle")
      tpBundle = val;
    else if (opt == "--tp-transport")
      tpTransport = val;
    else if (opt == "--tp-tcp-iface")
      tpIface = val;
    else if (opt == "--ranges")
      list = val;
    else if (opt == "--spec-mode") {
      mtp = true;
      legacyMtp = false;
      if (val == "mtp")
        spec.mode = pipeline::mtp::Mode::Mtp;
      else if (val == "ngram")
        spec.mode = pipeline::mtp::Mode::Ngram;
      else if (val == "dspark")
        spec.mode = pipeline::mtp::Mode::DSpark;
      else
        throw std::invalid_argument("--spec-mode expects mtp, ngram or dspark");
    } else if (opt == "--max-draft") {
      explicitDepth = mtp = true;
      auto [end, ec] =
          std::from_chars(val.data(), val.data() + val.size(), spec.maxDraft);
      if (ec != std::errc{} || end != val.data() + val.size() ||
          spec.maxDraft < 1 || spec.maxDraft > 3)
        throw std::invalid_argument("--max-draft expects 1..3");
    } else if (opt == "--dspark-tau" || opt == "--dspark-min-p") {
      size_t used = 0;
      double p = std::stod(val, &used);
      if (used != val.size() || !(p >= 0 && p <= 1))
        throw std::invalid_argument(opt + " expects a probability in [0,1]");
      (opt == "--dspark-tau" ? spec.tau : spec.minP) = p;
      mtp = true;
    } else if (opt == "--stage-range")
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
  if (legacyMtp && !explicitDepth)
    spec.maxDraft = 1;
  const bool predictor = spec.mode != pipeline::mtp::Mode::Ngram;
  if (launcherInput && (tpBundle.empty() || !serve))
    throw std::invalid_argument(
        "--launcher-stdin requires --tp-bundle and --serve");
  if (!tpBundle.empty() && (!bundle.empty() || list || single))
    throw std::invalid_argument("--tp-bundle owns both complete layer ranges");
  const auto tpMode = parseTpTransport(tpTransport);
  auto launch =
      nlohmann::json::array({mtp, serve, slotCount, int(wire), int(transport),
                             int(spec.mode), spec.maxDraft, spec.tau, spec.minP,
                             launcherInput})
          .dump();
  auto b = tpBundle.empty() ? TemplateBundle::load(bundle)
                            : TensorParallel::load(tpBundle, rank, size, launch,
                                                   MPI_COMM_WORLD, tpMode);
  std::unique_ptr<TensorParallel> tp;
  if (!tpBundle.empty()) {
    tp = std::make_unique<TensorParallel>(MPI_COMM_WORLD, rank, b,
                                          wire == pipeline::WireDtype::F16,
                                          tpMode, tpIface);
    if (physicalRank == b.tpHost)
      std::cerr << "tp_transport=" << tpTransport << '\n';
    rank = 0;
    size = 1;
    generatorComm = MPI_COMM_SELF;
  }
  if (mtp && !b.hybrid)
    throw std::invalid_argument("speculation requires a hybrid bundle");
  if (mtp && predictor && !b.hasMtp)
    throw std::invalid_argument(
        "--mtp requires a bundle with predictor weights");
  auto total = b.config.totalLayers;
  LayerRange range{0, total};
  std::string planIdentity =
      b.id + ",mtp=" + std::to_string(mtp) +
      nlohmann::json({int(spec.mode), spec.maxDraft, spec.tau, spec.minP})
          .dump() +
      ",streamCount=" + std::to_string(streamCount) +
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
  pipeline::checkAllStages(generatorComm, rank, size, b.config, check);
  if (rank == 0 && (!tp || physicalRank == b.tpHost))
    b.verifyVocab();
  auto stage =
      makeTemplateStage(b, range, verify, slotCount, mtp, spec.maxDraft,
                        predictor, tp ? &tp->hooks : nullptr);
  std::vector<int> pids;
  if (tp && serve) {
    pids.resize(2);
    int pid = getpid();
    mpiCheck(MPI_Allgather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT,
                           MPI_COMM_WORLD));
  }
  if (tp && physicalRank != b.tpHost) {
    std::thread relay;
    if (launcherInput && physicalRank == 0) {
      relay = std::thread([&] {
        try {
          tp->relayInput();
        } catch (const std::exception &e) {
          pipeline::detail::abortBatch(MPI_COMM_WORLD, physicalRank,
                                       "TP input relay", e.what());
        }
      });
    }
    try {
      tp->worker(*stage);
    } catch (const std::exception &e) {
      pipeline::detail::abortBatch(MPI_COMM_WORLD, physicalRank, "TP worker",
                                   e.what());
    }
    if (relay.joinable())
      relay.join();
    tp->reportMemory();
    reportPipelineMemory(physicalRank);
    return;
  }
  checkLocalConfig(b.spec, stage->configuration());
  auto opts = makeGenerateOptions(b.spec, b.vocab.string(), cfg,
                                  stage->configuration(), rank, size, bool(tp));
  opts.servePids = std::move(pids);
  if (launcherInput && b.tpHost != 0)
    opts.readServeLine = [&](std::string &line) {
      return tp->receiveInput(line);
    };
  if (thinking && b.tokenizerAssets.empty())
    throw std::invalid_argument("thinking requires a chat template");
  std::unique_ptr<ChatTokenizer> tokenizer;
  if (rank == 0 && !b.tokenizerAssets.empty()) {
    tokenizer = std::make_unique<ChatTokenizer>(b.vocab.parent_path());
    opts.chatEncode = [&](const std::string &text, bool structured) {
      using json = nlohmann::json;
      auto request =
          structured
              ? json::parse(text)
              : json{{"messages",
                      json::array({{{"role", "user"}, {"content", text}}})}};
      if (!request.contains("enable_thinking"))
        request["enable_thinking"] = thinking.value_or(true);
      if (!request.at("enable_thinking").is_boolean())
        throw std::invalid_argument("enable_thinking must be boolean");
      return tokenizer->encode(request);
    };
  }
  if (mtp) {
    pipeline::mtp::generate(
        opts, *stage, generatorComm, rank, size,
        b.weights.count({"embed", -1, "f16"}), serve, prompts,
        list.value_or(single.value_or("0:" + std::to_string(total))), spec);
  } else if (serve) {
    pipeline::generateResident(
        opts, *stage, generatorComm, rank, size, slots,
        list.value_or(single.value_or("0:" + std::to_string(total))));
  } else if (streamCount == 1) {
    opts.prompt = std::move(prompts.front());
    pipeline::generateDistributed(opts, *stage, generatorComm, rank, size);
  } else {
    pipeline::generateDistributedBatch(opts, prompts, *stage, generatorComm,
                                       rank, size);
  }
  if (tp) {
    tp->stop();
    tp->reportMemory();
  }
  reportPipelineMemory(physicalRank);
}
} // namespace
int main(int argc, char **argv) {
  int threading;
  int required = MPI_THREAD_FUNNELED;
  for (int i = 1; i < argc; ++i)
    if (std::string_view(argv[i]) == "--launcher-stdin")
      required = MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, required, &threading);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    if (threading < required)
      throw std::runtime_error("MPI lacks the requested thread support");
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
