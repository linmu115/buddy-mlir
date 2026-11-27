#include "../TensorParallel.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>

using namespace buddy::runtime;
namespace {
using Clock = std::chrono::steady_clock;
using Microseconds = std::chrono::duration<double, std::micro>;

struct Options {
  int hops = 128, elems = 2560, iters = 20;
  std::string iface;
};

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i += 2) {
    std::string key(argv[i]);
    if (i + 1 == argc)
      throw std::invalid_argument("missing value for " + key);
    std::string value(argv[i + 1]);
    if (key == "--iface") {
      o.iface = value;
      continue;
    }
    int n;
    auto [end, ec] =
        std::from_chars(value.data(), value.data() + value.size(), n);
    if (ec != std::errc{} || end != value.data() + value.size() || n <= 0)
      throw std::invalid_argument("expected a positive integer for " + key);
    if (key == "--hops")
      o.hops = n;
    else if (key == "--elems")
      o.elems = n;
    else if (key == "--iters")
      o.iters = n;
    else
      throw std::invalid_argument("unknown option " + key);
  }
  return o;
}

double quantile(const std::vector<double> &sorted, double q) {
  return sorted[size_t(std::ceil(q * sorted.size())) - 1];
}

void benchmark(const Options &o, int rank) {
  TemplateBundle b{};
  b.config.activationWidth = o.elems;
  b.config.prefillChunk = 1;
  std::vector<float> input(size_t(o.elems) * o.hops);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = (int((i * (rank ? 17 : 31) + rank) % 4093) - 2046) / 997.f;
  for (bool half : {false, true}) {
    const char *wire = half ? "f16" : "f32";
    std::vector<float> reference;
    for (const char *name : {"mpi", "tcp", "tcp-spin"}) {
      TensorParallel tp(MPI_COMM_WORLD, rank, b, half, parseTpTransport(name),
                        o.iface);
      std::vector<double> samples, totals;
      std::vector<double> hopUs(o.hops), slowest(o.hops);
      std::vector<float> work(input.size());
      float maxDiff = 0;
      bool mismatch = false;
      for (int iter = -1; iter < o.iters; ++iter) {
        work = input;
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = Clock::now(), prev = start;
        for (int hop = 0; hop < o.hops; ++hop) {
          float *x = work.data() + size_t(hop) * o.elems;
          if (tp.hooks.reduce(tp.hooks.context, x, o.elems))
            throw std::runtime_error("benchmark reduction failed");
          auto now = Clock::now();
          hopUs[hop] = Microseconds(now - prev).count();
          prev = now;
        }
        double total = Microseconds(prev - start).count() / 1000, slowTotal;
        if (iter < 0 && reference.empty())
          reference = work; // Actual matching-precision MPI warmup output.
        if (iter < 0)
          continue;
        mismatch |= std::memcmp(work.data(), reference.data(),
                                work.size() * sizeof(float)) != 0;
        for (size_t i = 0; i < work.size(); ++i)
          maxDiff = std::max(maxDiff, std::abs(work[i] - reference[i]));
        MPI_Reduce(hopUs.data(), slowest.data(), o.hops, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&total, &slowTotal, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (!rank) {
          samples.insert(samples.end(), slowest.begin(), slowest.end());
          totals.push_back(slowTotal);
        }
      }
      int bad = mismatch;
      MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPI_FLOAT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (bad)
        throw std::runtime_error(
            std::string(name) + "-" + wire +
            " bitwise mismatch against MPI (max_abs_diff=" +
            std::to_string(maxDiff) + ")");
      if (!rank) {
        std::sort(samples.begin(), samples.end());
        std::sort(totals.begin(), totals.end());
        std::cout << name << '-' << wire << std::fixed << std::setprecision(3)
                  << " median_us=" << quantile(samples, .5)
                  << " p95_us=" << quantile(samples, .95) << " " << o.hops
                  << "_hop_median_ms=" << quantile(totals, .5) << " vs=mpi-"
                  << wire << " max_abs_diff=" << maxDiff << '\n';
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    if (size != 2)
      throw std::invalid_argument("tp_allreduce_bench requires two ranks");
    benchmark(parse(argc, argv), rank);
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
}
