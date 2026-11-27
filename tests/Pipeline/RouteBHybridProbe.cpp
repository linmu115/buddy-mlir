// Exercise the exact DSO ABI on native or RISC-V and dump every numerical
// boundary. The Python oracle owns fixture creation and comparison.
#include "buddy/runtime/pipeline/RouteB.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
namespace fs = std::filesystem;

template <typename T> struct Buffer {
  std::unique_ptr<T, decltype(&free)> ptr{nullptr, free};
  size_t size;

  explicit Buffer(size_t count) : size(count) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, count * sizeof(T)))
      throw std::bad_alloc();
    ptr.reset(static_cast<T *>(p));
    std::fill_n(data(), size, T{});
  }
  T *data() const { return ptr.get(); }
  void read(fs::path path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(data()), size * sizeof(T)) ||
        in.peek() != EOF)
      throw std::runtime_error("weight size: " + path.string());
  }
};

struct Kernel {
  std::unique_ptr<void, decltype(&dlclose)> handle;
  const BJ47KernelInfoV2 *info;
  explicit Kernel(const fs::path &path)
      : handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), dlclose) {
    if (!handle)
      throw std::runtime_error(dlerror());
    auto *prefix =
        symbol<decltype(&bj47_kernel_info_v1)>("bj47_kernel_info_v1")();
    if (!prefix || prefix->struct_bytes != sizeof(BJ47KernelInfoV2))
      throw std::runtime_error("expected hybrid descriptor");
    info = reinterpret_cast<const BJ47KernelInfoV2 *>(prefix);
  }

  template <typename T> T symbol(const char *name) {
    auto *fn = dlsym(handle.get(), name);
    if (!fn)
      throw std::runtime_error(dlerror());
    return reinterpret_cast<T>(fn);
  }
};

struct Layer {
  Buffer<float> w, a, b;
  Buffer<int8_t> q4;
  decltype(&layer_prefill) prefill, decode;
  Layer(Kernel &kernel, const fs::path &weights)
      : w(kernel.info->base.f32_elements), a(kernel.info->state_a_elements),
        b(kernel.info->state_b_elements), q4(kernel.info->base.i8_elements),
        prefill(kernel.symbol<decltype(prefill)>("layer_prefill")),
        decode(kernel.symbol<decltype(decode)>("layer_decode")) {
    w.read(weights);
    if (q4.size)
      q4.read(fs::path(weights).replace_extension(".q4"));
  }

  void run(bool pre, const float *in, const BJ47FrameV1 &frame, float *out) {
    BJ47WeightsV1 weights{w.data(), q4.size ? q4.data() : nullptr};
    if ((pre ? prefill : decode)(&weights, in, a.data(), b.data(), &frame, out))
      throw std::runtime_error("layer failed");
  }
};

// Compare both rows and recurrent snapshots at the DSO boundary, before
// argmax can conceal rounding or a state transition error.
void verifyRows(const fs::path &dir) {
  Kernel full(dir / "full_attn_layer.so"), gdn(dir / "gdn_layer.so");
  size_t width = full.info->base.width, rotary = full.info->rotary_dim;
  Buffer<float> x(2 * width), one(2 * width), both(2 * width);
  for (size_t i = 0; i < x.size; ++i)
    x.data()[i] = float(int(i * 17 % 47) - 23) / 19;
  Buffer<float> inv(full.info->base.rope_elements), cos(2 * rotary),
      sin(2 * rotary);
  inv.read(dir / "rope.f32");
  auto prepare = full.symbol<decltype(&prepare_decode)>("prepare_decode");
  for (int i = 0; i < 2; ++i)
    if (prepare(inv.data(), 13 + i, cos.data() + i * rotary,
                sin.data() + i * rotary, nullptr))
      throw std::runtime_error("prepare verify failed");
  auto equal = [](const float *x, const float *y, size_t n) {
    if (std::memcmp(x, y, n * sizeof(float)))
      throw std::runtime_error("verify differs from decode");
  };
  for (Kernel *kernel : {&gdn, &full}) {
    auto path = dir / (kernel == &gdn ? "gdn-0.f32" : "full_attn-3.f32");
    Layer ref(*kernel, path), test(*kernel, path);
    for (size_t i = 0; i < ref.a.size; ++i)
      ref.a.data()[i] = test.a.data()[i] = float(i % 13) / 17;
    for (size_t i = 0; i < ref.b.size; ++i)
      ref.b.data()[i] = test.b.data()[i] = float(i % 11) / 19;
    Buffer<float> sa(ref.a.size), sb(ref.b.size);
    BJ47VerifyRows aux{{2}, sa.data(), sb.data()};
    BJ47FrameV1 frame{13, cos.data(), sin.data(), &aux};
    BJ47WeightsV1 w{test.w.data(), test.q4.data()};
    auto fn = kernel->symbol<decltype(&layer_verify_2)>("layer_verify_2");
    if (fn(&w, x.data(), test.a.data(), test.b.data(), &frame, both.data()))
      throw std::runtime_error("verify failed");
    BJ47FrameRowsV2 single{1};
    for (int i = 0; i < 2; ++i) {
      BJ47FrameV1 row{13 + i, cos.data() + i * rotary, sin.data() + i * rotary,
                      &single};
      ref.run(false, x.data() + i * width, row, one.data() + i * width);
      equal(one.data() + i * width, both.data() + i * width, width);
      if (!i && kernel == &gdn) {
        equal(ref.a.data(), sa.data(), sa.size);
        equal(ref.b.data(), sb.data(), sb.size);
      }
    }
    equal(ref.a.data(), test.a.data(), ref.a.size);
    equal(ref.b.data(), test.b.data(), ref.b.size);
  }
  std::cout << "verify rows 0/1, GDN snapshot and final GDN/KV: bit_diff=0\n";
}

template <typename T> void dump(std::ostream &out, const T *p, size_t count) {
  if (!out.write(reinterpret_cast<const char *>(p), count * sizeof(T)))
    throw std::runtime_error("write failed");
}

// Compare an individual production layer across two builds without loading
// the other 63 layers. Include the caller-owned caches and rollback image.
void layerTrace(const fs::path &dir, const fs::path &weight, const char *role,
                int rows, int position, const fs::path &trace, bool benchmark) {
  Kernel kernel(dir / (std::string(role) + ".so"));
  auto &info = *kernel.info;
  int capacity = rows <= 2 ? rows : info.base.chunk;
  if (rows < 1 || rows > capacity || position < 0 ||
      position + capacity > int(info.base.context))
    throw std::runtime_error("invalid layer trace range");
  Layer layer(kernel, weight);
  Buffer<float> x(capacity * info.base.width), y(x.size), sa(layer.a.size),
      sb(layer.b.size), cosine(capacity * info.rotary_dim),
      sine(capacity * info.rotary_dim);
  auto seed = [](Buffer<float> &buffer, unsigned int state, float scale) {
    for (size_t i = 0; i < buffer.size; ++i) {
      state = state * 1664525u + 1013904223u;
      buffer.data()[i] = (int(state >> 16) - 32768) * scale;
    }
  };
  seed(x, 1, 1.f / 32768);
  seed(layer.a, 2, 1.f / 65536);
  seed(layer.b, 3, 1.f / 65536);
  if (info.rotary_dim) {
    Buffer<float> inv(info.base.rope_elements);
    inv.read(weight.parent_path() / "rope.f32");
    auto prep = kernel.symbol<decltype(&prepare_decode)>("prepare_decode");
    for (int t = 0; t < rows; ++t)
      if (prep(inv.data(), position + t, cosine.data() + t * info.rotary_dim,
               sine.data() + t * info.rotary_dim, nullptr))
        throw std::runtime_error("RoPE failed");
  }
  BJ47VerifyRows aux{{rows}, sa.data(), sb.data()};
  BJ47FrameV1 frame{position, cosine.data(), sine.data(), &aux};
  BJ47WeightsV1 w{layer.w.data(), layer.q4.data()};
  auto run = rows == 2
                 ? kernel.symbol<decltype(&layer_decode)>("layer_verify_2")
             : rows == 1 ? layer.decode
                         : layer.prefill;
  auto call = [&] {
    if (run(&w, x.data(), layer.a.data(), layer.b.data(), &frame, y.data()))
      throw std::runtime_error("layer call failed");
  };
  if (benchmark) {
    // Reset caller state outside each timed interval. Report all eight samples.
    std::array<int64_t, 8> elapsed;
    for (int i = -2; i < int(elapsed.size()); ++i) {
      seed(layer.a, 2, 1.f / 65536);
      seed(layer.b, 3, 1.f / 65536);
      auto start = std::chrono::steady_clock::now();
      call();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
      if (i >= 0)
        elapsed[i] = ns;
    }
    std::cout << role << " M=" << rows << " prior=" << position << " ns";
    for (auto ns : elapsed)
      std::cout << ' ' << ns;
    std::cout << '\n';
    return;
  }
  call();
  std::ofstream out(trace, std::ios::binary);
  dump(out, y.data(), y.size);
  dump(out, layer.a.data(), layer.a.size);
  dump(out, layer.b.data(), layer.b.size);
  if (rows == 2 && !info.rotary_dim) {
    dump(out, sa.data(), sa.size);
    dump(out, sb.data(), sb.size);
  }
}

void probe(const fs::path &dir, const fs::path &input, const fs::path &output,
           int rows, int steps, int64_t padding) {
  Kernel embed(dir / "embed.so"), gdn(dir / "gdn_layer.so"),
      full(dir / "full_attn_layer.so"), head(dir / "lm_head.so");
  const auto &config = full.info->base;
  const int chunk = config.chunk, width = config.width,
            vocab = config.vocab_size;
  if (rows <= 0 || steps <= 0 || rows + steps > int(config.context) + 1)
    throw std::runtime_error("invalid probe dimensions");
  auto &descriptor = *embed.info;
  const bool half = descriptor.embed_dtype == 2;
  Buffer<float> table(half ? 1 : descriptor.base.f32_elements);
  Buffer<uint16_t> tableHalf(half ? descriptor.f16_elements : 1);
  if (half)
    tableHalf.read(dir / "embed.f16");
  else
    table.read(dir / "embed.f32");

  auto embedFloatPre =
      half ? nullptr : embed.symbol<decltype(&embed_prefill)>("embed_prefill");
  auto embedFloatDec =
      half ? nullptr : embed.symbol<decltype(&embed_decode)>("embed_decode");
  auto embedHalfPre =
      half ? embed.symbol<decltype(&embed_prefill_f16)>("embed_prefill_f16")
           : nullptr;
  auto embedHalfDec =
      half ? embed.symbol<decltype(&embed_decode_f16)>("embed_decode_f16")
           : nullptr;
  auto prepPre = full.symbol<decltype(&prepare_prefill)>("prepare_prefill");
  auto prepDec = full.symbol<decltype(&prepare_decode)>("prepare_decode");
  auto headPre = head.symbol<decltype(&lm_head_prefill)>("lm_head_prefill");
  auto headDec = head.symbol<decltype(&lm_head_decode)>("lm_head_decode");
  Buffer<float> projection(head.info->base.f32_elements),
      inv(config.rope_elements);
  projection.read(dir / "head.f32");
  inv.read(dir / "rope.f32");
  Buffer<int8_t> packed(head.info->base.i8_elements);
  if (packed.size)
    packed.read(dir / "head.q4");
  BJ47WeightsV1 weight{projection.data(),
                       packed.size ? packed.data() : nullptr};

  std::array<Layer, 4> layers{{{gdn, dir / "gdn-0.f32"},
                               {gdn, dir / "gdn-1.f32"},
                               {gdn, dir / "gdn-2.f32"},
                               {full, dir / "full_attn-3.f32"}}};
  Buffer<float> a(chunk * width), b(chunk * width), logits(vocab);
  Buffer<float> cos(chunk * full.info->rotary_dim), sin(cos.size);
  Buffer<int64_t> tokens(rows), ids(chunk);
  tokens.read(input);
  std::ofstream out(output, std::ios::binary);
  std::vector<int64_t> generated;
  const bool timings = std::getenv("BUDDY_HYBRID_PROBE_TIMINGS");
  auto step = [&](int pos, int validRows, bool pre, bool sample) {
    const int rc = half ? (pre ? embedHalfPre : embedHalfDec)(
                              tableHalf.data(), ids.data(), a.data())
                        : (pre ? embedFloatPre : embedFloatDec)(
                              table.data(), ids.data(), a.data());
    if (rc || (pre ? prepPre : prepDec)(inv.data(), pos, cos.data(), sin.data(),
                                        nullptr))
      throw std::runtime_error("embed/prepare failed");
    BJ47FrameRowsV2 aux{validRows};
    BJ47FrameV1 frame{pos, cos.data(), sin.data(), &aux};
    float *in = a.data(), *hidden = b.data();
    for (size_t i = 0; i < layers.size(); ++i) {
      auto &layer = layers[i];
      const auto start = std::chrono::steady_clock::now();
      layer.run(pre, in, frame, hidden);
      if (timings) {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        std::cerr << "layer_us " << pos << ' ' << validRows << ' ' << i << ' '
                  << us.count() << '\n';
      }
      dump(out, hidden, size_t(validRows) * width);
      dump(out, layer.a.data(), layer.a.size);
      dump(out, layer.b.data(), layer.b.size);
      std::swap(in, hidden);
    }
    if (sample) {
      if ((pre ? headPre : headDec)(&weight, in + (validRows - 1) * width,
                                    logits.data()))
        throw std::runtime_error("head failed");
      dump(out, logits.data(), logits.size);
      generated.push_back(
          std::max_element(logits.data(), logits.data() + vocab) -
          logits.data());
    }
  };

  for (int pos = 0; pos < rows; pos += chunk) {
    const int validRows = std::min(rows - pos, chunk);
    std::copy_n(tokens.data() + pos, validRows, ids.data());
    std::fill(ids.data() + validRows, ids.data() + chunk, padding);
    step(pos, validRows, true, pos + chunk >= rows);
  }
  for (int i = 0; i < steps - 1; ++i) {
    ids.data()[0] = generated.back();
    step(rows + i, 1, false, true);
  }
  dump(out, generated.data(), generated.size());
}
} // namespace

int main(int argc, char **argv) {
  try {
    bool benchmark = argc == 7 && std::string(argv[1]) == "--layer-bench";
    if (benchmark || (argc == 8 && std::string(argv[1]) == "--layer-trace")) {
      layerTrace(argv[2], argv[3], argv[4], std::stoi(argv[5]),
                 std::stoi(argv[6]), benchmark ? "" : argv[7], benchmark);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--verify-rows") {
      verifyRows(argv[2]);
      return 0;
    }
    if (argc != 7)
      throw std::runtime_error(
          "probe BUNDLE IDS DUMP PROMPT_ROWS STEPS PAD_ID");
    probe(argv[1], argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]),
          std::stoll(argv[6]));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
