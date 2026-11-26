#include "TemplateBundle.h"
#include "buddy/runtime/pipeline/RouteB.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace buddy::runtime {
namespace {
size_t product(std::initializer_list<size_t> dims) {
  size_t n = 1;
  for (auto d : dims) {
    if (!d || n > SIZE_MAX / d)
      throw std::length_error("Route B buffer size overflow");
    n *= d;
  }
  return n;
}
struct Buffer {
  std::unique_ptr<float, decltype(&free)> ptr{nullptr, free};
  void allocate(size_t elements, bool zero = false) {
    size_t bytes = product({elements, sizeof(float)});
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes))
      throw std::bad_alloc();
    ptr.reset(static_cast<float *>(p));
    if (zero)
      std::memset(p, 0, bytes);
  }
  float *data() const { return ptr.get(); }
};
struct Mapping {
  void *addr;
  size_t bytes;
  explicit Mapping(const WeightEntry &e) : bytes(e.bytes) {
    int fd = open(e.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error("open " + e.path.string());
    struct stat st{};
    bool valid = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
                 st.st_size >= 0 && static_cast<uint64_t>(st.st_size) == bytes;
    addr = valid ? mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0)
                 : MAP_FAILED;
    close(fd);
    if (addr == MAP_FAILED)
      throw std::runtime_error("weight length/mmap " + e.path.string());
  }
  Mapping(const Mapping &) = delete;
  ~Mapping() { munmap(addr, bytes); }
};
struct Library {
  using Handle = std::unique_ptr<void, int (*)(void *)>;
  Handle handle{nullptr, dlclose};
  const BJ47KernelInfoV1 *info = nullptr;
  template <class T> T symbol(const char *name) {
    auto p = dlsym(handle.get(), name);
    if (!p)
      throw std::runtime_error(std::string(name) + ": " + dlerror());
    return reinterpret_cast<T>(p);
  }
  void open(const TemplateBundle &b, const std::string &role, unsigned id) {
    auto &e = b.kernels.at(role);
    std::ifstream input(e.path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(e.bytes))
      throw std::runtime_error("kernel size: " + e.path.string());
    std::vector<char> code(e.bytes);
    input.seekg(0);
    if (!input.read(code.data(), code.size()) ||
        contentDigest(code.data(), code.size()) != e.digest)
      throw std::runtime_error("kernel digest: " + e.path.string());
    handle.reset(dlopen(e.path.c_str(), RTLD_LOCAL | RTLD_NOW));
    if (!handle)
      throw std::runtime_error(e.path.string() + ": " + dlerror());
    info = symbol<decltype(&bj47_kernel_info_v1)>("bj47_kernel_info_v1")();
    auto &c = b.config;
    if (!info || info->struct_bytes != sizeof(*info) || info->role != id ||
        info->abi_version != unsigned(e.abi) ||
        info->chunk != unsigned(c.prefillChunk) ||
        info->context != unsigned(c.context) ||
        info->width != unsigned(c.activationWidth) ||
        info->kv_heads != unsigned(b.kvHeads) ||
        info->head_dim != unsigned(b.headDim) ||
        info->vocab_size != unsigned(c.vocabSize) ||
        hexDigest(info->layout_id) != e.layout ||
        hexDigest(info->shape_id) != b.shape || info->aux_prefill_bytes ||
        info->aux_decode_bytes || !info->required_alignment ||
        64 % info->required_alignment ||
        info->rope_elements != (id == 2 ? unsigned(b.headDim) / 2 : 0) ||
        (id == 1 && info->i8_elements))
      throw std::runtime_error("incompatible kernel descriptor: " + role);
  }
};
struct Phase {
  decltype(&prepare_prefill) prepare;
  decltype(&layer_prefill) layer;
  decltype(&embed_prefill) embed = nullptr;
  decltype(&lm_head_prefill) head = nullptr;
};
struct Layer {
  BJ47WeightsV1 weights;
  std::vector<Buffer> k, v;
};
class TemplateStage final : public Stage {
  StageConfig cfg;
  Library decoder, embed, head;
  Phase pre{}, dec{};
  std::vector<std::unique_ptr<Mapping>> mappings;
  std::vector<Layer> layers;
  const float *rope = nullptr, *table = nullptr;
  BJ47WeightsV1 projection{};
  Buffer a, b, cosine, sine, logits;
  std::vector<int64_t> tokenStorage;
  size_t mapped = 0;
  bool verifyWeights;
  const int slots;
  int activeSlot = 0;

  const void *map(const WeightEntry &e, size_t count) {
    if (e.elements != count)
      throw std::runtime_error("DSO weight elements: " + e.path.string());
    mappings.push_back(std::make_unique<Mapping>(e));
    if (verifyWeights &&
        contentDigest(mappings.back()->addr, e.bytes) != e.digest)
      throw std::runtime_error("weight digest: " + e.path.string());
    mapped += e.bytes;
    return mappings.back()->addr;
  }
  const void *run(void *input, int pos, bool prefill, bool needLogits) {
    auto &p = prefill ? pre : dec;
    int rows = prefill ? cfg.prefillChunk : 1;
    if (!input || pos < 0 || pos > cfg.context - rows)
      throw std::out_of_range("invalid stage input/position");
    auto checked = [&](int code, int layer, const char *operation) {
      if (code)
        throw std::runtime_error(
            std::string(operation) + (prefill ? " prefill" : " decode") +
            " layer=" + std::to_string(layer) + " pos=" + std::to_string(pos) +
            " rc=" + std::to_string(code));
    };
    BJ47FrameV1 frame{pos, cosine.data(), sine.data(), nullptr};
    checked(p.prepare(rope, pos, cosine.data(), sine.data(), nullptr),
            cfg.startLayer, "prepare");
    auto in = static_cast<const float *>(input);
    float *out = a.data();
    if (table) {
      // Scalar decode tokens can reside on an unaligned MPI caller stack.
      void *aligned = tokenStorage.data();
      size_t space = tokenStorage.size() * sizeof(int64_t);
      auto ids = static_cast<int64_t *>(
          std::align(64, rows * sizeof(int64_t), aligned, space));
      std::copy_n(static_cast<const int64_t *>(input), rows, ids);
      checked(p.embed(table, ids, out), 0, "embed");
      in = out;
      out = b.data();
    } else if (reinterpret_cast<uintptr_t>(in) %
               decoder.info->required_alignment) {
      // Transport-owned receive buffers only promise float alignment.
      // Use the spare aligned buffer as input to the first local layer.
      std::copy_n(in, size_t(rows) * cfg.activationWidth, b.data());
      in = b.data();
    }
    for (int i = cfg.startLayer; i < cfg.endLayer; ++i) {
      auto &l = layers[i - cfg.startLayer];
      checked(p.layer(&l.weights, in, l.k[activeSlot].data(),
                      l.v[activeSlot].data(), &frame, out),
              i, "layer");
      in = out;
      out = (out == a.data() ? b.data() : a.data());
    }
    if (!p.head)
      return in;
    if (needLogits)
      checked(p.head(&projection, in + size_t(rows - 1) * cfg.activationWidth,
                     logits.data()),
              cfg.endLayer, "head");
    return logits.data();
  }

public:
  TemplateStage(const TemplateBundle &bundle, LayerRange range, bool verify,
                int slotCount)
      : cfg(bundle.config), verifyWeights(verify), slots(slotCount) {
    if (slots < 1)
      throw std::invalid_argument("Route B requires at least one KV slot");
    auto [start, end] = range;
    if (start < 0 || start >= end || end > cfg.totalLayers)
      throw std::invalid_argument("invalid local layer range");
    cfg.startLayer = start;
    cfg.endLayer = end;
    decoder.open(bundle, "decoder_layer", 2);
    auto weights = [&](const char *owner, int id, const Library &lib) {
      return BJ47WeightsV1{
          static_cast<const float *>(map(bundle.weights.at({owner, id, "f32"}),
                                         lib.info->f32_elements)),
          static_cast<const int8_t *>(map(bundle.weights.at({owner, id, "q4"}),
                                          lib.info->i8_elements))};
    };
    pre.prepare = decoder.symbol<decltype(pre.prepare)>("prepare_prefill");
    dec.prepare = decoder.symbol<decltype(dec.prepare)>("prepare_decode");
    pre.layer = decoder.symbol<decltype(pre.layer)>("layer_prefill");
    dec.layer = decoder.symbol<decltype(dec.layer)>("layer_decode");
    rope = static_cast<const float *>(map(
        bundle.weights.at({"rope", -1, "f32"}), decoder.info->rope_elements));
    if (start == 0) {
      embed.open(bundle, "embed", 1);
      table = static_cast<const float *>(map(
          bundle.weights.at({"embed", -1, "f32"}), embed.info->f32_elements));
      pre.embed = embed.symbol<decltype(pre.embed)>("embed_prefill");
      dec.embed = embed.symbol<decltype(dec.embed)>("embed_decode");
      tokenStorage.resize(size_t(cfg.prefillChunk) + 8);
    }
    if (end == cfg.totalLayers) {
      head.open(bundle, "lm_head", 3);
      projection = weights("head", -1, head);
      pre.head = head.symbol<decltype(pre.head)>("lm_head_prefill");
      dec.head = head.symbol<decltype(dec.head)>("lm_head_decode");
      logits.allocate(cfg.vocabSize);
    }
    size_t kv = product(
        {size_t(bundle.kvHeads), size_t(bundle.headDim), size_t(cfg.context)});
    layers.resize(end - start);
    for (int i = start; i < end; ++i) {
      auto &l = layers[i - start];
      l.weights = weights("decoder", i, decoder);
      l.k.resize(slots);
      l.v.resize(slots);
      for (int slot = 0; slot < slots; ++slot) {
        l.k[slot].allocate(kv, true);
        l.v[slot].allocate(kv, true);
      }
    }
    auto hidden =
        product({size_t(cfg.prefillChunk), size_t(cfg.activationWidth)});
    a.allocate(hidden);
    b.allocate(hidden);
    auto frame = product({size_t(cfg.prefillChunk), size_t(bundle.headDim)});
    cosine.allocate(frame);
    sine.allocate(frame);
    if (std::getenv("BUDDY_PP_TIMING") || std::getenv("BUDDY_PP_STORAGE"))
      std::fprintf(
          stderr, "pp_storage start=%d end=%d mapped_bytes=%zu kv_bytes=%zu\n",
          start, end, mapped,
          product({kv, 2 * sizeof(float), layers.size(), size_t(slots)}));
  }
  const StageConfig &configuration() const override { return cfg; }
  int maxSlots() const override { return slots; }
  void setActiveSlot(int slot) override {
    if (slot < 0 || slot >= slots)
      throw std::out_of_range("Route B KV slot");
    activeSlot = slot;
  }
  const uint16_t *prefill(void *) override {
    throw std::logic_error("Route B requires f32");
  }
  const uint16_t *decode(void *, int) override {
    throw std::logic_error("Route B requires f32");
  }
  const void *prefillData(void *p) override { return prefillChunk(p, 0, true); }
  const void *prefillChunk(void *p, int pos, bool head) override {
    return run(p, pos, true, head);
  }
  const void *decodeData(void *p, int pos) override {
    return run(p, pos, false, true);
  }
};
} // namespace
std::unique_ptr<Stage> makeTemplateStage(const TemplateBundle &b,
                                         LayerRange range, bool verify,
                                         int slots) {
  return std::make_unique<TemplateStage>(b, range, verify, slots);
}
} // namespace buddy::runtime
