#include "TemplateBundle.h"
#include "buddy/runtime/pipeline/ActivationWire.h"
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
  size_t bytes = 0;
  void allocate(size_t elements, bool zero = false) {
    bytes = product({elements, sizeof(float)});
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes))
      throw std::bad_alloc();
    ptr.reset(static_cast<float *>(p));
    if (zero)
      clear();
  }
  void clear() { std::memset(ptr.get(), 0, bytes); }
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
  const BJ47KernelInfoV2 *hybrid = nullptr;
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
    const bool shard = b.tpRank >= 0 && (id == 4 || id == 5);
    const auto heads = shard ? b.bodyKvHeads : b.kvHeads;
    const auto infoBytes = b.hybrid ? sizeof(BJ47KernelInfoV2) : sizeof(*info);
    if (!info || info->struct_bytes != infoBytes || info->role != id ||
        info->abi_version != unsigned(e.abi) ||
        info->chunk != unsigned(c.prefillChunk) ||
        info->context != unsigned(c.context) ||
        info->width != unsigned(c.activationWidth) ||
        info->kv_heads != unsigned(heads) ||
        info->head_dim != unsigned(b.headDim) ||
        info->vocab_size != unsigned(c.vocabSize) ||
        hexDigest(info->layout_id) != e.layout ||
        hexDigest(info->shape_id) != (shard ? b.bodyShape : b.shape) ||
        info->aux_prefill_bytes || info->aux_decode_bytes ||
        !info->required_alignment || 64 % info->required_alignment ||
        (!b.hybrid &&
         info->rope_elements != (id == 2 ? unsigned(b.headDim) / 2 : 0)) ||
        (id == 1 && info->i8_elements))
      throw std::runtime_error("incompatible kernel descriptor: " + role);
    if (!b.hybrid)
      return;
    hybrid = reinterpret_cast<const BJ47KernelInfoV2 *>(info);
    const auto kv =
        product({size_t(c.context), size_t(heads), size_t(b.headDim)});
    const auto rope = hybrid->rotary_dim;
    const auto dtype = hybrid->embed_dtype;
    const bool validState =
        id == 4 ? hybrid->state_a_elements && hybrid->state_b_elements
        : (id == 5 || id == 6)
            ? hybrid->state_a_elements == kv && hybrid->state_b_elements == kv
            : !hybrid->state_a_elements && !hybrid->state_b_elements;
    const auto tableSize =
        product({size_t(c.vocabSize), size_t(c.activationWidth)});
    const bool validWeights =
        id == 1 ? (dtype == 1 && info->f32_elements == tableSize &&
                   !hybrid->f16_elements) ||
                      (dtype == 2 && hybrid->f16_elements == tableSize &&
                       !info->f32_elements)
                : dtype == 0 && !hybrid->f16_elements && info->f32_elements;
    if (!validState || !validWeights ||
        ((id == 5 || id == 6) ? !rope || rope > unsigned(b.headDim) ||
                                    rope % 2 || info->rope_elements != rope / 2
                              : rope || info->rope_elements))
      throw std::runtime_error("incompatible hybrid kernel descriptor: " +
                               role);
  }
};
struct Phase {
  decltype(&prepare_prefill) prepare;
  decltype(&embed_prefill) embed = nullptr;
  decltype(&embed_prefill_f16) embedF16 = nullptr;
  decltype(&lm_head_prefill) head = nullptr;
};
struct Layer {
  BJ47WeightsV1 weights;
  decltype(&layer_prefill) prefill;
  decltype(&layer_decode) decode;
  decltype(&layer_verify_2) verify{};
  decltype(&layer_tp_rows) tensor{};
  std::vector<Buffer> k, v, savedK, savedV;
};
class TemplateStage final : public MtpStage {
  StageConfig cfg;
  Library::Handle runtime{nullptr, dlclose};
  std::map<std::string, Library> templates;
  Library embed, head, mtp;
  Layer predictor{};
  decltype(&mtp_forward) predict{};
  decltype(&lm_head_normalize) normalize{};
  decltype(&lm_head_linear) project{};
  Buffer normalized, embeddings, predicted;
  decltype(&prepare_decode) ropeOne{};
  Phase pre{}, dec{};
  std::vector<std::unique_ptr<Mapping>> mappings;
  std::vector<Layer> layers;
  const float *rope = nullptr;
  const void *table = nullptr;
  BJ47WeightsV1 projection{};
  Buffer a, b, cosine, sine, logits;
  std::vector<int64_t> tokenStorage;
  size_t mapped = 0, stateBytes = 0;
  TensorHooks *tp;
  std::vector<Mapping *> bodyPacks;
  bool usedPrefill = false;
  bool verifyWeights;
  const int slots;
  int activeSlot = 0;
  uint64_t inputAlignment = 64;
  bool hybrid;

  const void *map(const WeightEntry &e, size_t count) {
    if (e.elements != count)
      throw std::runtime_error("DSO weight elements: " + e.path.string());
    mappings.push_back(std::make_unique<Mapping>(e));
    if (verifyWeights && tp)
      verifyFileDigest(e.path, e.digest, "TP weight");
    if (verifyWeights && !tp &&
        contentDigest(mappings.back()->addr, e.bytes) != e.digest)
      throw std::runtime_error("weight digest: " + e.path.string());
    mapped += e.bytes;
    return mappings.back()->addr;
  }
  const void *run(void *input, int pos, bool prefill, bool needLogits,
                  int validRows, int rows, bool returnHidden = false) {
    auto &p = prefill ? pre : dec;

    if (!input || validRows < 1 || validRows > rows || pos < 0 ||
        pos > cfg.context - rows)
      throw std::out_of_range("invalid stage input/position");
    auto checked = [&](int code, int layer, const char *operation) {
      if (code)
        throw std::runtime_error(
            std::string(operation) + (prefill ? " prefill" : " decode") +
            " layer=" + std::to_string(layer) + " pos=" + std::to_string(pos) +
            " rc=" + std::to_string(code));
    };
    BJ47FrameRowsV2 rowInfo{validRows};
    BJ47FrameV1 frame{pos, cosine.data(), sine.data(),
                      hybrid ? &rowInfo : nullptr};
    if (rows == 2 && p.prepare) {
      int stride =
          static_cast<int>(cosine.bytes / (cfg.prefillChunk * sizeof(float)));
      for (int i = 0; i < 2; ++i)
        checked(ropeOne(rope, pos + i, cosine.data() + i * stride,
                        sine.data() + i * stride, nullptr),
                cfg.startLayer, "prepare verify");
    } else if (p.prepare)
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
      int calls = rows == 2 ? 2 : 1;
      for (int i = 0; i < calls; ++i)
        checked(p.embedF16 ? p.embedF16(static_cast<const uint16_t *>(table),
                                        ids + i, out + i * cfg.activationWidth)
                           : p.embed(static_cast<const float *>(table), ids + i,
                                     out + i * cfg.activationWidth),
                0, "embed");
      in = out;
      out = b.data();
    } else if (reinterpret_cast<uintptr_t>(in) % inputAlignment) {
      // Transport-owned receive buffers only promise float alignment.
      // Use the spare aligned buffer as input to the first local layer.
      std::copy_n(in, size_t(rows) * cfg.activationWidth, b.data());
      in = b.data();
    }
    if (tp && tp->dispatch)
      tp->dispatch(in, activeSlot, pos, rows, validRows);
    if (!bodyPacks.empty() && usedPrefill != prefill) {
      // Release the other packed layout's resident pages at phase boundaries.
      for (auto pack : bodyPacks)
        if (madvise(pack->addr, pack->bytes, MADV_DONTNEED))
          throw std::runtime_error("discard cold packed layout");
      usedPrefill = prefill;
    }
    for (int i = cfg.startLayer; i < cfg.endLayer; ++i) {
      auto &l = layers[i - cfg.startLayer];
      auto kernel = prefill ? l.prefill : l.decode;
      BJ47VerifyRows snapshot{{2}, nullptr, nullptr};
      if (rows == 2) {
        kernel = l.verify;
        if (!l.savedK.empty()) {
          snapshot.conv_snapshot = l.savedK[activeSlot].data();
          snapshot.ssm_snapshot = l.savedV[activeSlot].data();
        }
        frame.aux = &snapshot;
      }
      if (l.tensor)
        checked(l.tensor(&l.weights, in, l.k[activeSlot].data(),
                         l.v[activeSlot].data(), &frame, out, rows, tp->reduce,
                         tp->context),
                i, "TP layer");
      else
        checked(kernel(&l.weights, in, l.k[activeSlot].data(),
                       l.v[activeSlot].data(), &frame, out),
                i, "layer");
      in = out;
      out = (out == a.data() ? b.data() : a.data());
    }
    if (!p.head)
      return in;
    if (returnHidden) {
      checked(normalize(&projection, in, normalized.data(), validRows),
              cfg.endLayer, "final norm");
      return normalized.data();
    }
    if (needLogits)
      checked(p.head(&projection,
                     in + size_t(validRows - 1) * cfg.activationWidth,
                     logits.data()),
              cfg.endLayer, "head");
    return logits.data();
  }

public:
  TemplateStage(const TemplateBundle &bundle, LayerRange range, bool verify,
                int slotCount, bool enableMtp, TensorHooks *hooks)
      : cfg(bundle.config), tp(hooks), verifyWeights(verify), slots(slotCount),
        hybrid(bundle.hybrid) {
    if ((bundle.tpRank >= 0) != bool(tp) || (tp && !tp->reduce))
      throw std::invalid_argument("TP bundle requires a tensor collective");
    const bool endpoints = !tp || bundle.tpRank == bundle.tpHost;
    if (slots < 1)
      throw std::invalid_argument("Route B requires at least one KV slot");
    auto [start, end] = range;
    if (start < 0 || start >= end || end > cfg.totalLayers)
      throw std::invalid_argument("invalid local layer range");
    cfg.startLayer = start;
    cfg.endLayer = end;
    if (!bundle.runtimeDigest.empty()) {
      verifyFileDigest(bundle.runtime, bundle.runtimeDigest, "runtime");
      // Load the verified SONAME before opening template dependencies, even
      // when the launcher supplied additional library search paths.
      runtime.reset(dlopen(bundle.runtime.c_str(), RTLD_LOCAL | RTLD_NOW));
      if (!runtime)
        throw std::runtime_error(bundle.runtime.string() + ": " + dlerror());
    }
    auto weights = [&](const std::string &owner, int id, const Library &lib) {
      BJ47WeightsV1 w{
          static_cast<const float *>(map(bundle.weights.at({owner, id, "f32"}),
                                         lib.info->f32_elements)),
          nullptr};
      if (lib.info->i8_elements || bundle.weights.count({owner, id, "q4"}))
        w.q4 = static_cast<const int8_t *>(
            map(bundle.weights.at({owner, id, "q4"}), lib.info->i8_elements));
      return w;
    };
    if (endpoints && start == 0) {
      embed.open(bundle, "embed", 1);
      if (embed.hybrid && embed.hybrid->embed_dtype == 2) {
        table = map(bundle.weights.at({"embed", -1, "f16"}),
                    embed.hybrid->f16_elements);
        pre.embedF16 =
            embed.symbol<decltype(pre.embedF16)>("embed_prefill_f16");
        dec.embedF16 = embed.symbol<decltype(dec.embedF16)>("embed_decode_f16");
      } else {
        table = map(bundle.weights.at({"embed", -1, "f32"}),
                    embed.info->f32_elements);
        pre.embed = embed.symbol<decltype(pre.embed)>("embed_prefill");
        dec.embed = embed.symbol<decltype(dec.embed)>("embed_decode");
      }
      tokenStorage.resize(size_t(cfg.prefillChunk) + 8);
    }
    if (endpoints && end == cfg.totalLayers) {
      head.open(bundle, "lm_head", 3);
      projection = weights("head", -1, head);
      pre.head = head.symbol<decltype(pre.head)>("lm_head_prefill");
      dec.head = head.symbol<decltype(dec.head)>("lm_head_decode");
      logits.allocate(size_t(cfg.vocabSize) * (enableMtp ? 2 : 1));
      if (enableMtp) {
        normalize = head.symbol<decltype(normalize)>("lm_head_normalize");
        project = head.symbol<decltype(project)>("lm_head_linear");
        mtp.open(bundle, "mtp_layer", 6);
        predictor.weights = weights("mtp", -1, mtp);
        predict = mtp.symbol<decltype(predict)>("mtp_forward");
        predictor.k.resize(slots);
        predictor.v.resize(slots);
        for (int i = 0; i < slots; ++i) {
          predictor.k[i].allocate(mtp.hybrid->state_a_elements, true);
          predictor.v[i].allocate(mtp.hybrid->state_b_elements, true);
          stateBytes += predictor.k[i].bytes + predictor.v[i].bytes;
        }
      }
    }
    size_t kv = product(
        {size_t(bundle.kvHeads), size_t(bundle.headDim), size_t(cfg.context)});
    size_t rotary = bundle.headDim;
    layers.resize(end - start);
    for (int i = start; i < end; ++i) {
      const auto &type = bundle.layerTypes[i];
      const auto role = bundle.hybrid ? type + "_layer" : type;
      auto [entry, inserted] = templates.try_emplace(role);
      auto &lib = entry->second;
      if (inserted) {
        lib.open(bundle, role, !hybrid ? 2 : type == "gdn" ? 4 : 5);
        if (i == start)
          inputAlignment = lib.info->required_alignment;
        if (lib.info->rope_elements) {
          pre.prepare = lib.symbol<decltype(pre.prepare)>("prepare_prefill");
          dec.prepare = lib.symbol<decltype(dec.prepare)>("prepare_decode");
          rope = static_cast<const float *>(map(
              bundle.weights.at({"rope", -1, "f32"}), lib.info->rope_elements));
          if (lib.hybrid)
            rotary = lib.hybrid->rotary_dim;
        }
      }
      auto &l = layers[i - start];
      l.weights = weights(bundle.hybrid ? type : "decoder", i, lib);
      if (tp) {
        l.tensor = lib.symbol<decltype(l.tensor)>("layer_tp_rows");
        if (l.weights.q4)
          bodyPacks.push_back(mappings.back().get());
      }
      l.prefill = lib.symbol<decltype(l.prefill)>("layer_prefill");
      l.decode = lib.symbol<decltype(l.decode)>("layer_decode");
      if (enableMtp)
        l.verify = lib.symbol<decltype(l.verify)>("layer_verify_2");
      const size_t stateA = lib.hybrid ? lib.hybrid->state_a_elements : kv;
      const size_t stateB = lib.hybrid ? lib.hybrid->state_b_elements : kv;
      if (enableMtp && type == "gdn") {
        l.savedK.resize(slots);
        l.savedV.resize(slots);
        for (int slot = 0; slot < slots; ++slot) {
          l.savedK[slot].allocate(stateA, true);
          l.savedV[slot].allocate(stateB, true);
          stateBytes += l.savedK[slot].bytes + l.savedV[slot].bytes;
        }
      }
      l.k.resize(slots);
      l.v.resize(slots);
      for (int slot = 0; slot < slots; ++slot) {
        l.k[slot].allocate(stateA, true);
        l.v[slot].allocate(stateB, true);
      }
      const auto aBytes = product({stateA, sizeof(float), size_t(slots)});
      const auto bBytes = product({stateB, sizeof(float), size_t(slots)});
      if (aBytes > SIZE_MAX - bBytes ||
          stateBytes > SIZE_MAX - (aBytes + bBytes))
        throw std::length_error("Route B state storage size overflow");
      stateBytes += aBytes + bBytes;
    }
    auto hidden =
        product({size_t(cfg.prefillChunk), size_t(cfg.activationWidth)});
    if (predict) {
      normalized.allocate(hidden);
      embeddings.allocate(hidden);
      predicted.allocate(hidden);
      if (!rope) {
        rope = static_cast<const float *>(map(
            bundle.weights.at({"rope", -1, "f32"}), mtp.info->rope_elements));
        rotary = mtp.hybrid->rotary_dim;
      }
      ropeOne = mtp.symbol<decltype(ropeOne)>("prepare_decode");
    } else
      ropeOne = dec.prepare;
    a.allocate(hidden);
    b.allocate(hidden);
    if (rope) {
      auto frame = product({size_t(cfg.prefillChunk), rotary});
      cosine.allocate(frame);
      sine.allocate(frame);
    }
    if (std::getenv("BUDDY_PP_TIMING") || std::getenv("BUDDY_PP_STORAGE"))
      std::fprintf(stderr,
                   "pp_storage start=%d end=%d mapped_bytes=%zu kv_bytes=%zu\n",
                   start, end, mapped, stateBytes);
  }
  const StageConfig &configuration() const override { return cfg; }
  int maxSlots() const override { return slots; }
  void checkSlot(int slot) const {
    if (slot < 0 || slot >= slots)
      throw std::out_of_range("Route B KV slot");
  }
  void setActiveSlot(int slot) override {
    checkSlot(slot);
    activeSlot = slot;
  }
  void clearSlot(int slot) override {
    checkSlot(slot);
    if (tp && tp->clear)
      tp->clear(slot);
    for (auto &layer : layers) {
      layer.k[slot].clear();
      layer.v[slot].clear();
      if (!layer.savedK.empty()) {
        layer.savedK[slot].clear();
        layer.savedV[slot].clear();
      }
    }
    if (predict) {
      predictor.k[slot].clear();
      predictor.v[slot].clear();
    }
  }
  const float *backbone(void *input, int pos, int rows, int valid) override {
    if (rows != 1 && rows != 2 && rows != cfg.prefillChunk)
      throw std::invalid_argument("invalid speculative frame capacity");
    return static_cast<const float *>(
        run(input, pos, rows == cfg.prefillChunk, false, valid, rows, true));
  }
  void rewindOne() override {
    if (tp && tp->rewind)
      tp->rewind(activeSlot);
    for (auto &l : layers)
      if (!l.savedK.empty()) {
        std::memcpy(l.k[activeSlot].data(), l.savedK[activeSlot].data(),
                    l.k[activeSlot].bytes);
        std::memcpy(l.v[activeSlot].data(), l.savedV[activeSlot].data(),
                    l.v[activeSlot].bytes);
      }
  }
  void gatherShifted(const int64_t *ids, int rows, void *packed) override {
    size_t bytes = size_t(cfg.activationWidth) * (pre.embedF16 ? 2 : 4);
    for (int i = 0; i < rows; ++i) {
      if (ids[i] < 0 || ids[i] >= cfg.vocabSize)
        throw std::out_of_range("MTP token");
      std::memcpy(static_cast<char *>(packed) + bytes * i,
                  static_cast<const char *>(table) + bytes * ids[i], bytes);
    }
  }
  const float *draftHidden(const void *packed, bool half, const float *hidden,
                           int pos, int rows, int valid) override {
    auto n = size_t(valid) * cfg.activationWidth;
    if (half) {
      auto src = static_cast<const uint16_t *>(packed);
      pipeline::decodeInto(src, n, embeddings.data(), n);
    } else
      std::memcpy(embeddings.data(), packed, n * sizeof(float));
    int stride = mtp.hybrid->rotary_dim;
    for (int i = 0; i < valid; ++i)
      if (ropeOne(rope, pos + i, cosine.data() + i * stride,
                  sine.data() + i * stride, nullptr))
        throw std::runtime_error("MTP rotary position");
    BJ47FrameRowsV2 aux{valid};
    BJ47FrameV1 frame{pos, cosine.data(), sine.data(), &aux};
    if (predict(&predictor.weights, embeddings.data(), hidden,
                predictor.k[activeSlot].data(), predictor.v[activeSlot].data(),
                &frame, predicted.data(), rows))
      throw std::runtime_error("MTP forward");
    return predicted.data();
  }
  const float *linear(const float *hidden, int rows) override {
    if (project(&projection, hidden, logits.data(), rows))
      throw std::runtime_error("MTP shared head");
    return logits.data();
  }
  const uint16_t *prefill(void *) override {
    throw std::logic_error("Route B requires f32");
  }
  const uint16_t *decode(void *, int) override {
    throw std::logic_error("Route B requires f32");
  }
  const void *prefillData(void *p) override { return prefillChunk(p, 0, true); }
  const void *prefillChunk(void *p, int pos, bool head) override {
    return prefillValidRows(p, pos, head, cfg.prefillChunk);
  }
  const void *prefillValidRows(void *p, int pos, bool head, int rows) override {
    return run(p, pos, true, head, rows, cfg.prefillChunk);
  }
  const void *decodeData(void *p, int pos) override {
    return run(p, pos, false, true, 1, 1);
  }
};
} // namespace
std::unique_ptr<MtpStage> makeTemplateStage(const TemplateBundle &b,
                                            LayerRange range, bool verify,
                                            int slots, bool enableMtp,
                                            TensorHooks *tp) {
  return std::make_unique<TemplateStage>(b, range, verify, slots, enableMtp,
                                         tp);
}
} // namespace buddy::runtime
