#pragma once
// Independent, batch-one text backbone for differential tests.
// Input directory is produced by export_v41_fixture.py.
#include "buddy/runtime/DeepSeekV4Attention.h"
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace buddy::v4::tiny {
using std::string;
using Rows = std::vector<Vec>;

struct Tensor {
  Ids shape;
  string type;
  size_t bytes = 0;
  void *addr = MAP_FAILED;
  Tensor(const string &file, std::istream &is) {
    int rank;
    if (!(is >> type >> rank) || rank < 1 || rank > 6)
      throw std::runtime_error("bad tensor rank");
    size_t numel = 1;
    for (int j = 0; j < rank; j++) {
      int n;
      if (!(is >> n) || n <= 0 || numel > SIZE_MAX / 8 / size_t(n))
        throw std::runtime_error("bad dimension");
      shape.push_back(n);
      numel *= n;
    }
    if (type == "w4g32") {
      if (shape.size() != 2 || shape[0] % 128 || shape[1] % 32)
        throw std::runtime_error("invalid K3 tile shape");
      bytes = numel / 32 * 18;
    } else
      bytes = numel * (type == "int" ? 8 : type == "float" ? 4 : 1);
    int fd = open(file.c_str(), O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) || size_t(st.st_size) != bytes) {
      if (fd >= 0)
        close(fd);
      throw std::runtime_error(file);
    }
    addr = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED)
      throw std::runtime_error("mmap " + file);
  }
  ~Tensor() {
    if (addr != MAP_FAILED)
      munmap(addr, bytes);
  }
  const float *f() const { return static_cast<const float *>(addr); }
  const uint8_t *u() const { return static_cast<const uint8_t *>(addr); }
  const int64_t *id() const { return static_cast<const int64_t *>(addr); }
};
struct Store {
  std::map<string, std::unique_ptr<Tensor>> tensors;
  Store(const string &dir, const string &index) {
    std::ifstream in(dir + "/" + index + ".txt");
    if (!in)
      throw std::runtime_error("missing " + index);
    string line;
    while (std::getline(in, line)) {
      std::istringstream s(line);
      string name, file;
      s >> name >> file;
      tensors.emplace(name, std::make_unique<Tensor>(dir + "/" + file, s));
    }
  }
  const Tensor &operator[](const string &k) const { return *tensors.at(k); }
};
inline void roundBF(Vec &x) {
  for (float &v : x)
    v = bf16(v);
}
struct Weights : Store {
  using Store::Store;
  Vec proj(const string &key, Vec x, bool rounded = true) const {
    auto &w = (*this)[key + ".weight"];
    int n = w.shape[0], k = x.size();
    if (w.shape.size() != 2 || w.shape[1] * (w.type == "fp4" ? 2 : 1) != k)
      throw std::invalid_argument("bad projection " + key);
    Vec out(n);
    if (w.type == "float")
      linear(x.data(), w.f(), k, n, out.data());
    else if (w.type == "w4g32")
      w4linear(x.data(), w.u(), k, n, out.data());
    else {
      quantize(x.data(), k, 32, Format::E4);
      auto &s = (*this)[key + ".scale"];
      if (w.type == "fp4")
        fp4MatVec(x.data(), w.u(), s.u(), k, n, out.data());
      else
        fp8MatVec(x.data(), w.u(), s.u(), k, n, 32, out.data());
    }
    if (rounded)
      roundBF(out);
    return out;
  }
  // Map both row-major arrays read-only. A released table is larger than
  // 2^32 bytes even though its row IDs still fit in int32.
  Vec gatherEngram(const string &prefix, const Ids &ids) const {
    auto &data = (*this)[prefix + ".weight"],
         &scales = (*this)[prefix + ".scale"];
    if (data.type != "fp8" || scales.type != "scale" ||
        data.shape.size() != 2 || data.shape[1] % 32 ||
        scales.shape != Ids{data.shape[0], data.shape[1] / 32})
      throw std::runtime_error("invalid Engram embedding table");
    size_t width = data.shape[1], groups = width / 32;
    Vec result(ids.size() * width);
    for (size_t j = 0; j < ids.size(); j++) {
      if (ids[j] < 0 || ids[j] >= data.shape[0])
        continue;
      size_t row = static_cast<size_t>(ids[j]);
      auto dst = result.data() + j * width;
      decodeFP8(data.u() + row * width, scales.u() + row * groups, width, 32,
                dst);
      for (size_t k = 0; k < width; k++)
        dst[k] = bf16(dst[k]);
    }
    return result;
  }
  void normalize(const string &key, Vec &x) const {
    norm(x, (*this)[key + ".weight"].f());
  }
};

struct Cache {
  Rows window, kv, keys;
  Vec tail, score;
};
struct Shared {
  Rows kv, keys;
  std::vector<Ids> indices;
  std::vector<std::vector<uint8_t>> mask;
};
struct Runner {
  const Weights &w;
  std::vector<int64_t> history;
  std::array<Cache, 6> cache;
  using Observer =
      std::function<void(const string &, const string &, const Rows &)>;
  Observer observe;
  explicit Runner(const Weights &weights, Observer callback = {})
      : w(weights), observe(std::move(callback)) {}
  void check(const string &prefix, const string &name, const Rows &rows) {
    if (observe)
      observe(prefix, name, rows);
  }
  void engram(int l, int pos, const string &p, Vec &x) {
    int layer = l == 1 ? 0 : 1;
    auto &primes = w["hash.primes"];
    int heads = primes.shape.at(2), order = primes.shape.at(1) + 1;
    size_t cols = size_t(order - 1) * heads;
    auto ids = hashRows(history.data(), pos, w["hash.token_map"].id()[2], order,
                        heads, w["hash.multipliers"].id() + layer * order,
                        primes.id() + layer * cols,
                        w["hash.offsets"].id() + layer * cols);
    auto emb = w.gatherEngram(p + "embed", ids);
    auto kv = w.proj(p + "wkv", emb);
    engramWrite(x.data(), kv.data(), w[p + "q_weight"].f(),
                w[p + "k_weight"].f(), 128);
    roundBF(x);
  }
  Vec moe(const string &p, const Vec &x) {
    auto scores = w.proj(p + "gate", x, false);
    auto project = [&](int id, int branch, const Vec &input) {
      string expert =
          id == 16 ? "shared_experts" : "experts." + std::to_string(id);
      return w.proj(p + expert + ".w" + std::to_string(branch), input);
    };
    return moeV4(x, std::move(scores), w[p + "gate.bias"].f(), 4, 1, 1.5f, 10,
                 project, bf16);
  }
  Rows attention(int l, int start, const string &p, const Rows &x,
                 Shared &shared) {
    int n = x.size(), ratio = l == 0 ? 0 : l <= 2 ? 2 : 1;
    bool owns = l == 1 || l == 3, index = owns || l == 5;
    auto &c = cache[l];
    const float *freq = w["freq." + std::to_string(l)].f();
    Rows queries, low, window;
    if (c.window.empty())
      c.window.resize(16, Vec(64));
    for (int i = 0; i < n; i++) {
      int pos = start + i;
      auto qr = w.proj(p + "wq_a", x[i]);
      w.normalize(p + "q_norm", qr);
      auto q = w.proj(p + "wq_b", qr);
      for (int h = 0; h < 8; h++)
        rotate(q.data() + 64 * h + 48, freq + 16 * pos, 16);
      queries.push_back(q);
      low.push_back(qr);
      auto v = w.proj(p + "wkv", x[i]);
      w.normalize(p + "kv_norm", v);
      rotate(v.data() + 48, freq + 16 * pos, 16);
      quantize(v.data(), 64, 32, Format::E4);
      window.push_back(v);
      c.window[pos % 16] = v;
      if (!owns)
        continue;
      auto cp = p + "compressor.";
      auto latent = w.proj(cp + "wkv", x[i], ratio == 1);
      if (ratio == 2) {
        auto score = w.proj(cp + "wgate", x[i], false);
        if (pos % 2 == 0) {
          c.tail = latent;
          c.score = score;
          continue;
        }
        for (int j = 0; j < 64; j++) {
          float mx = std::max(c.score[j], score[j]);
          float a = std::exp(c.score[j] - mx), b = std::exp(score[j] - mx),
                den = a + b;
          latent[j] = bf16(c.tail[j] * (a / den) + latent[j] * (b / den));
        }
      }
      w.normalize(cp + "norm", latent);
      auto key = w.proj(p + "indexer.wk", latent);
      w.normalize(p + "indexer.k_norm", key);
      auto rot = freq + (pos + 1 - ratio) * 16;
      rotate(key.data() + 16, rot, 16);
      quantize(key.data(), 32, 32, Format::E2);
      c.keys.push_back(key);
      rotate(latent.data() + 48, rot, 16);
      quantize(latent.data(), 64, 16, Format::E2ScaleE4);
      c.kv.push_back(latent);
    }
    if (owns) {
      shared.kv = c.kv;
      shared.keys = c.keys;
    }
    if (index)
      shared.indices.resize(n);
    if (l == 3)
      shared.mask.resize(n);
    if (start)
      window = c.window;
    int offset = window.size();
    Rows outputs;
    for (int i = 0; i < n; i++) {
      int pos = start + i;
      if (index) {
        auto iq = w.proj(p + "indexer.wq_b", low[i]);
        auto iw = w.proj(p + "indexer.weights_proj", x[i]);
        for (int h = 0; h < 4; h++) {
          rotate(iq.data() + h * 32 + 16, freq + pos * 16, 16);
          quantize(iq.data() + h * 32, 32, 32, Format::E2);
          iw[h] = bf16(iw[h] * (1 / std::sqrt(32.f) * .5f));
        }
        int len = (pos + 1) / ratio;
        Vec score(shared.keys.size(), -INFINITY);
        for (int j = 0; j < len; j++) {
          float val = 0;
          for (int h = 0; h < 4; h++)
            val += bf16(std::max(0.f, bf16(dot(iq.data() + h * 32,
                                               shared.keys[j].data(), 32))) *
                        iw[h]);
          score[j] = bf16(val);
        }
        if (l == 3)
          shared.mask[i] = candidates(score, len, 4, 4);
        shared.indices[i] = selectIndex(score, len, 8, offset,
                                        l == 5 ? &shared.mask[i] : nullptr);
      }
      std::vector<const float *> keys;
      for (int j = 0; j < std::min(16, offset); j++) {
        int slot = start ? (pos + 1 + j) % 16 : std::max(0, pos - 15) + j;
        keys.push_back(slot > pos ? nullptr : window[slot].data());
      }
      if (ratio)
        for (int id : shared.indices[i])
          keys.push_back(id < 0 ? nullptr : shared.kv[id - offset].data());
      Vec o(512), grouped(128);
      sparse(queries[i].data(), keys, 8, 64, w[p + "attn_sink"].f(), o.data());
      for (int h = 0; h < 8; h++)
        rotate(o.data() + 64 * h + 48, freq + 16 * pos, 16, true);
      quantize(o.data(), 512, 32, Format::E4);
      auto wo = w[p + "wo_a.weight"].f();
      for (int g = 0; g < 4; g++)
        linear(o.data() + 128 * g, wo + 4096 * g, 128, 32,
               grouped.data() + 32 * g);
      roundBF(grouped);
      outputs.push_back(w.proj(p + "wo_b", grouped));
    }
    return outputs;
  }
  Ids run(Ids ids, int steps, const Ids &cuts = {}) {
    int prompt = ids.size();
    if (prompt < 1 || steps < 0 || prompt + steps > 64)
      throw std::invalid_argument("tiny context exceeds 64 tokens");
    for (int id : ids)
      if (id < 0 || id >= 128)
        throw std::invalid_argument("token outside tiny vocabulary");
    history.clear();
    cache = {};
    std::map<int, Shared> received;
    Ids tokens;
    for (int step = 0; step <= steps; step++) {
      int start = step ? prompt + step - 1 : 0, n = ids.size();
      Rows h(n, Vec(512)), pre(n, Vec{1, 0, 0, 0});
      Shared shared;
      for (int i = 0; i < n; i++) {
        history.push_back(w["hash.token_map"].id()[ids[i]]);
        for (int c = 0; c < 4; c++)
          std::copy_n(w["embed.weight"].f() + 128 * ids[i], 128,
                      h[i].data() + c * 128);
      }
      for (int l = 0; l < 6; l++) {
        string p = "layers." + std::to_string(l) + ".";
        if (std::find(cuts.begin(), cuts.end(), l) != cuts.end()) {
          // Materialize stage handoffs in independent storage.
          Rows wireH = h, wirePre = pre;
          h.swap(wireH);
          pre.swap(wirePre);
          Shared incoming;
          if (l != 1 && l != 3) {
            // A consumer retains its source prefix across steps. Only append
            // completed rows; a ratio-two partial group sends no cache row.
            auto &mirror = received[l];
            mirror.kv.insert(mirror.kv.end(),
                             shared.kv.begin() + mirror.kv.size(),
                             shared.kv.end());
            mirror.keys.insert(mirror.keys.end(),
                               shared.keys.begin() + mirror.keys.size(),
                               shared.keys.end());
            mirror.indices = shared.indices;
            mirror.mask = shared.mask;
            incoming = mirror;
          }
          shared = std::move(incoming);
        }
        if (l == 1 || l == 4)
          for (int i = 0; i < n; i++)
            engram(l, start + i, p + "engram.", h[i]);
        for (string sub : {"attn", "ffn"}) {
          Rows x(n, Vec(128));
          std::vector<Mix> z(n);
          for (int i = 0; i < n; i++) {
            string hc = p + "hc_" + sub + "_";
            z[i] = mixes(h[i].data(), 128, w[hc + "fn"].f(),
                         w[hc + "scale"].f(), w[hc + "base"].f(), n == 1);
            hcPre(h[i].data(), pre[i].data(), 128, x[i].data());
            roundBF(x[i]);
            w.normalize(p + sub + "_norm", x[i]);
          }
          if (sub == "attn")
            x = attention(l, start, p + "attn.", x, shared);
          else
            for (auto &v : x)
              v = moe(p + "ffn.", v);
          for (int i = 0; i < n; i++) {
            Vec residual(512);
            hcPost(x[i].data(), h[i].data(), z[i], 128, residual.data());
            roundBF(residual);
            h[i] = std::move(residual);
            pre[i].assign(z[i].pre, z[i].pre + 4);
          }
        }
        if (!observe)
          continue;
        string prefix = std::to_string(start) + "/" + std::to_string(l) + "/";
        check(prefix, "h", h);
        check(prefix, "pre_mix", pre);
        if (l) {
          check(prefix, "compress_kv", shared.kv);
          check(prefix, "index_k", shared.keys);
          Rows idx;
          for (auto &v : shared.indices)
            idx.emplace_back(v.begin(), v.end());
          check(prefix, "topk_idxs", idx);
          if (l >= 3) {
            Rows mask;
            for (auto &v : shared.mask)
              mask.emplace_back(v.begin(), v.end());
            check(prefix, "candidates", mask);
          }
        }
      }
      Vec y(128);
      hcPre(h.back().data(), pre.back().data(), 128, y.data());
      roundBF(y);
      w.normalize("norm", y);
      auto logits = w.proj("head", y, false);
      check(std::to_string(start) + "/", "logits", Rows{logits});
      int token =
          std::max_element(logits.begin(), logits.end()) - logits.begin();
      ids = {token};
      tokens.push_back(token);
    }
    return tokens;
  }
};

} // namespace buddy::v4::tiny
