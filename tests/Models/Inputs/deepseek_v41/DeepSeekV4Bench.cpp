// Single-token, real-shape microbenchmarks. Setup is outside timed sections.
#include "buddy/runtime/DeepSeekV4Attention.h"
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

using namespace buddy::v4;
using Clock = std::chrono::steady_clock;
static std::mt19937 rng(4106);
static volatile float checksum;
static bool smoke = false;
static int fp8Block = 128;
Vec values(size_t n, float scale = .02f) {
  Vec a(n);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto &v : a)
    v = dist(rng);
  return a;
}
template <class F>
void measure(const std::string &name, F run, int reps = 5,
             size_t weightBytes = 0) {
  if (smoke)
    reps = 1;
  else
    checksum = run();
  std::vector<double> elapsed;
  std::cout << "{\"name\":\"" << name << "\",\"ms\":[";
  for (int i = 0; i < reps; i++) {
    auto start = Clock::now();
    checksum = run();
    double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (!std::isfinite(checksum))
      throw std::runtime_error("nonfinite checksum");
    elapsed.push_back(ms);
    std::cout << (i ? "," : "") << ms;
  }
  if (weightBytes) {
    std::cout << "],\"weight_bytes\":" << weightBytes
              << ",\"effective_GB_s\":[";
    for (size_t i = 0; i < elapsed.size(); ++i)
      std::cout << (i ? "," : "") << weightBytes / (elapsed[i] * 1e6);
  }
  std::cout << "],\"checksum\":" << checksum;
  if (name.rfind("csa_", 0) == 0 || name.rfind("engram_", 0) == 0 ||
      name == "moe_native")
    std::cout << ",\"fp8_weight_block\":" << fp8Block;
  std::cout << "}" << std::endl;
}
struct W4 {
  int k, n;
  std::vector<uint8_t> bytes;
  W4(int k, int n) : k(k), n(n), bytes(size_t(k / 32) * n * 18) {
    uint32_t state = rng();
    for (size_t i = 0; i < bytes.size(); i += 2304) {
      for (int j = 0; j < 2048; j += 4) {
        state = 1664525 * state + 1013904223;
        std::memcpy(bytes.data() + i + j, &state, 4);
      }
      for (int j = 2048; j < 2304; j += 2) {
        _Float16 sc = static_cast<_Float16>(.002f);
        std::memcpy(bytes.data() + i + j, &sc, 2);
      }
    }
  }
  Vec apply(const Vec &x) const {
    Vec y(n);
    w4linear(x.data(), bytes.data(), k, n, y.data());
    return y;
  }
};
struct FP4 {
  int k, n;
  std::vector<uint8_t> data, scale;
  FP4(int k, int n)
      : k(k), n(n), data(size_t(k / 2) * n), scale(size_t(k / 32) * n) {
    std::uniform_int_distribution<unsigned> byte(0, 255);
    for (auto &b : data)
      b = byte(rng);
    for (auto &b : scale)
      b = 117 + (byte(rng) % 5);
  }
  Vec apply(const Vec &x) const {
    Vec y(n);
    fp4MatVec(x.data(), data.data(), scale.data(), k, n, y.data());
    return y;
  }
};
void benchFP4() {
  for (auto shape :
       {std::pair{5120, 2304}, std::pair{2304, 5120}, std::pair{5120, 1280}}) {
    auto [k, n] = shape;
    FP4 w(k, n);
    auto x = values(k);
    measure(
        "fp4_k" + std::to_string(k) + "_n" + std::to_string(n),
        [&] { return w.apply(x).back(); }, 5, w.data.size() + w.scale.size());
  }
}
struct FP8 {
  int k, n, block;
  std::vector<uint8_t> data, scale;
  FP8(int k, int n)
      : k(k), n(n), block(fp8Block), data(size_t(k) * n),
        scale(size_t((n + block - 1) / block) * ((k + block - 1) / block),
              119) {
    uint32_t state = rng();
    for (size_t i = 0; i < data.size(); i += 4) {
      state = 1664525 * state + 1013904223;
      auto v = state & 0xbfbfbfbfU;
      std::memcpy(data.data() + i, &v, 4);
    }
  }
  Vec apply(Vec x) const {
    quantize(x.data(), k, 32, Format::E4);
    Vec out(n);
    fp8MatVec(x.data(), data.data(), scale.data(), k, n, block, out.data());
    for (float &v : out)
      v = bf16(v);
    return out;
  }
};
void benchHC(int rows) {
  int h = 5120;
  auto x = values(rows * 4 * h), fn = values(24 * 4 * h), base = values(24),
       scale = values(3);
  Vec y(rows * h), out(rows * 4 * h);
  float pre[] = {.25f, .15f, .4f, .2f};
  measure("mhc_" + std::to_string(rows), [&] {
#pragma omp parallel for if (rows > 1)
    for (int i = 0; i < rows; i++) {
      auto src = x.data() + i * 4 * h;
      auto z = mixes(src, h, fn.data(), scale.data(), base.data(), rows == 1);
      hcPre(src, pre, h, y.data() + i * h);
      hcPost(y.data() + i * h, src, z, h, out.data() + i * 4 * h);
    }
    return out.back();
  });
}
void benchMoE() {
  constexpr int h = 5120, d = 2304;
  auto router = values(h * 384), bias = values(384), x = values(h);
  Vec scores(384), coef;
  linear(x.data(), router.data(), h, 384, scores.data());
  auto active = route(scores, bias.data(), 6, 1, 1.5f, coef);
  {
    std::vector<W4> mats;
    for (int i = 0; i < 7; i++) {
      mats.emplace_back(h, d);
      mats.emplace_back(h, d);
      mats.emplace_back(d, h);
    }
    measure("moe", [&] {
      linear(x.data(), router.data(), h, 384, scores.data());
      auto project = [&](int id, int branch, const Vec &v) {
        int slot = id == 384 ? 6
                             : int(std::find(active.begin(), active.end(), id) -
                                   active.begin());
        return mats[3 * slot + (branch == 1   ? 0
                                : branch == 3 ? 1
                                              : 2)]
            .apply(v);
      };
      auto out = moeV4(x, scores, bias.data(), 6, 1, 1.5f, 10, project,
                       [](float v) { return v; });
      return out.back();
    });
  }
  std::vector<FP4> experts;
  for (int i = 0; i < 6; ++i) {
    experts.emplace_back(h, d);
    experts.emplace_back(h, d);
    experts.emplace_back(d, h);
  }
  std::array<FP8, 3> shared{FP8(h, d), FP8(h, d), FP8(d, h)};
  measure("moe_native", [&] {
    linear(x.data(), router.data(), h, 384, scores.data());
    auto project = [&](int id, int branch, const Vec &v) {
      int b = branch == 1 ? 0 : branch == 3 ? 1 : 2;
      if (id == 384)
        return shared[b].apply(v);
      int slot = std::find(active.begin(), active.end(), id) - active.begin();
      Vec act = v;
      quantize(act.data(), act.size(), 32, Format::E4);
      Vec y = experts[3 * slot + b].apply(act);
      for (float &f : y)
        f = bf16(f);
      return y;
    };
    return moeV4(x, scores, bias.data(), 6, 1, 1.5f, 10, project, bf16).back();
  });
}
struct BFMatrix {
  int k, n;
  std::vector<uint16_t> data;
  BFMatrix(int k, int n) : k(k), n(n), data(size_t(k) * n) {
    uint32_t state = rng();
    for (auto &v : data) {
      state = 1664525 * state + 1013904223;
      v = 0x3b00 | ((state >> 16) & 0x81ff);
    }
  }
  Vec group(const Vec &x, int groups = 1, bool rounded = true) const {
    Vec y(n);
#pragma omp parallel for
    for (int j = 0; j < n; j++) {
      auto a = x.data() + (j / (n / groups)) * k;
      float sum = 0;
      auto p = data.data() + size_t(j) * k;
#if defined(__riscv_v_intrinsic)
      size_t width = __riscv_vsetvl_e32m4(k);
      auto acc = __riscv_vfmv_v_f_f32m4(0, width);
      for (int i = 0; i < k;) {
        size_t vl = std::min<size_t>(width, k - i);
        auto bits =
            __riscv_vwcvtu_x_x_v_u32m4(__riscv_vle16_v_u16m2(p + i, vl), vl);
        auto w = __riscv_vreinterpret_v_u32m4_f32m4(
            __riscv_vsll_vx_u32m4(bits, 16, vl));
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, w,
                                         __riscv_vle32_v_f32m4(a + i, vl), vl);
        i += vl;
      }
      sum = reduceSum(acc, width);
#else
      for (int i = 0; i < k; i++) {
        uint32_t bits = uint32_t(p[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        sum += a[i] * f;
      }
#endif
      y[j] = rounded ? bf16(sum) : sum;
    }
    return y;
  }
};
void benchCSA(const std::string &mode) {
  constexpr int h = 5120, d = 512, heads = 64, rank = 1280, rd = 64, ctx = 4096;
  bool swa = mode == "swa", full = mode == "full1" || mode == "full2",
       index = full || mode == "reindex";
  int ratio = mode == "full2" ? 2 : 1, len = ctx / ratio;
  FP8 qa(h, rank), qb(rank, heads * d), wk(h, d), ob(8192, h);
  BFMatrix oa(4096, 8192);
  auto x = values(h), ring = values(128 * d), cache = values(len * d),
       idxk = values(len * 128);
  auto sink = values(heads);
  Vec ones(1280, 1), freq(rd), firstFreq(rd), latent, tail, tailScore;
  const float base = swa ? 10000.f : 160000.f;
  const float pi = std::acos(-1.f);
  auto corrected = [&](int rotations) {
    return rd * std::log(65536.f / (rotations * 2 * pi)) / (2 * std::log(base));
  };
  float low = std::max(0.f, std::floor(corrected(32))),
        high = std::min(float(rd - 1), std::ceil(corrected(1)));
  for (int j = 0; j < rd; j += 2) {
    float f = 1 / std::pow(base, float(j) / rd);
    if (!swa) {
      float ramp =
          std::clamp((j / 2 - low) / std::max(high - low, .001f), 0.f, 1.f);
      f = f / 16 * ramp + f * (1 - ramp);
    }
    for (auto pair : {std::make_pair(&freq, ctx - 1),
                      std::make_pair(&firstFreq, ctx - ratio)}) {
      (*pair.first)[j] = std::cos(pair.second * f);
      (*pair.first)[j + 1] = std::sin(pair.second * f);
    }
  }
  std::optional<BFMatrix> compress, idxwk, iw;
  std::optional<FP8> iq;
  Vec cw, cg;
  if (full) {
    idxwk.emplace(d, 128);
    if (ratio == 1)
      compress.emplace(h, d);
    else {
      cw = values(h * d);
      cg = values(h * d);
      tail = values(d);
      tailScore = values(d);
    }
  }
  if (index) {
    iq.emplace(rank, 4096);
    iw.emplace(h, 32);
  }
  Ids idx(512);
  for (int i = 0; i < 512; i++)
    idx[i] = 128 + i * (len / 512);
  std::vector<uint8_t> candidate(len, 1);
  // At context 4096, 2048 candidate blocks of 8 cover every visible key.
  measure(
      "csa_" + mode,
      [&] {
        auto qr = qa.apply(x);
        norm(qr, ones.data());
        auto q = qb.apply(qr);
        for (int j = 0; j < heads; j++)
          rotate(q.data() + j * d + d - rd, freq.data(), rd);
        auto kv = wk.apply(x);
        norm(kv, ones.data());
        rotate(kv.data() + d - rd, freq.data(), rd);
        quantize(kv.data(), d, 32, Format::E4);
        std::copy(kv.begin(), kv.end(), ring.end() - d);
        if (full) {
          if (ratio == 1)
            latent = compress->group(x);
          else {
            latent.resize(d);
            Vec score(d);
            linear(x.data(), cw.data(), h, d, latent.data());
            linear(x.data(), cg.data(), h, d, score.data());
            for (int j = 0; j < d; j++) {
              float m = std::max(tailScore[j], score[j]),
                    a = std::exp(tailScore[j] - m), b = std::exp(score[j] - m);
              latent[j] =
                  bf16(tail[j] * (a / (a + b)) + latent[j] * (b / (a + b)));
            }
          }
          norm(latent, ones.data());
          auto k = idxwk->group(latent);
          norm(k, ones.data());
          rotate(k.data() + 128 - rd, firstFreq.data(), rd);
          quantize(k.data(), 128, 32, Format::E2);
          std::copy(k.begin(), k.end(), idxk.end() - 128);
          rotate(latent.data() + d - rd, firstFreq.data(), rd);
          quantize(latent.data(), d, 16, Format::E2ScaleE4);
          std::copy(latent.begin(), latent.end(), cache.end() - d);
        }
        if (index) {
          auto query = iq->apply(qr), weights = iw->group(x);
          for (int j = 0; j < 32; j++) {
            rotate(query.data() + 128 * j + 128 - rd, freq.data(), rd);
            quantize(query.data() + 128 * j, 128, 32, Format::E2);
            weights[j] = bf16(weights[j] * (1.f / 64));
          }
          Vec scores(len);
#pragma omp parallel for
          for (int i = 0; i < len; i++) {
            float s = 0;
            for (int j = 0; j < 32; j++)
              s += bf16(std::max(0.f, bf16(dot(query.data() + j * 128,
                                               idxk.data() + i * 128, 128))) *
                        weights[j]);
            scores[i] = bf16(s);
          }
          if (mode == "full1")
            candidate = candidates(scores, len, 2048, 8);
          idx = selectIndex(scores, len, 512, 128,
                            mode == "reindex" ? &candidate : nullptr);
        }
        std::vector<const float *> keys;
        for (int i = 0; i < 128; i++)
          keys.push_back(ring.data() + i * d);
        if (!swa)
          for (int j : idx)
            keys.push_back(cache.data() + (j - 128) * d);
        Vec out(heads * d);
        sparse(q.data(), keys, heads, d, sink.data(), out.data());
        for (int j = 0; j < heads; j++)
          rotate(out.data() + j * d + d - rd, freq.data(), rd, true);
        quantize(out.data(), out.size(), 32, Format::E4);
        return ob.apply(oa.group(out, 8)).back();
      },
      3);
}
void benchEngram(const char *file) {
  constexpr size_t size = 1ULL << 30;
  constexpr int samples = 6, rows = 24;
  int fd = open(file, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0 || ftruncate(fd, size))
    throw std::runtime_error("exclusive sparse file creation");
  struct Cleanup {
    const char *path;
    ~Cleanup() { unlink(path); }
  } cleanup{file};
  alignas(4096) std::array<uint8_t, 4096> page;
  std::array<std::array<off_t, rows>, samples> offsets;
  for (auto &batch : offsets)
    for (auto &off : batch) {
      off = off_t(rng() % (size / 4096)) * 4096;
      for (auto &b : page)
        b = uint8_t(rng() & 0xbf);
      std::fill_n(page.begin() + 256, 8, 119);
      if (pwrite(fd, page.data(), 4096, off) != 4096)
        throw std::runtime_error("populate page");
    }
  if (fdatasync(fd))
    throw std::runtime_error("sync pages");
  auto map = static_cast<const uint8_t *>(
      mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
  int direct = open(file, O_RDONLY | O_DIRECT);
  if (map == MAP_FAILED || direct < 0)
    throw std::runtime_error("direct/mmap");
  FP8 wkv(6144, 25600);
  auto x = values(4 * 5120), qw = values(4 * 5120), kw = values(4 * 5120);
  for (bool cold : {true, false}) {
    int it = 0;
    // Prefault every measured warm row, including rows for subsequent trials.
    if (!cold)
      for (auto &batch : offsets)
        for (auto off : batch)
          checksum = map[off];
    Vec emb(rows * 256);
    auto lookup = [&] {
      for (int j = 0; j < rows; j++) {
        off_t off = offsets[it % samples][j];
        const uint8_t *p = map + off;
        if (cold) {
          if (pread(direct, page.data(), 4096, off) != 4096)
            throw std::runtime_error("direct read");
          p = page.data();
        }
        for (int k = 0; k < 256; k++)
          emb[j * 256 + k] = bf16(e4m3(p[k]) * ue8m0(p[256 + k / 32]));
      }
      it++;
      return emb.back();
    };
    measure(cold ? "lookup_direct" : "lookup_warm", lookup);
    measure(cold ? "engram_direct" : "engram_warm", [&] {
      lookup();
      auto y = wkv.apply(emb), res = x;
      engramWrite(res.data(), y.data(), qw.data(), kw.data(), 5120);
      return res.back();
    });
  }
  munmap(const_cast<uint8_t *>(map), size);
  close(direct);
  close(fd);
}
int main(int argc, char **argv) {
  if (argc < 2)
    return 2;
  while (argc > 2) {
    std::string option = argv[argc - 1];
    if (option == "--smoke")
      smoke = true;
    else if (option == "--fp8-block=32")
      fp8Block = 32;
    else if (option == "--fp8-block=128")
      fp8Block = 128;
    else
      break;
    --argc;
  }
  std::string m = argv[1];
  if (argc != (m == "engram" ? 3 : 2))
    return 2;
  if (m == "hc") {
    benchHC(1);
    benchHC(64);
  } else if (m == "head") {
    for (int h : {4096, 5120}) {
      auto x = values(h), w = values(h * 4096);
      Vec y(4096);
      measure("head4096_" + std::to_string(h), [&] {
        linear(x.data(), w.data(), h, 4096, y.data());
        return y.back();
      });
      BFMatrix b(h, 4096);
      measure("bf16head4096_" + std::to_string(h),
              [&] { return b.group(x, 1, false).back(); });
    }
  } else if (m == "moe")
    benchMoE();
  else if (m == "fp4")
    benchFP4();
  else if (m == "csa")
    benchCSA("full2");
  else if (m == "engram" && argc == 3)
    benchEngram(argv[2]);
  else if (m == "swa" || m == "reindex" || m == "reuse" || m == "full1" ||
           m == "full2")
    benchCSA(m);
  else
    return 2;
}
