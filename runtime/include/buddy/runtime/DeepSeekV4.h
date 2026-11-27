// Text-inference primitives. All matrix weights use [output, input] order
// except the K3 w4g32 tile stream (see k3_w4.pack_tiles).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

namespace buddy::v4 {
using Vec = std::vector<float>;
using Ids = std::vector<int>;
inline float bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffU) > 0x7f800000U)
    return x;
  u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000U;
  std::memcpy(&x, &u, 4);
  return x;
}
inline float e4m3(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = (b & 127) == 127
                ? NAN
                : std::ldexp(float(e ? 8 + m : m), std::max(e, 1) - 10);
  return b & 128 ? -v : v;
}
inline float ue8m0(uint8_t b) {
  uint32_t u = b == 0     ? 0x00400000u
               : b == 255 ? 0x7fc00000u
                          : uint32_t(b) << 23;
  float value;
  std::memcpy(&value, &u, sizeof(value));
  return value;
}
constexpr float E2M1[] = {0,    .5,  1,  1.5,  2,  3,  4,  6,
                          -0.f, -.5, -1, -1.5, -2, -3, -4, -6};
inline float scaledE2M1(uint8_t code, uint8_t scale) {
  // Build binary32 directly, including signed zero, subnormals and overflow.
  static constexpr int exponent[] = {0, -1, 0, 0, 1, 1, 2, 2};
  unsigned mag = code & 7;
  int e = int(scale) + exponent[mag];
  uint32_t fraction = (mag >= 3 && mag % 2) ? 0x400000u : 0;
  uint32_t bits = uint32_t(code & 8) << 28;
  if (scale == 255)
    bits |= 0x7fc00000;
  else if (mag) {
    if (e >= 255)
      bits |= 0x7f800000;
    else if (e <= 0)
      bits |= (0x800000u | fraction) >> (1 - e);
    else
      bits |= uint32_t(e) << 23 | fraction;
  }
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
inline void decodeFP4(const uint8_t *src, uint8_t scale, float *out,
                      int bytes) {
  for (int i = 0; i < bytes; i++) {
    out[2 * i] = scaledE2M1(src[i] & 15, scale);
    out[2 * i + 1] = scaledE2M1(src[i] >> 4, scale);
  }
}
#if defined(__riscv_v_intrinsic)
// A tail-undisturbed accumulator keeps earlier lanes live on the last strip.
inline float reduceSum(vfloat32m4_t v, size_t vl) {
  auto zero = __riscv_vfmv_v_f_f32m1(0.f, 1);
  return __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(v, zero, vl));
}
inline vfloat32m4_t bf16Vector(vfloat32m4_t v, size_t vl) {
  auto u = __riscv_vreinterpret_v_f32m4_u32m4(v);
  auto lsb = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(u, 16, vl), 1, vl);
  auto rounded = __riscv_vand_vx_u32m4(
      __riscv_vadd_vv_u32m4(__riscv_vadd_vx_u32m4(u, 0x7fff, vl), lsb, vl),
      0xffff0000u, vl);
  auto nan = __riscv_vmsgtu_vx_u32m4_b8(
      __riscv_vand_vx_u32m4(u, 0x7fffffff, vl), 0x7f800000, vl);
  return __riscv_vreinterpret_v_u32m4_f32m4(
      __riscv_vmerge_vvm_u32m4(rounded, u, nan, vl));
}
inline vfloat32m4_t loadE4M3(const uint8_t *src, size_t vl) {
  auto bits = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(src, vl), vl);
  auto mag = __riscv_vand_vx_u32m4(bits, 127, vl);
  auto normal =
      __riscv_vadd_vx_u32m4(__riscv_vsll_vx_u32m4(mag, 20, vl), 120u << 23, vl);
  auto small =
      __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(mag, vl), 0x1p-9f, vl);
  normal = __riscv_vmerge_vvm_u32m4(normal,
                                    __riscv_vreinterpret_v_f32m4_u32m4(small),
                                    __riscv_vmsltu_vx_u32m4_b8(mag, 8, vl), vl);
  normal = __riscv_vmerge_vxm_u32m4(
      normal, 0x7fc00000, __riscv_vmseq_vx_u32m4_b8(mag, 127, vl), vl);
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(
      normal,
      __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(bits, 128, vl), 24, vl), vl));
}
#endif
inline const std::array<float, 256> &e4m3Table() {
  static const auto lut = [] {
    std::array<float, 256> a{};
    for (int i = 0; i < 256; i++)
      a[i] = e4m3(i);
    return a;
  }();
  return lut;
}
inline void decodeFP8(const uint8_t *src, const uint8_t *scales, int k,
                      int block, float *out) {
  for (int j = 0; j < k; j += block) {
    float s = ue8m0(*scales++);
    int end = std::min(j + block, k);
#if defined(__riscv_v_intrinsic)
    for (int t = j; t < end;) {
      size_t vl = __riscv_vsetvl_e32m4(end - t);
      __riscv_vse32_v_f32m4(
          out + t, __riscv_vfmul_vf_f32m4(loadE4M3(src + t, vl), s, vl), vl);
      t += vl;
    }
#else
    for (int t = j; t < end; t++)
      out[t] = e4m3Table()[src[t]] * s;
#endif
  }
}
// Each row starts on a byte boundary. A partial final block has its own scale.
inline void fp4MatVec(const float *x, const uint8_t *weights,
                      const uint8_t *scales, int k, int n, float *out) {
  if (k == 0) {
    std::fill_n(out, n, 0.f);
    return;
  }
  const size_t stride = (size_t(k) + 1) / 2, groups = (size_t(k) + 31) / 32;
#pragma omp parallel for schedule(static) if (size_t(k) * n >= 262144)
  for (int r = 0; r < n; ++r) {
    const auto *row = weights + r * stride, *scale = scales + r * groups;
#if defined(__riscv_v_intrinsic)
    auto lut = __riscv_vle32_v_f32m4(E2M1, 16);
    size_t width = __riscv_vsetvl_e32m4(32);
    auto acc = __riscv_vfmv_v_f_f32m4(0, width);
    for (int j = 0; j < k; j += 32) {
      float s = ue8m0(scale[j / 32]);
      for (int t = j; t < std::min(k, j + 32);) {
        size_t vl = std::min<size_t>(width, std::min(k, j + 32) - t);
        size_t bytes = (vl + 1) / 2;
        auto pairs = __riscv_vzext_vf2_u16m1(
            __riscv_vle8_v_u8mf2(row + t / 2, bytes), bytes);
        // In little-endian e8 lanes each halfword becomes [low, high].
        pairs = __riscv_vor_vv_u16m1(
            pairs, __riscv_vsll_vx_u16m1(pairs, 4, bytes), bytes);
        auto ids = __riscv_vand_vx_u8m1(
            __riscv_vreinterpret_v_u16m1_u8m1(pairs), 15, vl);
        auto w = __riscv_vrgather_vv_f32m4(
            lut, __riscv_vzext_vf4_u32m4(ids, vl), vl);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, __riscv_vfmul_vf_f32m4(w, s, vl),
                                         __riscv_vle32_v_f32m4(x + t, vl), vl);
        t += vl;
      }
    }
    out[r] = reduceSum(acc, width);
#else
    float sum = 0;
    for (int j = 0; j < k; j += 32) {
      float s = ue8m0(scale[j / 32]);
      for (int t = j; t < std::min(k, j + 32); ++t)
        sum += x[t] * (E2M1[(row[t / 2] >> (4 * (t % 2))) & 15] * s);
    }
    out[r] = sum;
#endif
  }
}
// FP8 linear scales are shared by block x block tiles (including edge tiles).
inline void fp8MatVec(const float *x, const uint8_t *weights,
                      const uint8_t *scales, int k, int n, int block,
                      float *out) {
  if (k == 0) {
    std::fill_n(out, n, 0.f);
    return;
  }
  const size_t groups = (size_t(k) + block - 1) / block;
#if !defined(__riscv_v_intrinsic)
  const auto &lut = e4m3Table();
#endif
#pragma omp parallel for schedule(static) if (size_t(k) * n >= 262144)
  for (int r = 0; r < n; ++r) {
    const auto *row = weights + size_t(r) * k,
               *scale = scales + (r / block) * groups;
#if defined(__riscv_v_intrinsic)
    size_t width = __riscv_vsetvl_e32m4(std::min(k, block));
    auto acc = __riscv_vfmv_v_f_f32m4(0, width);
    for (int j = 0; j < k; j += block) {
      float s = ue8m0(scale[j / block]);
      for (int t = j; t < std::min(k, j + block);) {
        size_t vl = std::min<size_t>(width, std::min(k, j + block) - t);
        auto w = __riscv_vfmul_vf_f32m4(loadE4M3(row + t, vl), s, vl);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, w,
                                         __riscv_vle32_v_f32m4(x + t, vl), vl);
        t += vl;
      }
    }
    out[r] = reduceSum(acc, width);
#else
    float sum = 0;
    for (int j = 0; j < k; j += block) {
      float s = ue8m0(scale[j / block]);
      for (int t = j; t < std::min(k, j + block); ++t)
        sum += x[t] * (lut[row[t]] * s);
    }
    out[r] = sum;
#endif
  }
}
inline float dot(const float *a, const float *b, size_t n) {
#if defined(__riscv_v_intrinsic)
  if (n == 0)
    return 0;
  size_t width = __riscv_vsetvl_e32m4(n);
  auto acc = __riscv_vfmv_v_f_f32m4(0, width);
  for (size_t i = 0; i < n;) {
    size_t vl = std::min(width, n - i);
    acc = __riscv_vfmacc_vv_f32m4_tu(acc, __riscv_vle32_v_f32m4(a + i, vl),
                                     __riscv_vle32_v_f32m4(b + i, vl), vl);
    i += vl;
  }
  return reduceSum(acc, width);
#else
  float s = 0;
  for (size_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
#endif
}
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline void linear(const float *a, const float *w, int k, int n, float *out) {
#pragma omp parallel for if (n * k >= 262144)
  for (int j = 0; j < n; j++)
    out[j] = dot(a, w + j * k, k);
}
inline void w4linear(const float *x, const uint8_t *packed, int k, int n,
                     float *out) {
  constexpr int NB = 128, G = 32, bytes = (16 + 2) * NB;
  std::vector<int8_t> q(k);
  Vec scales(k / G);
  for (int g = 0; g < k / G; g++) {
    float peak = 0;
    for (int t = g * G; t < (g + 1) * G; t++)
      peak = std::max(peak, std::abs(x[t]));
    scales[g] = peak / 127;
    float inv = peak > 0 ? 127 / peak : 0;
    for (int t = g * G; t < (g + 1) * G; t++)
      q[t] = int8_t(std::clamp(std::nearbyint(x[t] * inv), -127.f, 127.f));
  }
#pragma omp parallel for
  for (int tile = 0; tile < n / NB; tile++) {
    float acc[NB] = {};
    for (int g = 0; g < k / G; g++) {
      auto p = packed + (size_t(tile) * (k / G) + g) * bytes;
      int16_t tmp[NB] = {};
      for (int i = 0; i < G / 2; i++) {
        int a = q[g * G + 2 * i], b = q[g * G + 2 * i + 1];
        for (int c = 0; c < NB; c++) {
          uint8_t v = p[i * NB + c];
          tmp[c] += a * (int((v & 15) ^ 8) - 8) + b * (int((v >> 4) ^ 8) - 8);
        }
      }
      for (int c = 0; c < NB; c++) {
        _Float16 sc;
        std::memcpy(&sc, p + 16 * NB + c * 2, 2);
        acc[c] = std::fma(float(tmp[c]), float(sc) * scales[g], acc[c]);
      }
    }
    std::copy_n(acc, NB, out + tile * NB);
  }
}
struct Mix {
  float pre[4], post[4], comb[4][4];
};
inline Mix splitSinkhorn(const float *m, const float *scale, const float *base,
                         int iters = 20, float eps = 1e-6f) {
  Mix z{};
  for (int i = 0; i < 4; i++) {
    z.pre[i] = sigmoid(m[i] * scale[0] + base[i]) + eps;
    z.post[i] = 2 * sigmoid(m[4 + i] * scale[1] + base[4 + i]);
    float *row = z.comb[i];
    for (int j = 0; j < 4; j++) {
      int id = 8 + 4 * i + j;
      row[j] = m[id] * scale[2] + base[id];
    }
    float mx = *std::max_element(row, row + 4);
    for (int j = 0; j < 4; j++)
      row[j] = std::exp(row[j] - mx);
    float sum = std::accumulate(row, row + 4, 0.f);
    for (int j = 0; j < 4; j++)
      row[j] = row[j] / sum + eps;
  }
  for (int n = 0; n < iters; n++) {
    if (n)
      for (auto &row : z.comb) {
        float sum = std::accumulate(row, row + 4, 0.f) + eps;
        for (float &v : row)
          v /= sum;
      }
    for (int j = 0; j < 4; j++) {
      float sum = 0;
      for (auto &row : z.comb)
        sum += row[j];
      for (auto &row : z.comb)
        row[j] /= (sum + eps);
    }
  }
  return z;
}
// Blocked dot for a panel of tokens; the single-row kernel instead uses
// eight independent accumulators so it can hide the latency of a vector FMA.
template <int Lanes>
inline float panelDot(const float *x, const float *w, int n, int block) {
#if defined(__riscv_v_intrinsic)
  if (n > 512)
    return dot(x, w, n);
  // On short panels preserve the reference lane order. A one-ulp change
  // in a mixing coefficient can put the residual on the other BF16 tie.
  float result = 0;
  for (int start = 0; start < n; start += block) {
    auto acc = __riscv_vfmv_v_f_f32m4(0, Lanes);
    int end = std::min(n, start + block);
    for (int i = start; i < end; i += Lanes) {
      size_t vl = std::min(Lanes, end - i);
      acc = __riscv_vfmacc_vv_f32m4_tu(acc, __riscv_vle32_v_f32m4(x + i, vl),
                                       __riscv_vle32_v_f32m4(w + i, vl), vl);
    }
    auto zero = __riscv_vfmv_v_f_f32m1(0, 1);
    result += __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredosum_vs_f32m4_f32m1(acc, zero, Lanes));
  }
  return result;
#else
  float result = 0;
  for (int start = 0; start < n; start += block) {
    std::array<float, Lanes> lanes{};
    int end = std::min(start + block, n);
    for (int j = start; j < end; j++)
      lanes[(j - start) % Lanes] =
          std::fma(x[j], w[j], lanes[(j - start) % Lanes]);
    result += std::accumulate(lanes.begin(), lanes.end(), 0.f);
  }
  return result;
#endif
}
inline Mix mixes(const float *x, int h, const float *fn, const float *scale,
                 const float *base, bool single = true) {
  int k = 4 * h;
  float m[24];
#pragma omp parallel for schedule(static) if (k > 4096)
  for (int r = 0; r < 24; r++)
    m[r] = single ? panelDot<8>(x, fn + r * k, k, k)
                  : panelDot<2>(x, fn + r * k, k, 256);
  float squares = single ? panelDot<8>(x, x, k, k) : dot(x, x, k);
  float inv = 1 / std::sqrt(squares / k + 1e-20f);
  for (float &v : m)
    v *= inv;
  return splitSinkhorn(m, scale, base);
}
inline void hcPre(const float *x, const float *pre, int h, float *out) {
#if defined(__riscv_v_intrinsic)
  for (int j = 0; j < h;) {
    size_t vl = __riscv_vsetvl_e32m4(h - j);
    auto sum =
        __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x + j, vl), pre[0], vl);
    for (int r = 1; r < 4; ++r) {
      auto term = __riscv_vfmul_vf_f32m4(
          __riscv_vle32_v_f32m4(x + r * h + j, vl), pre[r], vl);
      sum = __riscv_vfadd_vv_f32m4(sum, term, vl);
    }
    __riscv_vse32_v_f32m4(out + j, sum, vl);
    j += vl;
  }
#else
  for (int j = 0; j < h; j++)
    out[j] = pre[0] * x[j] + pre[1] * x[h + j] + pre[2] * x[2 * h + j] +
             pre[3] * x[3 * h + j];
#endif
}
inline void hcPost(const float *y, const float *x, const Mix &z, int h,
                   float *out) {
#if defined(__riscv_v_intrinsic)
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < h;) {
      size_t vl = __riscv_vsetvl_e32m4(h - i);
      auto sum = __riscv_vfmv_v_f_f32m4(0, vl);
      for (int r = 0; r < 4; ++r) {
        auto term = __riscv_vfmul_vf_f32m4(
            __riscv_vle32_v_f32m4(x + r * h + i, vl), z.comb[r][c], vl);
        sum = __riscv_vfadd_vv_f32m4(sum, term, vl);
      }
      auto yy = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + i, vl),
                                       z.post[c], vl);
      __riscv_vse32_v_f32m4(out + c * h + i,
                            __riscv_vfadd_vv_f32m4(sum, yy, vl), vl);
      i += vl;
    }
#else
  for (int c = 0; c < 4; c++)
    for (int i = 0; i < h; i++) {
      float sum = 0;
      for (int r = 0; r < 4; r++)
        sum += z.comb[r][c] * x[r * h + i];
      out[c * h + i] = z.post[c] * y[i] + sum;
    }
#endif
}
// Match the CPU reference's topk tie order for finite scores and -infinity.
inline Ids topk(const Vec &scores, int k, bool sorted = true) {
  Ids ids(scores.size());
  std::iota(ids.begin(), ids.end(), 0);
  if (k <= 0)
    return {};
  auto cmp = [&](int a, int b) { return scores[a] > scores[b]; };
  if (size_t(k) * 64 <= ids.size())
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), cmp);
  else {
    std::nth_element(ids.begin(), ids.begin() + k - 1, ids.end(), cmp);
    if (sorted)
      std::sort(ids.begin(), ids.begin() + k - 1, cmp);
  }
  ids.resize(k);
  return ids;
}
inline Ids route(Vec &scores, const float *bias, int k, float temp, float scale,
                 Vec &coeff) {
  Vec corrected(scores.size());
  for (size_t i = 0; i < scores.size(); i++) {
    float x = scores[i] / temp;
    scores[i] = std::sqrt(x > 20 ? x : std::log1p(std::exp(x)));
    corrected[i] = scores[i] + bias[i];
  }
  auto ids = topk(corrected, k);
  float sum = 0;
  for (int i : ids)
    sum += scores[i];
  coeff.resize(k);
  for (int j = 0; j < k; j++)
    coeff[j] =
        (k > 1 ? scores[ids[j]] / (sum + 1e-20f) : scores[ids[j]]) * scale;
  return ids;
}
inline float swiglu(float gate, float up, float weight, float limit) {
  if (limit > 0) {
    gate = std::min(gate, limit);
    up = std::clamp(up, -limit, limit);
  }
  return (gate * sigmoid(gate)) * up * weight;
}
// Project labels 1, 3 and 2 denote gate, up and down. The caller chooses
// f32, checkpoint FP4/FP8 or K3 tiles, and the sublayer's storage rounding.
template <class Projection, class StoreActivation>
Vec moeV4(const Vec &x, Vec scores, const float *bias, int k, float temp,
          float scale, float limit, Projection project, StoreActivation store) {
  Vec coeff;
  auto ids = route(scores, bias, k, temp, scale, coeff);
  Vec out(x.size());
  auto expert = [&](int id, float weight) {
    auto gate = project(id, 1, x), up = project(id, 3, x);
    for (size_t j = 0; j < gate.size(); j++)
      gate[j] = store(swiglu(gate[j], up[j], weight, limit));
    auto y = project(id, 2, gate);
    for (size_t j = 0; j < out.size(); j++)
      out[j] += y[j];
  };
  auto sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  for (int id : sorted)
    expert(id, coeff[std::find(ids.begin(), ids.end(), id) - ids.begin()]);
  expert(scores.size(), 1.f);
  for (float &v : out)
    v = store(v);
  return out;
}
inline void addScaled(float *dst, const float *src, float factor, int n) {
#if defined(__riscv_v_intrinsic)
  for (int i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m4(n - i);
    auto y = __riscv_vfmacc_vf_f32m4(__riscv_vle32_v_f32m4(dst + i, vl), factor,
                                     __riscv_vle32_v_f32m4(src + i, vl), vl);
    __riscv_vse32_v_f32m4(dst + i, y, vl);
    i += vl;
  }
#else
  for (int i = 0; i < n; ++i)
    dst[i] += factor * src[i];
#endif
}
inline void engramWrite(float *h, const float *kv, const float *qw,
                        const float *kw, int d) {
  auto value = kv + 4 * d;
#pragma omp parallel for schedule(static) if (d >= 4096)
  for (int c = 0; c < 4; c++) {
    float *x = h + c * d;
    auto key = kv + c * d;
    float rstd = (1 / std::sqrt(dot(x, x, d) / d + 1e-20f)) *
                 (1 / std::sqrt(dot(key, key, d) / d + 1e-20f));
#if defined(__riscv_v_intrinsic)
    size_t width = __riscv_vsetvl_e32m4(d);
    auto acc = __riscv_vfmv_v_f_f32m4(0, width);
    for (int i = 0; i < d;) {
      size_t vl = std::min<size_t>(width, d - i);
      auto w =
          __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(qw + c * d + i, vl),
                                 __riscv_vle32_v_f32m4(kw + c * d + i, vl), vl);
      auto wx = __riscv_vfmul_vv_f32m4(w, __riscv_vle32_v_f32m4(x + i, vl), vl);
      acc = __riscv_vfmacc_vv_f32m4_tu(acc, wx,
                                       __riscv_vle32_v_f32m4(key + i, vl), vl);
      i += vl;
    }
    float sum = reduceSum(acc, width);
#else
    float sum = 0;
    for (int i = 0; i < d; i++)
      sum += x[i] * (qw[c * d + i] * kw[c * d + i]) * key[i];
#endif
    float score = sum * rstd / std::sqrt(float(d));
    float gate = sigmoid(
        std::copysign(std::sqrt(std::max(std::abs(score), 1e-6f)), score));
    addScaled(x, value, gate, d);
  }
}
inline Ids hashRows(const int64_t *history, int position, int pad, int order,
                    int heads, const int64_t *mult, const int64_t *prime,
                    const int64_t *offset) {
  bool blocked = false;
  int64_t hash = 0;
  Ids rows;
  for (int s = 0; s < order; s++) {
    blocked = blocked || position < s || history[position - s] < 0;
    hash ^= int64_t(blocked ? pad : history[position - s]) * mult[s];
    if (s)
      for (int j = 0; j < heads; j++) {
        int k = (s - 1) * heads + j;
        rows.push_back(int(offset[k] + hash % prime[k]));
      }
  }
  return rows;
}
} // namespace buddy::v4
