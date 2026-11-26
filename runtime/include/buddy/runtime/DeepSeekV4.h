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
  return b == 255 ? NAN : std::ldexp(1.f, int(b) - 127);
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
inline void decodeFP8(const uint8_t *src, const uint8_t *scales, int k,
                      int block, float *out) {
  static const auto lut = [] {
    std::array<float, 256> a{};
    for (int i = 0; i < 256; i++)
      a[i] = e4m3(i);
    return a;
  }();
  for (int j = 0; j < k; j += block) {
    float s = ue8m0(*scales++);
    for (int t = j; t < std::min(j + block, k); t++)
      out[t] = lut[src[t]] * s;
  }
}
inline float dot(const float *a, const float *b, int n) {
  float s = 0;
  for (int i = 0; i < n; i++)
    s += a[i] * b[i];
  return s;
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
}
inline Mix mixes(const float *x, int h, const float *fn, const float *scale,
                 const float *base, bool single = true) {
  int k = 4 * h;
  float m[24];
#pragma omp parallel for if (k > 4096)
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
  for (int j = 0; j < h; j++)
    out[j] = pre[0] * x[j] + pre[1] * x[h + j] + pre[2] * x[2 * h + j] +
             pre[3] * x[3 * h + j];
}
inline void hcPost(const float *y, const float *x, const Mix &z, int h,
                   float *out) {
  for (int c = 0; c < 4; c++)
    for (int i = 0; i < h; i++) {
      float sum = 0;
      for (int r = 0; r < 4; r++)
        sum += z.comb[r][c] * x[r * h + i];
      out[c * h + i] = z.post[c] * y[i] + sum;
    }
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
inline void engramWrite(float *h, const float *kv, const float *qw,
                        const float *kw, int d) {
  auto value = kv + 4 * d;
  for (int c = 0; c < 4; c++) {
    float *x = h + c * d;
    auto key = kv + c * d;
    float rstd = (1 / std::sqrt(dot(x, x, d) / d + 1e-20f)) *
                 (1 / std::sqrt(dot(key, key, d) / d + 1e-20f));
    float sum = 0;
    for (int i = 0; i < d; i++)
      sum += x[i] * (qw[c * d + i] * kw[c * d + i]) * key[i];
    float score = sum * rstd / std::sqrt(float(d));
    float gate = sigmoid(
        std::copysign(std::sqrt(std::max(std::abs(score), 1e-6f)), score));
    for (int i = 0; i < d; i++)
      x[i] += gate * value[i];
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
