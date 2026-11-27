#pragma once
#include "DeepSeekV4.h"
#include <stdexcept>

namespace buddy::v4 {
inline uint8_t fp8Code(float value) {
  if (std::isnan(value) || std::abs(value) > 464.f)
    return std::signbit(value) ? 255 : 127;
  float x = std::min(std::abs(value), 448.f);
  int e = std::max(-6, std::ilogb(x));
  int unit = int(std::nearbyint(std::ldexp(x, 3 - e)));
  int code = (e + 6) * 8 + unit;
  return std::min(code, 126) | (std::signbit(value) ? 128 : 0);
}
inline float roundScale(float x) {
  int e;
  float m = std::frexp(x, &e);
  return std::ldexp(1.f, e - (m == .5f));
}
inline uint8_t fp4Code(float v) {
  constexpr float mid[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
  float x = std::abs(v);
  int c = 0;
  while (c < 7 && (x > mid[c] || (x == mid[c] && c % 2)))
    ++c;
  return c | (std::signbit(v) ? 8 : 0);
}
enum class Format { E4, E2, E2ScaleE4 };
inline void quantize(float *x, int n, int block, Format format) {
  for (int i = 0; i < n; i += block) {
    float peak = 0;
    for (int j = 0; j < block; j++)
      peak = std::max(peak, std::abs(x[i + j]));
    float s;
    if (format == Format::E4)
      s = roundScale(std::max(1e-4f, peak) * (1.f / 448));
    else if (format == Format::E2)
      s = roundScale(std::max(6 * 0x1p-126f, peak) * (1.f / 6));
    else
      s = e4m3(fp8Code(std::max(6 * 0x1p-9f, peak) / 6));
    for (int j = 0; j < block; j++) {
      float v = x[i + j] / s;
      x[i + j] = bf16(
          s * (format == Format::E4 ? e4m3(fp8Code(v)) : E2M1[fp4Code(v)]));
    }
  }
}
inline void norm(Vec &x, const float *w, float eps = 1e-20f) {
  float inv = 1 / std::sqrt(dot(x.data(), x.data(), x.size()) / x.size() + eps);
#if defined(__riscv_v_intrinsic)
  for (size_t i = 0; i < x.size();) {
    size_t vl = __riscv_vsetvl_e32m4(x.size() - i);
    auto y = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x.data() + i, vl),
                                    inv, vl);
    y = __riscv_vfmul_vv_f32m4(y, __riscv_vle32_v_f32m4(w + i, vl), vl);
    __riscv_vse32_v_f32m4(x.data() + i, bf16Vector(y, vl), vl);
    i += vl;
  }
#else
  for (size_t j = 0; j < x.size(); j++)
    x[j] = bf16(w[j] * (x[j] * inv));
#endif
}
inline void rotate(float *tail, const float *freq, int rd,
                   bool inverse = false) {
#if defined(__riscv_v_intrinsic)
  for (int j = 0; j < rd;) {
    size_t vl = __riscv_vsetvl_e32m4((rd - j) / 2);
    auto a = __riscv_vlse32_v_f32m4(tail + j, 2 * sizeof(float), vl);
    auto b = __riscv_vlse32_v_f32m4(tail + j + 1, 2 * sizeof(float), vl);
    auto c = __riscv_vlse32_v_f32m4(freq + j, 2 * sizeof(float), vl);
    auto s = __riscv_vlse32_v_f32m4(freq + j + 1, 2 * sizeof(float), vl);
    if (inverse)
      s = __riscv_vfneg_v_f32m4(s, vl);
    auto real = __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(a, c, vl),
                                       __riscv_vfmul_vv_f32m4(b, s, vl), vl);
    auto imag = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(a, s, vl),
                                       __riscv_vfmul_vv_f32m4(b, c, vl), vl);
    __riscv_vsse32_v_f32m4(tail + j, 2 * sizeof(float), bf16Vector(real, vl),
                           vl);
    __riscv_vsse32_v_f32m4(tail + j + 1, 2 * sizeof(float),
                           bf16Vector(imag, vl), vl);
    j += 2 * vl;
  }
#else
  for (int j = 0; j < rd; j += 2) {
    float a = tail[j], b = tail[j + 1], c = freq[j],
          s = inverse ? -freq[j + 1] : freq[j + 1];
    tail[j] = bf16(a * c - b * s);
    tail[j + 1] = bf16(a * s + b * c);
  }
#endif
}
inline std::vector<uint8_t> candidates(const Vec &score, int len, int blocks,
                                       int size) {
  Vec maxima((score.size() + size - 1) / size, -INFINITY);
  for (int i = 0; i < len; i++)
    maxima[i / size] = std::max(maxima[i / size], score[i]);
  if (len)
    maxima[(len - 1) / size] = INFINITY;
  auto ids = topk(maxima, std::min<int>(blocks, maxima.size()));
  std::vector<uint8_t> mask(score.size());
  for (int b : ids)
    if (maxima[b] > -INFINITY)
      std::fill(mask.begin() + b * size,
                mask.begin() + std::min<size_t>((b + 1) * size, mask.size()),
                1);
  return mask;
}
inline Ids selectIndex(Vec scores, int len, int k, int offset,
                       const std::vector<uint8_t> *mask) {
  for (size_t j = 0; j < scores.size(); j++)
    if (int(j) >= len || (mask && !(*mask)[j]))
      scores[j] = -INFINITY;
  auto idx = topk(scores, std::min<int>(k, scores.size()), false);
  std::sort(idx.begin(), idx.end());
  for (int &i : idx)
    i = i < len ? i + offset : -1;
  return idx;
}
inline void sparse(const float *q, const std::vector<const float *> &keys,
                   int h, int d, const float *sink, float *out) {
  const float invSqrtD = 1 / std::sqrt(float(d));
#pragma omp parallel for schedule(static) if (h * d >= 8192)
  for (int head = 0; head < h; head++) {
    float max = -1e30f, den = 0;
    float *o = out + head * d;
    std::fill_n(o, d, 0.f);
    for (size_t off = 0; off < keys.size(); off += 64) {
      int n = std::min<size_t>(64, keys.size() - off);
      float s[64], old = max;
      for (int j = 0; j < n; j++) {
        auto v = keys[off + j];
        s[j] = v ? dot(q + head * d, v, d) * invSqrtD : -INFINITY;
        max = std::max(max, s[j]);
      }
      float adj = std::exp(old - max);
      den *= adj;
#if defined(__riscv_v_intrinsic)
      for (int k = 0; k < d;) {
        size_t vl = __riscv_vsetvl_e32m4(d - k);
        auto v = __riscv_vle32_v_f32m4(o + k, vl);
        __riscv_vse32_v_f32m4(o + k, __riscv_vfmul_vf_f32m4(v, adj, vl), vl);
        k += vl;
      }
#else
      for (int k = 0; k < d; k++)
        o[k] *= adj;
#endif
      float sum = 0;
      for (int j = 0; j < n; j++) {
        float exp = std::exp(s[j] - max);
        sum += exp;
        if (auto v = keys[off + j]) {
          float p = bf16(exp);
          addScaled(o, v, p, d);
        }
      }
      den += sum;
    }
    den += std::exp(sink[head] - max);
#if defined(__riscv_v_intrinsic)
    for (int k = 0; k < d;) {
      size_t vl = __riscv_vsetvl_e32m4(d - k);
      auto v = __riscv_vle32_v_f32m4(o + k, vl);
      __riscv_vse32_v_f32m4(
          o + k, bf16Vector(__riscv_vfdiv_vf_f32m4(v, den, vl), vl), vl);
      k += vl;
    }
#else
    for (int k = 0; k < d; k++)
      o[k] = bf16(o[k] / den);
#endif
  }
}
} // namespace buddy::v4
