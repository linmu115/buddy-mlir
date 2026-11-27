// Floating point helpers for Route B. The reduction seed crosses VL boundaries
// so changing VLEN does not change the order of the dot product.
#pragma once
#include <cmath>
#include <cstddef>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace hybrid {
inline float orderedDot(const float *a, const float *b, int n) {
#ifdef __riscv_vector
  auto sum = __riscv_vfmv_v_f_f32m1(0.f, 1);
  while (n) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    auto product = __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(a, vl),
                                          __riscv_vle32_v_f32m1(b, vl), vl);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(product, sum, vl);
    a += vl;
    b += vl;
    n -= vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(sum);
#else
  float sum = 0;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
#endif
}

inline void addScaled(float *__restrict dst, const float *__restrict src, int n,
                      float scale) {
#ifdef __riscv_vector
  while (n) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    auto x = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(src, vl), scale, vl);
    auto y = __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(dst, vl), x, vl);
    __riscv_vse32_v_f32m1(dst, y, vl);
    dst += vl;
    src += vl;
    n -= vl;
  }
#else
  for (int i = 0; i < n; ++i)
    dst[i] += scale * src[i];
#endif
}

inline float sumScores(const float *scores, int n, float seed) {
#ifdef __riscv_vector
  auto sum = __riscv_vfmv_v_f_f32m1(seed, 1);
  while (n) {
    size_t vl = __riscv_vsetvl_e32m1(n);
    sum = __riscv_vfredosum_vs_f32m1_f32m1(__riscv_vle32_v_f32m1(scores, vl),
                                           sum, vl);
    scores += vl;
    n -= vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(sum);
#else
  for (int i = 0; i < n; ++i)
    seed += scores[i];
  return seed;
#endif
}

// Keep one value vector in registers across all probabilities in a KV tile.
inline void accumulateValues(float *out, const float *values, const float *p,
                             int n, int dim) {
#ifdef __riscv_vector
  for (int j = 0; j < dim;) {
    size_t vl = __riscv_vsetvl_e32m1(dim - j);
    auto y = __riscv_vle32_v_f32m1(out + j, vl);
    for (int i = 0; i < n; ++i) {
      auto v = __riscv_vle32_v_f32m1(values + i * dim + j, vl);
      y = __riscv_vfadd_vv_f32m1(y, __riscv_vfmul_vf_f32m1(v, p[i], vl), vl);
    }
    __riscv_vse32_v_f32m1(out + j, y, vl);
    j += vl;
  }
#else
  for (int i = 0; i < n; ++i)
    addScaled(out, values + i * dim, dim, p[i]);
#endif
}
} // namespace hybrid
