// One head of the fp32 gated delta rule. Q and K have already been
// normalized, and Q includes the inverse square root of the key dimension.
#pragma once
#include "HybridVector.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace hybrid {
class Scratch {
  std::unique_ptr<float, decltype(&std::free)> buffer{nullptr, std::free};

public:
  explicit Scratch(size_t count) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, count * sizeof(float)))
      throw std::bad_alloc();
    buffer.reset(static_cast<float *>(p));
    std::fill_n(buffer.get(), count, 0.f);
  }
  float *data() { return buffer.get(); }
  float &operator[](size_t i) { return data()[i]; }
};
} // namespace hybrid

namespace hybrid {
inline void chunkGDN(const float *q, const float *k, const float *v, int stride,
                     const float *logDecay, const float *beta, int heads,
                     float *state, float *out, int outStride, int length,
                     int dk, int dv) {
  Scratch cumulative(length), corrections(length * dv);
  float sum = 0;
  for (int t = 0; t < length; ++t)
    cumulative[t] = sum += logDecay[t * heads];

  // Forward substitution in the unit lower triangular system applied
  // directly to beta * (V - exp(G) K S0). S0 stays immutable in this loop.
  for (int t = 0; t < length; ++t) {
    const auto *kt = k + t * stride;
    const auto *qt = q + t * stride;
    float *u = corrections.data() + t * dv;
    float *y = out + t * outStride;
    float decay = std::exp(cumulative[t]);
    std::fill_n(y, dv, 0.f);
    for (int j = 0; j < dk; ++j) {
      addScaled(u, state + j * dv, dv, kt[j]);
      addScaled(y, state + j * dv, dv, qt[j]);
    }
    for (int i = 0; i < dv; ++i) {
      u[i] = beta[t * heads] * (v[t * stride + i] - decay * u[i]);
      y[i] *= decay;
    }
    for (int s = 0; s <= t; ++s) {
      const float *ks = k + s * stride;
      float kk = orderedDot(kt, ks, dk);
      float qk = orderedDot(qt, ks, dk);
      float relative = std::exp(cumulative[t] - cumulative[s]);
      const float *previous = corrections.data() + s * dv;
      if (s < t)
        addScaled(u, previous, dv, -beta[t * heads] * relative * kk);
      addScaled(y, previous, dv, relative * qk);
    }
  }
  // S_L = exp(G_L) S0 + K^T diag(exp(G_L - G)) U.
  const float finalDecay = std::exp(cumulative[length - 1]);
  for (int i = 0; i < dk * dv; ++i)
    state[i] *= finalDecay;
  for (int t = 0; t < length; ++t) {
    float scale = std::exp(cumulative[length - 1] - cumulative[t]);
    for (int j = 0; j < dk; ++j)
      addScaled(state + j * dv, corrections.data() + t * dv, dv,
                scale * k[t * stride + j]);
  }
}
} // namespace hybrid
