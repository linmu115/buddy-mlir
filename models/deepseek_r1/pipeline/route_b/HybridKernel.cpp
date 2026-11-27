// The generated header specializes dimensions, weights and MLIR entry points
// for one checkpoint shape. All state buffers belong to the Route B caller.
#include "HybridConfig.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "ChunkGDN.h"

namespace {

constexpr float Eps = normEpsilon;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

float squareSum(const float *x, int size) {
  float sum = 0;
  for (int i = 0; i < size; ++i)
    sum += x[i] * x[i];
  return sum;
}

void zeroCenteredNorm(const float *x, const float *w, int size, float *out) {
  const float scale = 1.0f / std::sqrt(squareSum(x, size) / size + Eps);
  for (int i = 0; i < size; ++i)
    out[i] = (x[i] * scale) * (1.0f + w[i]);
}

#if HYBRID_ROLE == 4
// Preserve the eight independent K accumulators of the original MLIR. Each
// input load feeds both projections and each weight load serves up to four
// rows.
using EightFloats = float __attribute__((vector_size(32)));
template <int R>
[[gnu::noinline]] void gateProjection(const float *x, const float *wa,
                                      const float *wb, float *a, float *b) {
#ifdef __riscv_vector
  // LMUL=2 also holds all eight lanes on a VLEN=128 implementation.
  constexpr size_t vl = 8;
  auto a0 = __riscv_vfmv_v_f_f32m2(0.f, vl), b0 = a0;
  auto a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
  for (int k = 0; k < width; k += vl) {
    auto aw = __riscv_vle32_v_f32m2(wa + k, vl);
    auto bw = __riscv_vle32_v_f32m2(wb + k, vl);
    auto step = [&](int r, vfloat32m2_t &av, vfloat32m2_t &bv) {
      auto input = __riscv_vle32_v_f32m2(x + r * width + k, vl);
      av =
          __riscv_vfadd_vv_f32m2(av, __riscv_vfmul_vv_f32m2(input, aw, vl), vl);
      bv =
          __riscv_vfadd_vv_f32m2(bv, __riscv_vfmul_vv_f32m2(input, bw, vl), vl);
    };
    step(0, a0, b0);
    if constexpr (R >= 2)
      step(1, a1, b1);
    if constexpr (R == 4) {
      step(2, a2, b2);
      step(3, a3, b3);
    }
  }
  auto save = [&](int r, vfloat32m2_t av, vfloat32m2_t bv) {
    auto zero = __riscv_vfmv_v_f_f32m1(0.f, 1);
    a[r] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredosum_vs_f32m2_f32m1(av, zero, vl));
    b[r] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredosum_vs_f32m2_f32m1(bv, zero, vl));
  };
  save(0, a0, b0);
  if constexpr (R >= 2)
    save(1, a1, b1);
  if constexpr (R == 4) {
    save(2, a2, b2);
    save(3, a3, b3);
  }
#else
  EightFloats sa[R] = {}, sb[R] = {};
  for (int k = 0; k < width; k += 8) {
    EightFloats aw, bw;
    std::memcpy(&aw, wa + k, sizeof aw);
    std::memcpy(&bw, wb + k, sizeof bw);
    for (int r = 0; r < R; ++r) {
      EightFloats input;
      std::memcpy(&input, x + r * width + k, sizeof input);
      sa[r] += input * aw;
      sb[r] += input * bw;
    }
  }
  for (int r = 0; r < R; ++r) {
    float av = 0, bv = 0;
    for (int i = 0; i < 8; ++i) {
      av += sa[r][i];
      bv += sb[r][i];
    }
    a[r] = av;
    b[r] = bv;
  }
#endif
}

[[gnu::noinline]] void convolve(float *row, float *history, const float *filter,
                                int channels, float *snapshot) {
#ifdef __riscv_vector
  if constexpr (convKernel == 4) {
    for (int c = 0; c < channels;) {
      size_t vl = __riscv_vsetvl_e32m1(channels - c);
      auto hist = __riscv_vlseg4e32_v_f32m1x4(history + 4 * c, vl);
      auto taps = __riscv_vlseg4e32_v_f32m1x4(filter + 4 * c, vl);
      auto incoming = __riscv_vle32_v_f32m1(row + c, vl);
      auto sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
      auto shifted = hist;
      auto tap = [&](auto index, vfloat32m1_t sample) {
        constexpr int j = decltype(index)::value;
        shifted = __riscv_vset_v_f32m1_f32m1x4(shifted, j, sample);
        auto weight = __riscv_vget_v_f32m1x4_f32m1(taps, j);
        sum = __riscv_vfadd_vv_f32m1(
            sum, __riscv_vfmul_vv_f32m1(sample, weight, vl), vl);
      };
      tap(std::integral_constant<int, 0>{},
          __riscv_vget_v_f32m1x4_f32m1(hist, 1));
      tap(std::integral_constant<int, 1>{},
          __riscv_vget_v_f32m1x4_f32m1(hist, 2));
      tap(std::integral_constant<int, 2>{},
          __riscv_vget_v_f32m1x4_f32m1(hist, 3));
      tap(std::integral_constant<int, 3>{}, incoming);
      __riscv_vsseg4e32_v_f32m1x4(history + 4 * c, shifted, vl);
      __riscv_vse32_v_f32m1(row + c, sum, vl);
      c += vl;
    }
  } else
#endif
    for (int c = 0; c < channels; ++c) {
      float *state = history + c * convKernel;
      for (int j = 1; j < convKernel; ++j)
        state[j - 1] = state[j];
      state[convKernel - 1] = row[c];
      float sum = 0;
      for (int j = 0; j < convKernel; ++j)
        sum += state[j] * filter[c * convKernel + j];
      row[c] = sum;
    }
  if (snapshot)
    std::copy_n(history, channels * convKernel, snapshot);
  for (int c = 0; c < channels; ++c)
    row[c] = silu(row[c]);
}

[[gnu::noinline]] void recurrentStep(float *state, const float *q,
                                     const float *k, const float *v, float *out,
                                     float beta, float alpha) {
#ifdef __riscv_vector
  for (int j = 0; j < valueDim;) {
    size_t vl = __riscv_vsetvl_e32m1(valueDim - j);
    auto old = __riscv_vfmv_v_f_f32m1(0.f, vl);
    for (int i = 0; i < keyDim; ++i) {
      auto *s = state + i * valueDim + j;
      auto prior =
          __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(s, vl), alpha, vl);
      __riscv_vse32_v_f32m1(s, prior, vl);
      old = __riscv_vfadd_vv_f32m1(old, __riscv_vfmul_vf_f32m1(prior, k[i], vl),
                                   vl);
    }
    auto delta = __riscv_vfmul_vf_f32m1(
        __riscv_vfsub_vv_f32m1(__riscv_vle32_v_f32m1(v + j, vl), old, vl), beta,
        vl);
    auto y = __riscv_vfmv_v_f_f32m1(0.f, vl);
    for (int i = 0; i < keyDim; ++i) {
      auto *s = state + i * valueDim + j;
      auto updated =
          __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(s, vl),
                                 __riscv_vfmul_vf_f32m1(delta, k[i], vl), vl);
      __riscv_vse32_v_f32m1(s, updated, vl);
      y = __riscv_vfadd_vv_f32m1(y, __riscv_vfmul_vf_f32m1(updated, q[i], vl),
                                 vl);
    }
    __riscv_vse32_v_f32m1(out + j, y, vl);
    j += vl;
  }
#else
  deltaStep(state, q, k, v, out, beta, alpha);
#endif
}

[[gnu::noinline]] void gatedNormRow(float *row, const float *z,
                                    const float *weight) {
  const float norm =
      1.f / std::sqrt(hybrid::orderedDot(row, row, valueDim) / valueDim + Eps);
#ifdef __riscv_vector
  // Keep libm rounding for the gate. Its independent f32 epilogue uses full VL.
  float denominator[valueDim];
  for (int i = 0; i < valueDim; ++i)
    denominator[i] = 1.f + std::exp(-z[i]);
  for (int j = 0; j < valueDim;) {
    size_t vl = __riscv_vsetvl_e32m1(valueDim - j);
    auto sig = __riscv_vfrdiv_vf_f32m1(
        __riscv_vle32_v_f32m1(denominator + j, vl), 1.f, vl);
    auto gate =
        __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(z + j, vl), sig, vl);
    auto y =
        __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(row + j, vl), norm, vl);
    y = __riscv_vfmul_vv_f32m1(y, __riscv_vle32_v_f32m1(weight + j, vl), vl);
    __riscv_vse32_v_f32m1(row + j, __riscv_vfmul_vv_f32m1(y, gate, vl), vl);
    j += vl;
  }
#else
  for (int j = 0; j < valueDim; ++j)
    row[j] = row[j] * norm * weight[j] * silu(z[j]);
#endif
}

void mix(const BJ47WeightsV1 &weights, const float *x, float *convState,
         float *ssmState, const BJ47FrameV1 &frame, int rows, int valid,
         const float *residual, float *out) {
  const float *w = weights.f32;
  hybrid::Scratch qkvBuffer(rows * convDim), z(rows * valueWidth),
      core(rows * valueWidth);
  proj_inProjQKV(weights, x, qkvBuffer.data(), rows);
  proj_inProjZ(weights, x, z.data(), rows);
  constexpr int group = valueHeads / keyHeads;
  const auto *snapshot =
      rows == 2 ? static_cast<const BJ47VerifyRows *>(frame.aux) : nullptr;
#ifdef HYBRID_THREADS
#pragma omp parallel for schedule(static) num_threads(HYBRID_THREADS)
#endif
  for (uint64_t kh = 0; kh < keyHeads; ++kh) {
    // Q and K belong to this group. Every history and S slice has one writer.
    auto prepare = [&](int t, int offset, int n) {
      auto base = offset * convKernel;
      convolve(qkvBuffer.data() + t * convDim + offset, convState + base,
               w + convWeight + base, n,
               snapshot && t == 0 ? snapshot->conv_snapshot + base : nullptr);
    };
    for (int t = 0; t < valid; ++t) {
      prepare(t, kh * keyDim, keyDim);
      prepare(t, keyWidth + kh * keyDim, keyDim);
      prepare(t, 2 * keyWidth + kh * group * valueDim, group * valueDim);
      auto *q = qkvBuffer.data() + t * convDim + kh * keyDim;
      auto *k = q + keyWidth;
      const float normQ =
          1.f / std::sqrt(hybrid::orderedDot(q, q, keyDim) + 1e-6f);
      const float normK =
          1.f / std::sqrt(hybrid::orderedDot(k, k, keyDim) + 1e-6f);
      const float queryScale = 1.f / std::sqrt(float(keyDim));
      for (int j = 0; j < keyDim; ++j) {
        q[j] = (q[j] * normQ) * queryScale;
        k[j] *= normK;
      }
    }
    for (int h = kh * group; h < int((kh + 1) * group); ++h) {
      float a[chunk], b[chunk];
      const float *wa = w + inProjA + h * width, *wb = w + inProjB + h * width;
      int t = 0;
      for (; t + 4 <= valid; t += 4)
        gateProjection<4>(x + t * width, wa, wb, a + t, b + t);
      if (t + 2 <= valid) {
        gateProjection<2>(x + t * width, wa, wb, a + t, b + t);
        t += 2;
      }
      if (t < valid)
        gateProjection<1>(x + t * width, wa, wb, a + t, b + t);
      float rate = -std::exp(w[aLog + h]);
      for (t = 0; t < valid; ++t) {
        b[t] = sigmoid(b[t]);
        float dt = a[t] + w[dtBias + h];
        a[t] = rate * (dt > 20.f ? dt : std::log1p(std::exp(dt)));
      }
      const auto *q = qkvBuffer.data() + kh * keyDim;
      const auto *k = q + keyWidth;
      const auto *v = qkvBuffer.data() + 2 * keyWidth + h * valueDim;
      float *state = ssmState + h * keyDim * valueDim;
      float *head = core.data() + h * valueDim;
      if (rows <= 2
#ifdef HYBRID_RECURRENT_REFERENCE
          || rows == chunk
#endif
      ) {
        for (int t = 0; t < valid; ++t) {
          recurrentStep(state, q + t * convDim, k + t * convDim,
                        v + t * convDim, head + t * valueWidth, b[t],
                        std::exp(a[t]));
          if (rows == 2 && t == 0)
            std::copy_n(
                state, keyDim * valueDim,
                static_cast<const BJ47VerifyRows *>(frame.aux)->ssm_snapshot +
                    h * keyDim * valueDim);
        }
      } else
        hybrid::chunkGDN(q, k, v, convDim, a, b, 1, state, head, valueWidth,
                         valid, keyDim, valueDim);
      for (int t = 0; t < valid; ++t)
        gatedNormRow(head + t * valueWidth,
                     z.data() + t * valueWidth + h * valueDim, w + gatedNorm);
    }
  }
#ifdef HYBRID_FUSED_GLU
  proj_mixerOut(weights, core.data(), out, rows, residual);
#else
  proj_mixerOut(weights, core.data(), out, rows);
#endif
}
#elif (HYBRID_ROLE == 5 || HYBRID_ROLE == 6)
void partialRotary(float *x, const float *cos, const float *sin) {
  for (int i = 0; i < rotaryDim / 2; ++i) {
    const float left = x[i], right = x[i + rotaryDim / 2];
    x[i] = left * cos[i] - right * sin[i];
    x[i + rotaryDim / 2] = right * cos[i] + left * sin[i];
  }
}

// Several adjacent queries use the same KV head. Finish scoring before
// value accumulation so no query stays live across the softmax.
template <int G>
void decodeGroup(float *q, const float *norm, const float *cos,
                 const float *sin, const float *keys, const float *values,
                 int length, float *out) {
  hybrid::Scratch scores(G * length);
  float maximum[G], denominator[G] = {};
  for (int h = 0; h < G; ++h) {
    auto *query = q + h * 2 * headDim;
    zeroCenteredNorm(query, norm, headDim, query);
    partialRotary(query, cos, sin);
    maximum[h] = -INFINITY;
  }
  const float scale = 1.0f / std::sqrt(float(headDim));
  for (int t = 0; t < length; ++t) {
    float dot[G] = {};
    for (int i = 0; i < headDim; ++i)
      for (int h = 0; h < G; ++h)
        dot[h] += q[h * 2 * headDim + i] * keys[t * headDim + i];
    for (int h = 0; h < G; ++h) {
      scores[t * G + h] = dot[h] * scale;
      maximum[h] = std::max(maximum[h], scores[t * G + h]);
    }
  }
  for (int t = 0; t < length; ++t)
    for (int h = 0; h < G; ++h) {
      float &s = scores[t * G + h];
      s = std::exp(s - maximum[h]);
      denominator[h] += s;
    }
  std::fill_n(out, G * headDim, 0.0f);
  for (int t = 0; t < length; ++t) {
    float prob[G];
    for (int h = 0; h < G; ++h)
      prob[h] = scores[t * G + h] / denominator[h];
    for (int i = 0; i < headDim; ++i)
      for (int h = 0; h < G; ++h)
        out[h * headDim + i] += prob[h] * values[t * headDim + i];
  }
  for (int h = 0; h < G; ++h)
    for (int i = 0; i < headDim; ++i)
      out[h * headDim + i] *= sigmoid(q[(h * 2 + 1) * headDim + i]);
}

// Stream tiles at the running maximum. When a later tile raises it, defer
// the invalidated prefix until the final maximum is known. Each exponential
// is evaluated at most twice. Replay preserves the ordered f32 denominator
// and rounds each probability before multiplying V, as required by W4.
[[gnu::noinline]] void onlineAttention(const float *q, const float *keys,
                                       const float *values, int length,
                                       float *out, float *scores,
                                       float *probabilities) {
  constexpr int tile = 32;
  float maximum = -INFINITY, denominator = 0;
  int stale = 0;
  const float scale = 1.f / std::sqrt(float(headDim));
  for (int start = 0; start < length; start += tile) {
    const int n = std::min(tile, length - start);
    float next = maximum;
    for (int i = start; i < start + n; ++i) {
      scores[i] = hybrid::orderedDot(q, keys + i * headDim, headDim) * scale;
      next = std::max(next, scores[i]);
    }
    if (next > maximum) {
      maximum = next;
      stale = start;
    }
    for (int i = start; i < start + n; ++i)
      probabilities[i] = std::exp(scores[i] - maximum);
    if (!stale)
      denominator = hybrid::sumScores(probabilities + start, n, denominator);
  }
  if (stale) {
    for (int i = 0; i < stale; ++i)
      probabilities[i] = std::exp(scores[i] - maximum);
    denominator = hybrid::sumScores(probabilities, length, 0.f);
  }
  std::fill_n(out, headDim, 0.f);
  for (int start = 0; start < length; start += tile) {
    int n = std::min(tile, length - start);
    for (int i = start; i < start + n; ++i)
      probabilities[i] /= denominator;
    hybrid::accumulateValues(out, values + start * headDim,
                             probabilities + start, n, headDim);
  }
}

void mix(const BJ47WeightsV1 &weights, const float *x, float *keys,
         float *values, const BJ47FrameV1 &frame, int rows, int valid,
         const float *residual, float *out) {
  const auto *w = weights.f32;
  constexpr int qWidth = attentionHeads * headDim;
  constexpr int kvWidth = kvHeads * headDim;
  hybrid::Scratch query(rows * 2 * qWidth), keyRows(rows * kvWidth),
      valueRows(rows * kvWidth), cores(rows * qWidth);
  proj_qProj(weights, x, query.data(), rows);
  proj_kProj(weights, x, keyRows.data(), rows);
  proj_vProj(weights, x, valueRows.data(), rows);
  for (int row = 0; row < valid; ++row) {
    const int position = frame.cache_position + row;
    const auto *cos = frame.cos_rows + row * rotaryDim;
    const auto *sin = frame.sin_rows + row * rotaryDim;
    float *k = keyRows.data() + row * kvWidth;
    float *v = valueRows.data() + row * kvWidth;
    for (int h = 0; h < kvHeads; ++h) {
      auto *key = k + h * headDim;
      zeroCenteredNorm(key, w + kNorm, headDim, key);
      partialRotary(key, cos, sin);
      const size_t offset = (size_t(h) * context + position) * headDim;
      for (int i = 0; i < headDim; ++i) {
        keys[offset + i] = key[i];
        values[offset + i] = v[h * headDim + i];
      }
    }
  }
#ifdef HYBRID_THREADS
  if (rows <= 2 && attentionHeads / kvHeads == 6 && headDim == 256) {
    int position = frame.cache_position;
    const int group = position < 384 ? 2 : 4;
    const int items = (6 + group - 1) / group;
#pragma omp parallel for schedule(static) num_threads(HYBRID_THREADS)
    for (uint64_t item = 0; item < uint64_t(kvHeads * items); ++item) {
      const int kv = item / items, first = (item % items) * group;
      const int h = kv * 6 + first;
      auto apply = (group == 4 && first == 0) ? decodeGroup<4> : decodeGroup<2>;
      for (int r = 0; r < valid; ++r)
        apply(query.data() + (r * attentionHeads + h) * 2 * headDim, w + qNorm,
              frame.cos_rows + r * rotaryDim, frame.sin_rows + r * rotaryDim,
              keys + size_t(kv) * context * headDim,
              values + size_t(kv) * context * headDim, position + r + 1,
              cores.data() + (r * attentionHeads + h) * headDim);
    }
  } else
#endif
  {
    // The complete KV suffix is now visible. Each row reads its causal prefix.
    constexpr bool online =
        headDim == 256 && attentionHeads == 24 && kvHeads == 4;
    const int stride = frame.cache_position + valid;
    hybrid::Scratch scratch(size_t(attentionHeads) * stride * (online ? 2 : 1));
#ifdef HYBRID_THREADS
#pragma omp parallel for schedule(static) num_threads(HYBRID_THREADS)
#endif
    for (uint64_t h = 0; h < attentionHeads; ++h) {
      for (int row = 0; row < valid; ++row) {
        const int position = frame.cache_position + row;
        const int kvHead = h / (attentionHeads / kvHeads);
        auto *q = query.data() + (row * attentionHeads + h) * 2 * headDim;
        auto *result = cores.data() + (row * attentionHeads + h) * headDim;
        const auto *gate = q + headDim;
        zeroCenteredNorm(q, w + qNorm, headDim, q);
        partialRotary(q, frame.cos_rows + row * rotaryDim,
                      frame.sin_rows + row * rotaryDim);
        const auto *headKeys = keys + size_t(kvHead) * context * headDim;
        const auto *headValues = values + size_t(kvHead) * context * headDim;
        if constexpr (online) {
          float *scores = scratch.data() + 2 * h * stride;
          onlineAttention(q, headKeys, headValues, position + 1, result, scores,
                          scores + stride);
        } else {
          float *scores = scratch.data() + h * stride;
          const float scale = 1.f / std::sqrt(float(headDim));
          float maxScore = -INFINITY;
          for (int t = 0; t <= position; ++t) {
            float dot = 0;
            for (int i = 0; i < headDim; ++i)
              dot += q[i] * headKeys[t * headDim + i];
            scores[t] = dot * scale;
            maxScore = std::max(maxScore, scores[t]);
          }
          float denominator = 0;
          for (int t = 0; t <= position; ++t) {
            scores[t] = std::exp(scores[t] - maxScore);
            denominator += scores[t];
          }

          std::fill_n(result, headDim, 0.0f);
          for (int t = 0; t <= position; ++t) {
            const float p = scores[t] / denominator;
            for (int i = 0; i < headDim; ++i)
              result[i] += p * headValues[t * headDim + i];
          }
        }
        for (int i = 0; i < headDim; ++i)
          result[i] *= sigmoid(gate[i]);
      }
    }
  }
#ifdef HYBRID_FUSED_GLU
  proj_mixerOut(weights, cores.data(), out, rows, residual);
#else
  proj_mixerOut(weights, cores.data(), out, rows);
#endif
}
#endif

#if HYBRID_ROLE == 4 || (HYBRID_ROLE == 5 || HYBRID_ROLE == 6)
int32_t layer(const BJ47WeightsV1 *weights, const float *input, float *a,
              float *b, const BJ47FrameV1 *frame, float *out, int rows) {
  if (!weights || !weights->f32 || !input || !a || !b || a == b || !frame ||
      !frame->aux || !out || input == out)
    return -1;
  if (kernelInfo.rotary_dim && (!frame->cos_rows || !frame->sin_rows))
    return -1;

  const auto validRows =
      static_cast<const BJ47FrameRowsV2 *>(frame->aux)->valid_rows;
  if (validRows < 1 || validRows > rows || frame->cache_position < 0 ||
      frame->cache_position > context - validRows)
    return -2;

  if (kernelInfo.base.i8_elements && !weights->q4)
    return -1;
  const auto *w = weights->f32;
  hybrid::Scratch norm(rows * width), residual(rows * width),
      gate(rows * intermediate);
  for (int t = 0; t < validRows; ++t)
    zeroCenteredNorm(input + t * width, w + inputNorm, width,
                     norm.data() + t * width);
  const float *skip = input;
#ifdef HYBRID_FUSED_GLU
  // The fused projection reads a full tile. Only valid input rows belong
  // to the caller on a padded final chunk.
  if (validRows < rows) {
    std::copy_n(input, validRows * width, residual.data());
    skip = residual.data();
  }
#endif
  mix(*weights, norm.data(), a, b, *frame, rows, validRows, skip,
      residual.data());
  for (int t = 0; t < validRows; ++t) {
    float *r = residual.data() + t * width;
#ifndef HYBRID_FUSED_GLU
    for (int i = 0; i < width; ++i)
      r[i] += input[t * width + i];
#endif
    zeroCenteredNorm(r, w + postNorm, width, norm.data() + t * width);
  }
#ifdef HYBRID_FUSED_GLU
  proj_glu(*weights, norm.data(), gate.data(), rows);
#else
  hybrid::Scratch up(rows * intermediate);
  proj_gateProj(*weights, norm.data(), gate.data(), rows);
  proj_upProj(*weights, norm.data(), up.data(), rows);
  for (int i = 0; i < validRows * intermediate; ++i)
    gate[i] = silu(gate[i]) * up[i];
#endif
#ifdef HYBRID_FUSED_GLU
  proj_downProj(*weights, gate.data(), out, rows, residual.data());
#else
  proj_downProj(*weights, gate.data(), out, rows);
  for (int i = 0; i < validRows * width; ++i)
    out[i] += residual[i];
#endif
  // Wire frames have fixed capacity. Define the tail without touching state.
  std::fill(out + validRows * width, out + rows * width, 0.0f);
  return 0;
}
#endif
} // namespace

extern "C" const BJ47KernelInfoV1 *bj47_kernel_info_v1() {
  return &kernelInfo.base;
}

#if HYBRID_ROLE == 1
extern "C" int32_t EMBED_PREFILL(const EmbedElement *table, const int64_t *ids,
                                 float *out) {
  return embed<chunk>(table, ids, out);
}
extern "C" int32_t EMBED_DECODE(const EmbedElement *table, const int64_t *ids,
                                float *out) {
  return embed<1>(table, ids, out);
}
#elif HYBRID_ROLE == 3
extern "C" int32_t lm_head_normalize(const BJ47WeightsV1 *w, const float *input,
                                     float *out, int rows) {
  if (!w || !w->f32 || !input || !out || rows < 1 || rows > chunk)
    return -1;
  for (int r = 0; r < rows; ++r)
    zeroCenteredNorm(input + r * width, w->f32 + finalNorm, width,
                     out + r * width);
  return 0;
}
extern "C" int32_t lm_head_linear(const BJ47WeightsV1 *w, const float *input,
                                  float *out, int rows) {
  if (!w || !input || !out || input == out || (rows != 1 && rows != 2) ||
      (kernelInfo.base.i8_elements ? !w->q4 : !w->f32))
    return -1;
  proj_lmHead(*w, input, out, rows);
  return 0;
}
extern "C" int32_t lm_head(const BJ47WeightsV1 *w, const float *input,
                           float *out) {
  if (!w || !w->f32 || !input || !out || input == out)
    return -1;
  if (kernelInfo.base.i8_elements && !w->q4)
    return -1;
  hybrid::Scratch norm(width);
  zeroCenteredNorm(input, w->f32 + finalNorm, width, norm.data());
  proj_lmHead(*w, norm.data(), out, 1);
  return 0;
}
extern "C" int32_t lm_head_prefill(const BJ47WeightsV1 *w, const float *input,
                                   float *out) {
  return lm_head(w, input, out);
}
extern "C" int32_t lm_head_decode(const BJ47WeightsV1 *w, const float *input,
                                  float *out) {
  return lm_head(w, input, out);
}
#elif HYBRID_ROLE == 6
extern "C" int32_t mtp_forward(const BJ47WeightsV1 *w, const float *embedding,
                               const float *hidden, float *k, float *v,
                               const BJ47FrameV1 *frame, float *out, int rows) {
  if (!w || !w->f32 || (kernelInfo.base.i8_elements && !w->q4) || !embedding ||
      !hidden || !frame || !frame->aux || !out ||
      (rows != 1 && rows != 2 && rows != chunk))
    return -1;
  const auto valid =
      static_cast<const BJ47FrameRowsV2 *>(frame->aux)->valid_rows;
  if (valid < 1 || valid > rows)
    return -2;
  hybrid::Scratch concat(rows * 2 * width), projected(rows * width);
  for (int r = 0; r < valid; ++r) {
    auto dst = concat.data() + r * 2 * width;
    zeroCenteredNorm(embedding + r * width, w->f32 + embeddingNorm, width, dst);
    zeroCenteredNorm(hidden + r * width, w->f32 + hiddenNorm, width,
                     dst + width);
  }
  proj_mtpFC(*w, concat.data(), projected.data(), rows);
  int rc = layer(w, projected.data(), k, v, frame, out, rows);
  if (rc)
    return rc;
  for (int r = 0; r < valid; ++r)
    zeroCenteredNorm(out + r * width, w->f32 + mtpNorm, width, out + r * width);
  return 0;
}
#else
extern "C" int32_t layer_verify_2(const BJ47WeightsV1 *w, const float *input,
                                  float *a, float *b, const BJ47FrameV1 *frame,
                                  float *out) {
  if (!frame || !frame->aux)
    return -1;
  const auto *aux = static_cast<const BJ47VerifyRows *>(frame->aux);
  if (aux->rows.valid_rows != 2)
    return -2;
#if HYBRID_ROLE == 4
  if (!aux->conv_snapshot || !aux->ssm_snapshot)
    return -1;
#endif
  return layer(w, input, a, b, frame, out, 2);
}
extern "C" int32_t layer_prefill(const BJ47WeightsV1 *w, const float *input,
                                 float *a, float *b, const BJ47FrameV1 *frame,
                                 float *out) {
  return layer(w, input, a, b, frame, out, chunk);
}
extern "C" int32_t layer_decode(const BJ47WeightsV1 *w, const float *input,
                                float *a, float *b, const BJ47FrameV1 *frame,
                                float *out) {
  return layer(w, input, a, b, frame, out, 1);
}
#endif

#if (HYBRID_ROLE == 5 || HYBRID_ROLE == 6)
static int32_t prepare(const float *inv, int64_t pos, float *cos, float *sin,
                       int rows) {
  if (!inv || !cos || !sin || cos == sin || pos < 0 || pos > context - rows)
    return -1;
  for (int t = 0; t < rows; ++t)
    for (int i = 0; i < rotaryDim / 2; ++i) {
      const float angle = static_cast<float>(pos + t) * inv[i];
      const auto index = t * rotaryDim + i;
      cos[index] = cos[index + rotaryDim / 2] = std::cos(angle);
      sin[index] = sin[index + rotaryDim / 2] = std::sin(angle);
    }
  return 0;
}
extern "C" int32_t prepare_prefill(const float *inv, int64_t pos, float *cos,
                                   float *sin, void *) {
  return prepare(inv, pos, cos, sin, chunk);
}
extern "C" int32_t prepare_decode(const float *inv, int64_t pos, float *cos,
                                  float *sin, void *) {
  return prepare(inv, pos, cos, sin, 1);
}
#endif
