// The generated header specializes dimensions, weights and MLIR entry points
// for one checkpoint shape. All state buffers belong to the Route B caller.
#include "HybridConfig.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr unsigned FullAttention = 5;
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
// Qwen3_5GatedDeltaNet: process each real token once. This sequential prefill
// evaluates the same update used by HF's single-token recurrent fallback.
void mix(const float *w, const float *x, float *convState, float *ssmState,
         const BJ47FrameV1 &, int, float *out) {
  float qkv[convDim], z[valueWidth], b[valueHeads], a[valueHeads];
  project<convDim, width>(w + inProjQKV, x, qkv);
  project<valueWidth, width>(w + inProjZ, x, z);
  project<valueHeads, width>(w + inProjB, x, b);
  project<valueHeads, width>(w + inProjA, x, a);

  for (int c = 0; c < convDim; ++c) {
    auto *history = convState + c * convKernel;
    for (int j = 1; j < convKernel; ++j)
      history[j - 1] = history[j];
    history[convKernel - 1] = qkv[c];
    float sum = 0;
    for (int j = 0; j < convKernel; ++j)
      sum += history[j] * w[convWeight + c * convKernel + j];
    qkv[c] = silu(sum);
  }

  auto *query = qkv;
  auto *key = qkv + keyWidth;
  const auto *value = qkv + 2 * keyWidth;
  const float queryScale = 1.0f / std::sqrt(float(keyDim));
  for (int h = 0; h < keyHeads; ++h) {
    auto *q = query + h * keyDim;
    auto *k = key + h * keyDim;
    const float normQ = 1.0f / std::sqrt(squareSum(q, keyDim) + 1e-6f);
    const float normK = 1.0f / std::sqrt(squareSum(k, keyDim) + 1e-6f);
    for (int j = 0; j < keyDim; ++j) {
      q[j] = (q[j] * normQ) * queryScale;
      k[j] *= normK;
    }
  }

  float core[valueWidth];
  for (int h = 0; h < valueHeads; ++h) {
    const int keyHead = h / (valueHeads / keyHeads);
    const auto *q = query + keyHead * keyDim;
    const auto *k = key + keyHead * keyDim;
    const auto *v = value + h * valueDim;
    auto *state = ssmState + h * keyDim * valueDim;
    auto *row = core + h * valueDim;
    const float beta = sigmoid(b[h]);
    const float dt = a[h] + w[dtBias + h];
    const float softplus = dt > 20.0f ? dt : std::log1p(std::exp(dt));
    const float alpha = std::exp(-std::exp(w[aLog + h]) * softplus);
    deltaStep(state, q, k, v, row, beta, alpha);

    const float norm =
        1.0f / std::sqrt(squareSum(row, valueDim) / valueDim + Eps);
    for (int j = 0; j < valueDim; ++j)
      row[j] = (row[j] * norm) * w[gatedNorm + j] * silu(z[h * valueDim + j]);
  }
  project<width, valueWidth>(w + mixerOut, core, out);
}
#elif HYBRID_ROLE == 5
void partialRotary(float *x, const float *cos, const float *sin) {
  for (int i = 0; i < rotaryDim / 2; ++i) {
    const float left = x[i], right = x[i + rotaryDim / 2];
    x[i] = left * cos[i] - right * sin[i];
    x[i + rotaryDim / 2] = right * cos[i] + left * sin[i];
  }
}

void mix(const float *w, const float *x, float *keys, float *values,
         const BJ47FrameV1 &frame, int row, float *out) {
  constexpr int qWidth = attentionHeads * headDim;
  constexpr int kvWidth = kvHeads * headDim;
  const int position = frame.cache_position + row;
  const auto *cos = frame.cos_rows + row * rotaryDim;
  const auto *sin = frame.sin_rows + row * rotaryDim;
  float qAndGate[2 * qWidth], k[kvWidth], v[kvWidth], core[qWidth];
  project<2 * qWidth, width>(w + qProj, x, qAndGate);
  project<kvWidth, width>(w + kProj, x, k);
  project<kvWidth, width>(w + vProj, x, v);

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

  const float scale = 1.0f / std::sqrt(float(headDim));
  for (int h = 0; h < attentionHeads; ++h) {
    const int kvHead = h / (attentionHeads / kvHeads);
    auto *q = qAndGate + h * 2 * headDim;
    const auto *gate = q + headDim;
    zeroCenteredNorm(q, w + qNorm, headDim, q);
    partialRotary(q, cos, sin);

    const auto *headKeys = keys + size_t(kvHead) * context * headDim;
    const auto *headValues = values + size_t(kvHead) * context * headDim;
    float scores[context];
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

    auto *result = core + h * headDim;
    std::fill_n(result, headDim, 0.0f);
    for (int t = 0; t <= position; ++t) {
      const float p = scores[t] / denominator;
      for (int i = 0; i < headDim; ++i)
        result[i] += p * headValues[t * headDim + i];
    }
    for (int i = 0; i < headDim; ++i)
      result[i] *= sigmoid(gate[i]);
  }
  project<width, qWidth>(w + mixerOut, core, out);
}
#endif

#if HYBRID_ROLE == 4 || HYBRID_ROLE == 5
int32_t layer(const BJ47WeightsV1 *weights, const float *input, float *a,
              float *b, const BJ47FrameV1 *frame, float *out, int rows) {
  if (!weights || !weights->f32 || !input || !a || !b || a == b || !frame ||
      !frame->aux || !out || input == out)
    return -1;
  if (kernelInfo.base.role == FullAttention &&
      (!frame->cos_rows || !frame->sin_rows))
    return -1;

  const auto validRows =
      static_cast<const BJ47FrameRowsV2 *>(frame->aux)->valid_rows;
  if (validRows < 1 || validRows > rows || frame->cache_position < 0 ||
      frame->cache_position > context - validRows)
    return -2;

  const auto *w = weights->f32;
  for (int t = 0; t < validRows; ++t) {
    const auto *x = input + t * width;
    auto *y = out + t * width;
    float norm[width], residual[width], gate[intermediate], up[intermediate];
    zeroCenteredNorm(x, w + inputNorm, width, norm);
    mix(w, norm, a, b, *frame, t, residual);
    for (int i = 0; i < width; ++i)
      residual[i] += x[i];
    zeroCenteredNorm(residual, w + postNorm, width, norm);
    project<intermediate, width>(w + gateProj, norm, gate);
    project<intermediate, width>(w + upProj, norm, up);
    for (int i = 0; i < intermediate; ++i)
      gate[i] = silu(gate[i]) * up[i];
    project<width, intermediate>(w + downProj, gate, y);
    for (int i = 0; i < width; ++i)
      y[i] += residual[i];
  }
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
extern "C" int32_t lm_head(const BJ47WeightsV1 *w, const float *input,
                           float *out) {
  if (!w || !w->f32 || !input || !out || input == out)
    return -1;
  float norm[width];
  zeroCenteredNorm(input, w->f32 + finalNorm, width, norm);
  project<vocab, width>(w->f32 + lmHead, norm, out);
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
#else
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

#if HYBRID_ROLE == 5
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
