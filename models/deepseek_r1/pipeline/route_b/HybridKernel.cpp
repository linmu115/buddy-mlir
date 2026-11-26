// The generated header specializes dimensions, weights and MLIR entry points
// for one checkpoint shape. All state buffers belong to the Route B caller.
#include "HybridConfig.h"

#include <algorithm>
#include <cmath>

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
void mix(const BJ47WeightsV1 &weights, const float *x, float *convState,
         float *ssmState, const BJ47FrameV1 &frame, int rows, int valid,
         float *out) {
  const float *w = weights.f32;
  hybrid::Scratch qkvBuffer(rows * convDim), z(rows * valueWidth),
      b(rows * valueHeads), a(rows * valueHeads), core(rows * valueWidth);
  proj_inProjQKV(weights, x, qkvBuffer.data(), rows);
  proj_inProjZ(weights, x, z.data(), rows);
  proj_inProjB(weights, x, b.data(), rows);
  proj_inProjA(weights, x, a.data(), rows);
  for (int t = 0; t < valid; ++t) {
    auto *qkv = qkvBuffer.data() + t * convDim;
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

    if (rows == 2 && t == 0)
      std::copy_n(
          convState, convDim * convKernel,
          static_cast<const BJ47VerifyRows *>(frame.aux)->conv_snapshot);
    auto *query = qkv;
    auto *key = qkv + keyWidth;

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

    for (int h = 0; h < valueHeads; ++h) {
      int i = t * valueHeads + h;
      b[i] = sigmoid(b[i]);
      float dt = a[i] + w[dtBias + h];
      a[i] =
          -std::exp(w[aLog + h]) * (dt > 20.f ? dt : std::log1p(std::exp(dt)));
    }
  }
#ifdef HYBRID_THREADS
#pragma omp parallel for schedule(static) num_threads(HYBRID_THREADS)
#endif
  for (uint64_t h = 0; h < valueHeads; ++h) {
    int kh = h / (valueHeads / keyHeads);
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
        deltaStep(state, q + t * convDim, k + t * convDim, v + t * convDim,
                  head + t * valueWidth, b[t * valueHeads + h],
                  std::exp(a[t * valueHeads + h]));
        if (rows == 2 && t == 0)
          std::copy_n(
              state, keyDim * valueDim,
              static_cast<const BJ47VerifyRows *>(frame.aux)->ssm_snapshot +
                  h * keyDim * valueDim);
      }
    } else
      hybrid::chunkGDN(q, k, v, convDim, a.data() + h, b.data() + h, valueHeads,
                       state, head, valueWidth, valid, keyDim, valueDim);
    for (int t = 0; t < valid; ++t) {
      auto *row = head + t * valueWidth;
      float norm = 1.f / std::sqrt(squareSum(row, valueDim) / valueDim + Eps);
      for (int j = 0; j < valueDim; ++j)
        row[j] = row[j] * norm * w[gatedNorm + j] *
                 silu(z[t * valueWidth + h * valueDim + j]);
    }
  }
  proj_mixerOut(weights, core.data(), out, rows);
}
#elif (HYBRID_ROLE == 5 || HYBRID_ROLE == 6)
void partialRotary(float *x, const float *cos, const float *sin) {
  for (int i = 0; i < rotaryDim / 2; ++i) {
    const float left = x[i], right = x[i + rotaryDim / 2];
    x[i] = left * cos[i] - right * sin[i];
    x[i + rotaryDim / 2] = right * cos[i] + left * sin[i];
  }
}

void mix(const BJ47WeightsV1 &weights, const float *x, float *keys,
         float *values, const BJ47FrameV1 &frame, int rows, int valid,
         float *out) {
  const auto *w = weights.f32;
  constexpr int qWidth = attentionHeads * headDim;
  constexpr int kvWidth = kvHeads * headDim;
  hybrid::Scratch query(rows * 2 * qWidth), keyRows(rows * kvWidth),
      valueRows(rows * kvWidth), cores(rows * qWidth),
      scores(frame.cache_position + valid);
  proj_qProj(weights, x, query.data(), rows);
  proj_kProj(weights, x, keyRows.data(), rows);
  proj_vProj(weights, x, valueRows.data(), rows);
  for (int row = 0; row < valid; ++row) {
    const int position = frame.cache_position + row;
    const auto *cos = frame.cos_rows + row * rotaryDim;
    const auto *sin = frame.sin_rows + row * rotaryDim;
    float *qAndGate = query.data() + row * 2 * qWidth;
    float *k = keyRows.data() + row * kvWidth;
    float *v = valueRows.data() + row * kvWidth;
    float *core = cores.data() + row * qWidth;
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
  }
  proj_mixerOut(weights, cores.data(), out, rows);
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
  mix(*weights, norm.data(), a, b, *frame, rows, validRows, residual.data());
  for (int t = 0; t < validRows; ++t) {
    float *r = residual.data() + t * width;
    for (int i = 0; i < width; ++i)
      r[i] += input[t * width + i];
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
  proj_downProj(*weights, gate.data(), out, rows);
  for (int i = 0; i < validRows * width; ++i)
    out[i] += residual[i];
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
