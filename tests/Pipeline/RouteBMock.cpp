#include "buddy/runtime/pipeline/RouteB.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {
constexpr unsigned role = MOCK_ROLE;
const BJ47KernelInfoV1 info = {role == 3 ? 2u : 1u,
                               sizeof(BJ47KernelInfoV1),
                               role,
                               64,
                               128,
                               16,
                               1,
                               16,
                               151936,
                               1,
                               role == 1 ? 0u : 1u,
                               role == 2 ? 8u : 0u,
                               0,
                               0,
                               64,
                               {},
                               {}};
int prepares = 0;
int calls = 0;
int prepare(int64_t pos, float *cos, float *sin) {
  const char *count = std::getenv("CHECK_ROUTE_B_LAYERS");
  if (prepares && count && calls != std::atoi(count))
    return -4;
  ++prepares;
  calls = 0;
  *cos = float(pos);
  *sin = float(prepares);
  return 0;
}
int layer(const BJ47WeightsV1 *w, const float *in, float *k, float *v,
          const BJ47FrameV1 *frame, float *out, int rows) {
  if (in == out || k == v || *frame->cos_rows != frame->cache_position ||
      *frame->sin_rows != prepares || reinterpret_cast<uintptr_t>(in) % 64 ||
      reinterpret_cast<uintptr_t>(out) % 64)
    return -1;
  int global = *w->q4;
  if (*w->f32 != global || in[0] != global)
    return -2;
  // Each KV position starts empty and retains the first prefill write.
  const int pos = frame->cache_position;
  if (std::getenv("CHECK_ROUTE_B_BATCH") &&
      (k[pos] || v[pos] || (pos && (k[0] != global + 1 || v[0] != global + 2))))
    return -6;
  // The local-stage probe alternates phases, keeping earlier KV intact.
  if (std::getenv("CHECK_ROUTE_B_KV") && frame->cache_position &&
      (k[0] != global + 1 || v[0] != global + 2))
    return -3;
  // Tag each frame at its source and check the tag across every layer/rank.
  // Reusing a still-pending send buffer can deliver a later frame's tag.
  const bool checkFrame = std::getenv("CHECK_ROUTE_B_FRAME");
  if (checkFrame && global && in[rows * 16 - 1] != frame->cache_position)
    return -5;
  ++calls;
  k[frame->cache_position] = global + 1;
  v[frame->cache_position] = global + 2;
  std::fill_n(out, rows * 16, float(global + 1));
  if (checkFrame)
    out[rows * 16 - 1] = float(frame->cache_position);
  return 0;
}
int head(const float *in, float *out, int token) {
  if (in[0] != 28)
    return -1;
  std::fill_n(out, 151936, 0.f);
  if (std::getenv("CHECK_ROUTE_B_BATCH")) {
    const int pos = static_cast<int>(in[15]);
    if (pos == 12)
      token = 151643;
    else
      token = 100 + pos;
  }
  out[token] = 1.f;
  return 0;
}
} // namespace
extern "C" const BJ47KernelInfoV1 *bj47_kernel_info_v1() { return &info; }
#if MOCK_ROLE == 1
extern "C" int32_t embed_prefill(const float *, const int64_t *, float *out) {
  std::fill_n(out, 64 * 16, 0.f);
  return 0;
}
extern "C" int32_t embed_decode(const float *, const int64_t *, float *out) {
  std::fill_n(out, 16, 0.f);
  return 0;
}
#elif MOCK_ROLE == 2
#define DEFINE_PHASE(phase, rows)                                              \
  extern "C" int32_t prepare_##phase(const float *, int64_t pos, float *c,     \
                                     float *s, void *) {                       \
    return prepare(pos, c, s);                                                 \
  }                                                                            \
  extern "C" int32_t layer_##phase(const BJ47WeightsV1 *w, const float *i,     \
                                   float *k, float *v, const BJ47FrameV1 *f,   \
                                   float *o) {                                 \
    return layer(w, i, k, v, f, o, rows);                                      \
  }
DEFINE_PHASE(prefill, 64)
#ifndef MOCK_INCOMPLETE
DEFINE_PHASE(decode, 1)
#else
extern "C" int32_t prepare_decode(const float *, int64_t, float *, float *,
                                  void *) {
  return 0;
}
#endif
#else
extern "C" int32_t lm_head_prefill(const BJ47WeightsV1 *, const float *i,
                                   float *o) {
  return head(i, o, 42);
}
extern "C" int32_t lm_head_decode(const BJ47WeightsV1 *, const float *i,
                                  float *o) {
  return head(i, o, 43);
}
#endif
