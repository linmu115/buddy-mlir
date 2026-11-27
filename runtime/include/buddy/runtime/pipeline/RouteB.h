#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t abi_version, struct_bytes, role; /* 1=embed, 2=decoder, 3=head */
  uint32_t chunk, context, width, kv_heads, head_dim, vocab_size;
  uint64_t f32_elements, i8_elements, rope_elements;
  uint64_t aux_prefill_bytes, aux_decode_bytes, required_alignment;
  uint8_t layout_id[32], shape_id[32]; /* SHA-256 */
} BJ47KernelInfoV1;

/* Hybrid DSOs return this prefix from bj47_kernel_info_v1. The legacy
 * ABI-2 head still returns a descriptor of exactly sizeof(BJ47KernelInfoV1).
 * Roles 4 and 5 are Gated DeltaNet and gated full attention, respectively. */
typedef struct {
  BJ47KernelInfoV1 base;
  uint64_t state_a_elements, state_b_elements, f16_elements;
  uint32_t rotary_dim, embed_dtype; /* 0=not embed, 1=f32, 2=f16 */
} BJ47KernelInfoV2;

typedef struct {
  int64_t valid_rows;
} BJ47FrameRowsV2;

typedef struct {
  const float *f32;
  const int8_t *q4;
} BJ47WeightsV1;

typedef struct {
  int64_t cache_position;
  const float *cos_rows;
  const float *sin_rows;
  const void *aux; /* same decoder DSO defines and consumes it */
} BJ47FrameV1;

/* Every DSO exports this on its own dlopen handle. Library owns the result. */
const BJ47KernelInfoV1 *bj47_kernel_info_v1(void);

/* embed.so: rows fixed to chunk or 1 by the entry */
int32_t embed_prefill(const float *table, const int64_t *ids, float *out);
int32_t embed_decode(const float *table, const int64_t *ids, float *out);
int32_t embed_prefill_f16(const uint16_t *table, const int64_t *ids,
                          float *out);
int32_t embed_decode_f16(const uint16_t *table, const int64_t *ids, float *out);

/* decoder_layer.so: caller owns cos, sin, aux, K, V and out */
int32_t prepare_prefill(const float *inv_freq, int64_t pos, float *cos_rows,
                        float *sin_rows, void *aux);
int32_t prepare_decode(const float *inv_freq, int64_t pos, float *cos_rows,
                       float *sin_rows, void *aux);
int32_t layer_prefill(const BJ47WeightsV1 *w, const float *in, float *k,
                      float *v, const BJ47FrameV1 *frame, float *out);
int32_t layer_decode(const BJ47WeightsV1 *w, const float *in, float *k,
                     float *v, const BJ47FrameV1 *frame, float *out);

/* Dynamic verify accepts 2..4 rows. Each GDN snapshot array holds rows-1
 * complete states in prefix order. Attention uses the committed position. */
typedef struct {
  BJ47FrameRowsV2 rows;
  float *conv_snapshot, *ssm_snapshot;
} BJ47VerifyRows;
int32_t layer_verify_2(const BJ47WeightsV1 *, const float *, float *, float *,
                       const BJ47FrameV1 *, float *);
int32_t layer_verify(const BJ47WeightsV1 *, const float *, float *, float *,
                     const BJ47FrameV1 *, float *);
int32_t lm_head_normalize(const BJ47WeightsV1 *, const float *, float *, int);
int32_t lm_head_linear(const BJ47WeightsV1 *, const float *, float *, int);
int32_t mtp_forward(const BJ47WeightsV1 *, const float *embedding,
                    const float *normalized_hidden, float *keys, float *values,
                    const BJ47FrameV1 *, float *, int rows);

/* lm_head.so: row points at one contiguous H-vector; logits has V floats */
int32_t lm_head(const BJ47WeightsV1 *w, const float *row, float *logits);

/* The info descriptor stays V1. Head kernel ABI 2 has two numerical paths.
 * Select by phase; lm_head is a compatibility alias for lm_head_decode.
 * Caller buffers must be contiguous, disjoint and aligned to
 * required_alignment. A successful prepare fills both [R,D] arrays. No aux
 * allocation is required. */
int32_t lm_head_decode(const BJ47WeightsV1 *, const float *, float *);
int32_t lm_head_prefill(const BJ47WeightsV1 *, const float *, float *);
#ifdef __cplusplus
}
#endif
