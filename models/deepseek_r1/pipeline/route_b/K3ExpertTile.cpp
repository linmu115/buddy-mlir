// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>

namespace {
// The IME reads two registers of eight interleaved activation rows.
// Its outputs are four 8x8 row-major tiles per N32 block.
constexpr int Rows = 8, Columns = 128, Group = 32, GroupsPerPass = 16;
struct Activations {
  int8_t q[Rows][Group];
  float scales[Rows];
};
struct Weights {
  int8_t packed[Group * Columns / 2];
  _Float16 scales[Columns];
};
struct alignas(128) Scratch {
  Activations acts[2][GroupsPerPass];
  Weights weights[GroupsPerPass];
  float sums[2][Columns / Rows][Rows][Rows];
};
static_assert(sizeof(Activations) == 288 && sizeof(Weights) == 2304);

extern "C" void buddy_k3_transpose_groups(const int8_t *, Weights *, long);
extern "C" void buddy_k3_multiply32(const int8_t *, const Activations *,
                                    const _Float16 *, float *, long, long);
extern "C" int64_t _mlir_ciface_buddy_spacemit_tcm_here(int64_t, int64_t);
extern "C" int32_t __kmpc_global_thread_num(void *);
} // namespace

// Called inside a single, non-nested parallel workshare. The Route B pool
// serializes teams. A logical worker owns the same private slot throughout
// this call, including when two workers share a physical TCM region.
extern "C" void _mlir_ciface_buddy_route_b_expert_tile(
    const int8_t *weight, const int8_t *q, const float *scales,
    const int64_t *pairs, float *out, int64_t k, int64_t count,
    int64_t pairStride) {
  Scratch fallback;
  auto *region = reinterpret_cast<Scratch *>(
      _mlir_ciface_buddy_spacemit_tcm_here(4, 8 * sizeof(Scratch)));
  int worker = __kmpc_global_thread_num(nullptr);
  Scratch &scratch =
      region && worker >= 0 && worker < 8 ? region[worker] : fallback;
  const int blocks = (count + Rows - 1) / Rows, groups = k / Group;
  int64_t row[2 * Rows];
  for (int i = 0; i < blocks * Rows; ++i)
    row[i] = pairs[i < count ? i : count - 1] / pairStride;

  for (int first = 0; first < groups; first += GroupsPerPass) {
    int remaining = groups - first;
    int step = remaining < GroupsPerPass ? remaining : GroupsPerPass;
    buddy_k3_transpose_groups(weight + sizeof(Weights) * first, scratch.weights,
                              step);
    for (int b = 0; b < blocks; ++b) {
      for (int r = 0; r < Rows; ++r) {
        const auto source = row[b * Rows + r];
        for (int j = 0; j < step; ++j) {
          auto &a = scratch.acts[b][j];
          std::memcpy(a.q[r], q + source * k + (first + j) * Group, Group);
          a.scales[r] = scales[source * groups + first + j];
        }
      }
      for (int col = 0; col < Columns; col += 32)
        buddy_k3_multiply32(scratch.weights[0].packed + col * 16,
                            scratch.acts[b], scratch.weights[0].scales + col,
                            &scratch.sums[b][col / Rows][0][0], step,
                            first == 0);
    }
  }
  for (int i = 0; i < count; ++i)
    for (int col = 0; col < Columns; col += Rows)
      std::memcpy(out + i * Columns + col,
                  scratch.sums[i / Rows][col / Rows][i % Rows],
                  Rows * sizeof(float));
}
