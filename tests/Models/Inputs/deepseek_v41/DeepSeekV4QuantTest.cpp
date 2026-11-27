// Compare the actual native integer decoder and K3 tile matvec with the
// independent fixture produced by deepseek_v4_quant_test.py.
#include "buddy/runtime/DeepSeekV4.h"
#ifdef BUDDY_TEST_MLIR
#include "../../../../models/deepseek_r1/pipeline/route_b/Shim.h"
extern "C" void _mlir_ciface_k3_q4_plain_m1_k256_n384(Descriptor<2> *,
                                                      Descriptor<2> *,
                                                      Descriptor<1> *);
#endif
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace buddy::v4;
template <class T>
std::vector<T> read(const std::string &dir, const char *file, int n) {
  std::ifstream f(dir + "/" + file, std::ios::binary);
  std::vector<T> result(n);
  if (!f.read(reinterpret_cast<char *>(result.data()), n * sizeof(T)))
    throw std::runtime_error(file);
  return result;
}
// Compare the unscaled decoder before FP arithmetic can canonicalize a NaN.
void checkFP8(const std::array<uint8_t, 256> &codes) {
  Vec decoded(256);
#if defined(__riscv_v_intrinsic)
  for (size_t i = 0; i < codes.size();) {
    size_t vl = __riscv_vsetvl_e32m4(codes.size() - i);
    __riscv_vse32_v_f32m4(decoded.data() + i, loadE4M3(codes.data() + i, vl),
                          vl);
    i += vl;
  }
#else
  std::copy(e4m3Table().begin(), e4m3Table().end(), decoded.begin());
#endif
  for (int i = 0; i < 256; ++i) {
    float ref = e4m3(codes[i]);
    if (std::memcmp(&ref, &decoded[i], sizeof(float)))
      throw std::runtime_error("E4M3 bit mismatch at " + std::to_string(i));
  }
  constexpr int k = 259, n = 133;
  Vec x(k), y(n);
  std::vector<uint8_t> weights(n * k);
  for (size_t i = 0; i < weights.size(); ++i) {
    uint8_t code = (17 * i + i / k) % 256;
    weights[i] = (code & 127) == 127 ? 0 : code;
  }
  for (int i = 0; i < k; ++i)
    x[i] = float(i % 19 - 9) / 13;
  for (int block : {32, 128}) {
    int groups = (k + block - 1) / block;
    std::vector<uint8_t> scales(((n + block - 1) / block) * groups);
    for (size_t i = 0; i < scales.size(); ++i)
      scales[i] = 122 + i % 9;
    fp8MatVec(x.data(), weights.data(), scales.data(), k, n, block, y.data());
    double err = 0, den = 0;
    for (int r = 0; r < n; ++r) {
      double ref = 0;
      for (int t = 0; t < k; ++t)
        ref += double(x[t]) * e4m3(weights[r * k + t]) *
               ue8m0(scales[r / block * groups + t / block]);
      err += (y[r] - ref) * (y[r] - ref);
      den += ref * ref;
    }
    if (!(std::sqrt(err / den) < 1e-5))
      throw std::runtime_error("FP8 matvec mismatch at block " +
                               std::to_string(block));
  }
}
void checkFP4Edges() {
  // Odd row stride, a short final block, the smallest UE8M0 scale and NaN.
  constexpr int k = 35, n = 3;
  std::array<uint8_t, n *((k + 1) / 2)> weights{};
  for (int r = 0; r < n; ++r)
    weights[r * ((k + 1) / 2) + (k - 1) / 2] = 0xf1 + r;
  std::array<uint8_t, n * 2> scales{127, 0, 127, 127, 127, 255};
  Vec x(k, 0), out(n);
  x.back() = 1;
  fp4MatVec(x.data(), weights.data(), scales.data(), k, n, out.data());
  if (out[0] != 0x1p-128f || out[1] != 1.f || !std::isnan(out[2]))
    throw std::runtime_error("FP4 edge block mismatch");
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  std::string dir = argv[1];
  std::array<uint8_t, 256> codes{};
  std::iota(codes.begin(), codes.end(), 0);
  checkFP8(codes);
  checkFP4Edges();
  for (int block : {32, 128}) {
    uint8_t scale[8] = {124, 125, 126, 127, 128, 129, 130, 131};
    Vec out(256);
    decodeFP8(codes.data(), scale, 256, block, out.data());
    auto ref = read<float>(dir, block == 32 ? "fp8_32" : "fp8_128", 256);
    for (int i = 0; i < 256; i++)
      if (!(std::isnan(out[i]) && std::isnan(ref[i])) &&
          std::memcmp(&out[i], &ref[i], 4))
        return 1;
  }
  constexpr int n = 384, k = 256;
  auto w = read<uint8_t>(dir, "codes", n * k / 2),
       s = read<uint8_t>(dir, "scales", n * k / 32);
  auto packed = read<uint8_t>(dir, "tiles", n * k / 32 * 18);
  auto x = read<float>(dir, "x", k),
       expected = read<float>(dir, "decoded", n * k);
  auto ue = read<float>(dir, "e2m1_all", 256 * 16);
  for (int scale = 0; scale < 256; scale++)
    for (int code = 0; code < 16; code++) {
      float a = scaledE2M1(code, scale), b = ue[scale * 16 + code];
      if (!(std::isnan(a) && std::isnan(b)) && std::memcmp(&a, &b, 4))
        return 1;
    }
  Vec y(n), native(n), row(k);
  fp4MatVec(x.data(), w.data(), s.data(), k, n, native.data());
  w4linear(x.data(), packed.data(), k, n, y.data());
#ifdef BUDDY_TEST_MLIR
  double mlirDiff = 0;
  Descriptor<2> out(nullptr, {1, n}), input(x.data(), {1, k});
  Descriptor<1> tiles(packed.data(), {int64_t(packed.size())});
  _mlir_ciface_k3_q4_plain_m1_k256_n384(&out, &input, &tiles);
  auto mlir = static_cast<const float *>(out.aligned) + out.offset;
  for (int i = 0; i < n; i++)
    mlirDiff = std::max(mlirDiff, double(std::abs(mlir[i] - y[i])));
  std::free(const_cast<void *>(out.allocated));
  if (!(mlirDiff < 1e-4))
    return 1;
  std::cout << "{\"mlir_w4_max_abs\":" << mlirDiff << "}" << std::endl;
#endif
  double max = 0, maxRel = 0, den = 0, err = 0, nativeErr = 0, weightErr = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j += 32)
      decodeFP4(&w[(i * k + j) / 2], s[(i * k + j) / 32], row.data() + j, 16);
    if (std::memcmp(row.data(), expected.data() + i * k, k * 4))
      return 1;
    double ref = 0, repacked = 0;
    for (int j = 0; j < k; ++j) {
      ref += double(x[j]) * expected[i * k + j];
      auto tile = packed.data() + ((i / 128) * (k / 32) + j / 32) * 18 * 128;
      int nibble = (tile[(j % 32 / 2) * 128 + i % 128] >> (4 * (j % 2))) & 15;
      _Float16 weightScale;
      std::memcpy(&weightScale, tile + 16 * 128 + 2 * (i % 128), 2);
      repacked += double(x[j]) * ((nibble ^ 8) - 8) * float(weightScale);
    }
    nativeErr += (native[i] - ref) * (native[i] - ref);
    weightErr += (repacked - ref) * (repacked - ref);
    double delta = std::abs(y[i] - ref);
    max = std::max(max, delta);
    if (ref != 0)
      maxRel = std::max(maxRel, delta / std::abs(ref));
    err += delta * delta;
    den += ref * ref;
  }
  double nativeL2 = std::sqrt(nativeErr / den);
  if (!(nativeL2 < 1e-5))
    throw std::runtime_error("native FP4 relative L2 " +
                             std::to_string(nativeL2));
  auto mx = read<float>(dir, "expert.x", 128),
       router = read<float>(dir, "router.f32", 4 * 128);
  auto dense = read<float>(dir, "expert.f32", 5 * 3 * 128 * 128);
  auto tilesFF = read<uint8_t>(dir, "expert.w4", 5 * 3 * 128 * 128 / 32 * 18);
  double ffnError[2] = {};
  Vec scores(4);
  linear(mx.data(), router.data(), 128, 4, scores.data());
  float bias[] = {2, -1, -.5, 1};
  for (int mode = 0; mode < 2; mode++) {
    auto project = [&](int id, int branch, const Vec &input) {
      int off = (id * 3 + (branch == 1 ? 0 : branch == 3 ? 1 : 2)) * 128 * 128;
      Vec out(128);
      if (mode)
        w4linear(input.data(), tilesFF.data() + off / 32 * 18, 128, 128,
                 out.data());
      else
        linear(input.data(), dense.data() + off, 128, 128, out.data());
      return out;
    };
    auto actual = moeV4(mx, scores, bias, 2, 1, 1.5f, 10, project,
                        [](float v) { return v; });
    auto oracle = read<float>(dir, mode ? "ffn.w4" : "ffn.f32", 128);
    for (int j = 0; j < 128; j++)
      ffnError[mode] =
          std::max(ffnError[mode], double(std::abs(actual[j] - oracle[j])));
    if (!(ffnError[mode] < 1e-4))
      return 1;
  }
  std::cout << "{\"moe_f32_max_abs\":" << ffnError[0]
            << ",\"moe_w4_max_abs\":" << ffnError[1]
            << ",\"decoded_bit_diff\":0,\"integer_cases\":4096,\"outputs\":384,"
               "\"dot_max_abs\":"
            << max << ",\"e4m3_bit_matches\":256,\"fp4_native_relative_l2\":"
            << nativeL2
            << ",\"w4_weight_only_relative_l2\":" << std::sqrt(weightErr / den)
            << ",\"dot_max_relative_nonzero\":" << maxRel
            << ",\"dot_relative_l2\":" << std::sqrt(err / den) << "}\n";
}
