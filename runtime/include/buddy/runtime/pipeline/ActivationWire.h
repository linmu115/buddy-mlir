//===----------------------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_PIPELINE_ACTIVATION_WIRE_H
#define BUDDY_RUNTIME_PIPELINE_ACTIVATION_WIRE_H

#include "buddy/runtime/pipeline/Stage.h"

#if defined(__riscv_zvfh) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

#include <cstring>
#include <limits>
#include <string>

namespace buddy::runtime::pipeline {

enum class Transport { Blocking, Async };
enum class WireDtype { Native, F16 };

struct WireSpec {
  ActivationType dtype;
  int version;
  bool compressed() const { return version == 2; }
};

inline WireSpec resolveWireSpec(ActivationType compute, WireDtype requested) {
  if (compute == ActivationType::F32 && requested == WireDtype::F16)
    return {ActivationType::F16, 2};
  return {compute, 1};
}

inline bool computeElementCount(int rows, int width, int &elements) {
  if (rows <= 0 || width <= 0 || rows > std::numeric_limits<int>::max() / width)
    return false;
  elements = rows * width;
  return true;
}

inline size_t checkedWireBytes(WireSpec spec, size_t elements) {
  const size_t stride = spec.dtype == ActivationType::F16 ? 2 : 4;
  if (elements > std::numeric_limits<size_t>::max() / stride)
    throw std::length_error("activation wire byte size overflow");
  return elements * stride;
}

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "activation wire requires IEEE binary32");

// Round an unsigned significand once, including ties at exponent boundaries.
inline uint32_t roundShiftEven(uint32_t value, unsigned shift) {
  const uint32_t quotient = value >> shift;
  const uint32_t halfway = uint32_t{1} << (shift - 1);
  const uint32_t remainder = value & ((halfway << 1) - 1);
  return quotient +
         (remainder > halfway || (remainder == halfway && (quotient & 1)));
}

// Native transfers use the caller's original bytes directly. Only v2 needs
// scratch storage. Report the first rejected element to the transport caller.
inline void encodeInto(const float *src, size_t count, uint16_t *dst,
                       size_t capacity) {
  if (count > capacity)
    throw std::length_error("activation encode buffer too small");
  size_t i = 0;
#if defined(__riscv_zvfh) && defined(__riscv_v_intrinsic)
  while (i < count) {
    const size_t vl = __riscv_vsetvl_e32m4(count - i);
    const auto values = __riscv_vle32_v_f32m4(src + i, vl);
    const auto bits = __riscv_vreinterpret_v_f32m4_u32m4(values);
    const auto magnitude = __riscv_vand_vx_u32m4(bits, 0x7fffffff, vl);
    // 65520 is the first magnitude which rounds to infinity under RNE.
    // Replay a rejected vector below to preserve the first error and prefix.
    if (__riscv_vfirst_m_b8(
            __riscv_vmsgeu_vx_u32m4_b8(magnitude, 0x477ff000, vl), vl) >= 0)
      break;
    const auto halves =
        __riscv_vfncvt_f_f_w_f16m2_rm(values, __RISCV_FRM_RNE, vl);
    __riscv_vse16_v_u16m2(dst + i, __riscv_vreinterpret_v_f16m2_u16m2(halves),
                          vl);
    i += vl;
  }
#endif
  for (; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, src + i, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000;
    const int exp = int((bits >> 23) & 0xff) - 127;
    const uint32_t significand = (bits & 0x7fffff) | 0x800000;
    uint32_t magnitude = 0;
    if (exp == 128)
      throw std::domain_error("non-finite activation at element " +
                              std::to_string(i));
    if (exp >= -14) {
      // The rounded significand includes its implicit leading 1.
      magnitude = ((exp + 14) << 10) + roundShiftEven(significand, 13);
      if (magnitude >= 0x7c00)
        throw std::overflow_error("activation overflows binary16 at element " +
                                  std::to_string(i));
    } else if (exp >= -25) {
      magnitude = roundShiftEven(significand, unsigned(-exp - 1));
    }
    dst[i] = static_cast<uint16_t>(sign | magnitude);
  }
}

inline void decodeInto(const uint16_t *src, size_t count, float *dst,
                       size_t capacity) {
  if (count > capacity)
    throw std::length_error("activation decode buffer too small");
  size_t i = 0;
#if defined(__riscv_zvfh) && defined(__riscv_v_intrinsic)
  while (i < count) {
    const size_t vl = __riscv_vsetvl_e16m2(count - i);
    const auto bits = __riscv_vle16_v_u16m2(src + i, vl);
    const auto magnitude = __riscv_vand_vx_u16m2(bits, 0x7fff, vl);
    if (__riscv_vfirst_m_b8(__riscv_vmsgeu_vx_u16m2_b8(magnitude, 0x7c00, vl),
                            vl) >= 0)
      break;
    const auto halves = __riscv_vreinterpret_v_u16m2_f16m2(bits);
    __riscv_vse32_v_f32m4(dst + i, __riscv_vfwcvt_f_f_v_f32m4(halves, vl), vl);
    i += vl;
  }
#endif
  for (; i < count; ++i) {
    const uint32_t sign = uint32_t(src[i] & 0x8000) << 16;
    int exp = (src[i] >> 10) & 31;
    uint32_t fraction = src[i] & 1023;
    if (exp == 31)
      throw std::domain_error("non-finite wire activation at element " +
                              std::to_string(i));
    if (exp == 0 && fraction != 0) {
      exp = 1;
      while (!(fraction & 1024)) {
        fraction <<= 1;
        --exp;
      }
      fraction &= 1023;
    }
    const uint32_t bits =
        (src[i] & 0x7fff) == 0
            ? sign
            : sign | (uint32_t(exp + 112) << 23) | (fraction << 13);
    std::memcpy(dst + i, &bits, sizeof(bits));
  }
}

} // namespace buddy::runtime::pipeline

#endif
