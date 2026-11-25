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

#include "buddy/runtime/pipeline/ActivationWire.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>

using namespace buddy::runtime::pipeline;

static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("activation wire test failed");
}

// Construct each positive half value mathematically, independently of the
// codec.
static float reference(int h) {
  int exp = h >> 10;
  return exp ? std::ldexp(float(1024 + (h & 1023)), exp - 25)
             : std::ldexp(float(h), -24);
}

static void checkEncode(float value, int expected) {
  uint16_t wire;
  encodeInto(&value, 1, &wire, 1);
  require(wire == expected);
}

template <class E, class F> static void rejects(F f) {
  try {
    f();
  } catch (const E &) {
    return;
  }
  throw std::runtime_error("missing expected codec error");
}

// Exercise multiple vectors and a partial final vector. Verify RNE even when
// the ambient rounding mode differs, and preserve signed zero on widening.
static void checkBatches() {
  constexpr size_t count = 513;
  std::array<float, count> values, decoded;
  std::array<uint16_t, count> wire, expected;
  for (size_t i = 0; i < count; ++i) {
    int h = i % 1024;
    values[i] = (reference(h) + reference(h + 1)) * 0.5f;
    expected[i] = h + (h & 1);
    if (i % 2 == 0) {
      values[i] = -values[i];
      expected[i] |= 0x8000;
    }
  }
  const int original = std::fegetround();
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(std::fesetround(mode) == 0);
    encodeInto(values.data(), count, wire.data(), count);
    require(wire == expected && std::fegetround() == mode);
  }
  require(std::fesetround(original) == 0);
  decodeInto(wire.data(), count, decoded.data(), count);
  for (size_t i = 0; i < count; ++i) {
    float f = reference(expected[i] & 0x7fff);
    if (expected[i] & 0x8000)
      f = -f;
    require(std::memcmp(&f, &decoded[i], sizeof(f)) == 0);
  }

  // An invalid lane must leave the failing element and suffix untouched.
  constexpr size_t bad = 259;
  values[bad] = 65520.0f;
  values[bad + 1] = NAN;
  wire.fill(0xffff);
  rejects<std::overflow_error>(
      [&] { encodeInto(values.data(), count, wire.data(), count); });
  require(std::equal(wire.begin(), wire.begin() + bad, expected.begin()));
  require(std::all_of(wire.begin() + bad, wire.end(),
                      [](uint16_t h) { return h == 0xffff; }));
  wire = expected;
  wire[bad] = 0x7c01;
  decoded.fill(42.0f);
  rejects<std::domain_error>(
      [&] { decodeInto(wire.data(), count, decoded.data(), count); });
  for (size_t i = 0; i < bad; ++i) {
    float f = reference(expected[i] & 0x7fff);
    if (expected[i] & 0x8000)
      f = -f;
    require(std::memcmp(&f, &decoded[i], sizeof(f)) == 0);
  }
  require(std::all_of(decoded.begin() + bad, decoded.end(),
                      [](float f) { return f == 42.0f; }));
}

int main() {
  checkBatches();
  for (int h = 0; h <= 0x7bff; ++h) {
    for (int sign : {0, 0x8000}) {
      const uint16_t half = h | sign;
      const float value = sign ? -reference(h) : reference(h);
      float decoded;
      decodeInto(&half, 1, &decoded, 1);
      require(std::memcmp(&decoded, &value, sizeof(value)) == 0);
      checkEncode(value, half);
      const float next = reference(h + 1);
      const float mid = (reference(h) + next) * 0.5f;
      const float below = std::nextafter(mid, 0.0f);
      checkEncode(sign ? -below : below, half);
      if (h == 0x7bff) {
        rejects<std::overflow_error>(
            [&] { checkEncode(sign ? -mid : mid, 0); });
        continue;
      }
      checkEncode(sign ? -mid : mid, (h + (h & 1)) | sign);
      const float above = std::nextafter(mid, next);
      checkEncode(sign ? -above : above, (h + 1) | sign);
    }
  }
  for (float f : {NAN, INFINITY, -INFINITY})
    rejects<std::domain_error>([&] { checkEncode(f, 0); });
  rejects<std::length_error>([] { encodeInto(nullptr, 1, nullptr, 0); });
  rejects<std::length_error>([] { decodeInto(nullptr, 1, nullptr, 0); });
  const auto compressed = resolveWireSpec(ActivationType::F32, WireDtype::F16);
  require(compressed.compressed() && checkedWireBytes(compressed, 3) == 6);
  require(!resolveWireSpec(ActivationType::F16, WireDtype::F16).compressed());
  require(
      !resolveWireSpec(ActivationType::F32, WireDtype::Native).compressed());
  rejects<std::length_error>([&] { checkedWireBytes(compressed, SIZE_MAX); });
  int n;
  require(!computeElementCount(0, 1, n));
  require(!computeElementCount(std::numeric_limits<int>::max(), 2, n));
  require(computeElementCount(3, 5, n) && n == 15);
  std::cout << "PASS activation wire codec\n";
}
