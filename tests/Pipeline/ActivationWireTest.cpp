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
#include <cmath>
#include <iostream>

using namespace buddy::runtime::pipeline;

static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("activation wire test failed");
}

// Construct each positive half value mathematically, independently of the codec.
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

template<class E, class F> static void rejects(F f) {
  try { f(); } catch (const E &) { return; }
  throw std::runtime_error("missing expected codec error");
}

int main() {
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
        rejects<std::overflow_error>([&] { checkEncode(sign ? -mid : mid, 0); });
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
  require(!resolveWireSpec(ActivationType::F32, WireDtype::Native).compressed());
  rejects<std::length_error>([&] { checkedWireBytes(compressed, SIZE_MAX); });
  int n;
  require(!computeElementCount(0, 1, n));
  require(!computeElementCount(std::numeric_limits<int>::max(), 2, n));
  require(computeElementCount(3, 5, n) && n == 15);
  std::cout << "PASS activation wire codec\n";
}
