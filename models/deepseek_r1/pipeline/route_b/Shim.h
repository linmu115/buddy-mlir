#include "buddy/runtime/pipeline/RouteB.h"
#include <array>
#include <cstddef>
#include <cstdint>

extern "C" void bj47_begin_call();
extern "C" void bj47_end_call();
struct KernelCall {
  KernelCall() { bj47_begin_call(); }
  ~KernelCall() { bj47_end_call(); }
};

template <size_t N> struct Descriptor {
  const void *allocated, *aligned;
  int64_t offset;
  std::array<int64_t, N> shape, stride;
  Descriptor(const void *ptr, std::array<int64_t, N> dims)
      : allocated(ptr), aligned(ptr), offset(0), shape(dims) {
    int64_t n = 1;
    for (size_t i = N; i-- > 0;) {
      stride[i] = n;
      n *= shape[i];
    }
  }
};
