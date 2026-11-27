// Probe private hot functions in the final ELF. Take their offsets from the
// symbol table of that exact file (the release ABI remains unchanged).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
using Vec = std::vector<float>;
struct Image {
  void *handle;
  Dl_info info{};
  Image(const char *path) : handle(dlopen(path, RTLD_NOW)) {
    if (!handle || !dladdr(dlsym(handle, "bj47_kernel_info_v1"), &info))
      throw std::runtime_error("could not load probe image");
  }
  template <typename Fn> Fn get(const char *offset) {
    return reinterpret_cast<Fn>(static_cast<char *>(info.dli_fbase) +
                                std::stoul(offset, nullptr, 16));
  }
};
void fill(Vec &v, unsigned seed) {
  for (float &x : v) {
    seed = seed * 1664525u + 1013904223u;
    x = (int(seed >> 16) - 32768) / 32768.f;
  }
}
void check(float got, float ref, float &worst) {
  if (!std::isfinite(got) || !std::isfinite(ref))
    throw std::runtime_error("nonfinite result");
  worst = std::max(worst, std::abs(got - ref));
}
void gates(char **args) {
  constexpr int width = 5120, heads = 48;
  using Gate =
      void (*)(const float *, const float *, const float *, float *, float *);
  Image layer(args[0]);
  Gate fn[] = {layer.get<Gate>(args[1]), layer.get<Gate>(args[2]),
               layer.get<Gate>(args[3])};
  Vec x(width * 64), weights(542944);
  std::ifstream source(args[4], std::ios::binary);
  if (!source.read(reinterpret_cast<char *>(weights.data()),
                   weights.size() * 4))
    throw std::runtime_error("layer 0 f32 pack");
  fill(x, 17);
  for (int m : {1, 2, 3, 4, 64}) {
    float worst = 0;
    for (int h = 0; h < heads; ++h) {
      const float *wa = weights.data() + 256000 + h * width,
                  *wb = weights.data() + 10240 + h * width;
      for (int r = 0; r < m;) {
        int n = m - r >= 4 ? 4 : m - r >= 2 ? 2 : 1;
        float a[4], b[4];
        fn[n / 2](x.data() + r * width, wa, wb, a, b);
        for (int j = 0; j < n; ++j) {
          float sa[8]{}, sb[8]{}, ra = 0, rb = 0;
          for (int k = 0; k < width; ++k) {
            sa[k % 8] += x[(r + j) * width + k] * wa[k];
            sb[k % 8] += x[(r + j) * width + k] * wb[k];
          }
          for (int i = 0; i < 8; ++i) {
            ra += sa[i];
            rb += sb[i];
          }
          check(a[j], ra, worst);
          check(b[j], rb, worst);
        }
        r += n;
      }
    }
    printf("gate M=%d both outputs, 48 heads: max_abs=%.9g\n", m, worst);
    if (worst != 0)
      throw std::runtime_error("gate mismatch");
  }
}
void attention(char **args) {
  constexpr int dim = 256;
  using Attn = void (*)(const float *, const float *, const float *, int,
                        float *, float *, float *);
  Image layer(args[0]);
  auto fn = layer.get<Attn>(args[1]);
  for (int prior : {64, 256, 1024}) {
    Vec keys((prior + 64) * dim), values(keys.size()), queries(64 * dim);
    fill(keys, prior + 5);
    fill(values, prior + 11);
    fill(queries, 93);
    float worst = 0;
    for (int r = 0; r < 64; ++r) {
      const int n = prior + r + 1;
      auto q = queries.data() + r * dim;
      // Include ordinary scores, near one-hot rows and multiple record maxima.
      if (r % 3 == 1)
        for (int i = 0; i < dim; ++i)
          q[i] *= 12;
      if (r % 3 == 2)
        for (int i = 0; i < dim; ++i)
          q[i] *= 100;
      Vec out(dim), ref(dim), scores(n);
      Vec scratch(2 * n);
      fn(q, keys.data(), values.data(), n, out.data(), scratch.data(),
         scratch.data() + n);
      for (int t = 0; t < n; ++t) {
        float dot = 0;
        for (int i = 0; i < dim; ++i)
          dot += q[i] * keys[t * dim + i];
        scores[t] = dot / 16;
      }
      float max = *std::max_element(scores.begin(), scores.end()), denom = 0;
      for (auto &s : scores) {
        s = std::exp(s - max);
        denom += s;
      }
      for (int t = 0; t < n; ++t) {
        float p = scores[t] / denom;
        for (int i = 0; i < dim; ++i)
          ref[i] += p * values[t * dim + i];
      }
      for (int i = 0; i < dim; ++i)
        check(out[i], ref[i], worst);
    }
    printf("attention P64 prior KV=%d all 64 causal rows: max_abs=%.9g\n",
           prior, worst);
    if (!(worst < 1e-5f))
      throw std::runtime_error("attention mismatch");
  }
}
} // namespace
int main(int argc, char **argv) {
  if (argc == 7 && std::string(argv[1]) == "gate")
    gates(argv + 2);
  else if (argc == 4 && std::string(argv[1]) == "attention")
    attention(argv + 2);
  else
    return 2;
}
