/* Qwen2 1.5B real-data differential fixture. Invoke with a bundle, two 14-layer
 * Route A stage directories, and a pure single-layer A reference directory.
 * Only this reference adapter knows the legacy result descriptor layout. */
#define _GNU_SOURCE
#include <buddy/runtime/pipeline/RouteB.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <link.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
  H = 1536,
  V = 151936,
  D = 128,
  J = 2,
  T = 1024,
  C = 64,
  L = 28,
  KV = J * T * D
};
#define REQUIRE(c)                                                             \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "failed line %d: %s\n", __LINE__, #c);                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
static char *join(const char *dir, const char *name) {
  char *p = NULL;
  REQUIRE(asprintf(&p, "%s/%s", dir, name) > 0);
  return p;
}
static void *library(const char *dir, const char *name) {
  char *p = join(dir, name);
  void *h = dlopen(p, RTLD_LOCAL | RTLD_NOW);
  if (!h)
    fprintf(stderr, "%s: %s\n", p, dlerror());
  free(p);
  REQUIRE(h);
  return h;
}
static void *sym(void *h, const char *name) {
  void *p = dlsym(h, name);
  REQUIRE(p);
  return p;
}
static void *blob(const char *dir, const char *name, size_t bytes) {
  char *p = join(dir, name);
  int fd = open(p, O_RDONLY);
  struct stat st;
  if (fd < 0)
    perror(p);
  REQUIRE(fd >= 0 && !fstat(fd, &st) && st.st_size == (off_t)bytes);
  void *v = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  REQUIRE(v != MAP_FAILED);
  close(fd);
  free(p);
  return v;
}
static float *floats(size_t n) {
  float *p;
  REQUIRE(!posix_memalign((void **)&p, 128, n * sizeof(*p)));
  memset(p, 0, n * sizeof(*p));
  return p;
}
static unsigned failures;
static void compare(const char *label, const float *a, const float *b,
                    size_t n) {
  double max = 0;
  size_t mismatch = 0, first = 0;
  if (memcmp(a, b, n * sizeof(float)))
    for (size_t i = 0; i < n; i++) {
      if (memcmp(a + i, b + i, 4)) {
        if (!mismatch)
          first = i;
        mismatch++;
        double d = fabs((double)a[i] - b[i]);
        if (!isfinite(d))
          d = INFINITY;
        if (d > max)
          max = d;
      }
    }
  printf("  %s n=%zu differing=%zu max_abs=%.9g first=%zu\n", label, n,
         mismatch, max, first);
  failures += mismatch != 0;
}
#define DESC(N)                                                                \
  struct {                                                                     \
    void *base, *data;                                                         \
    int64_t off, size[N], stride[N];                                           \
  }
typedef DESC(1) D1;
typedef DESC(2) D2;
typedef DESC(3) D3;
typedef DESC(4) D4;
typedef struct {
  D1 p;
  D4 k, v;
} ALayer;
typedef struct {
  ALayer layer[14];
  D3 out;
} A14;
typedef struct {
  ALayer layer;
  D3 out;
} A1;
typedef struct {
  void *lib;
  D1 f, q;
  D4 kv[28];
  int layers;
  void (*reset)(void);
} Reference;
static D1 vec(void *p, int64_t n) { return (D1){p, p, 0, {n}, {1}}; }
static D4 slot(void *p) {
  return (D4){p, p, 0, {1, J, T, D}, {KV, T * D, D, 1}};
}
static Reference reference(const char *dir, const char *dso, int layers, int nf,
                           int nq, const int8_t *single_q) {
  Reference a = {0};
  a.lib = library(dir, dso);
  a.layers = layers;
  a.reset = sym(a.lib, "buddy_arena_reset");
  a.f = vec(blob(dir, "arg0-w4g32-f32.data", 4ULL * nf), nf);
  a.q = vec(single_q ? (void *)single_q : blob(dir, "arg0-w4g32-q4.data", nq),
            nq);
  for (int i = 0; i < 2 * layers; i++)
    a.kv[i] = slot(floats(KV));
  return a;
}
#define TRIPLE(i) &p, &a->kv[2 * (i)], &a->kv[2 * (i) + 1]
#define VP3 void *, void *, void *
#define V7 VP3, VP3, VP3, VP3, VP3, VP3, VP3
typedef void (*Fn14)(void *, void *, void *, void *, V7, V7);
typedef void (*Fn1)(void *, void *, void *, void *, VP3);
static const float *run_a(Reference *a, int rows, int64_t pos, void *input) {
  static A14 r14;
  static A1 r1;
  D1 p = vec(&pos, 1);
  const char *entry = rows == 1 ? "_mlir_ciface_forward_decode"
                                : "_mlir_ciface_forward_prefill";
  a->reset();
  D3 out;
  ALayer *result;
  if (a->layers == 14) {
    ((Fn14)sym(a->lib, entry))(&r14, &a->f, &a->q, input, TRIPLE(0), TRIPLE(1),
                               TRIPLE(2), TRIPLE(3), TRIPLE(4), TRIPLE(5),
                               TRIPLE(6), TRIPLE(7), TRIPLE(8), TRIPLE(9),
                               TRIPLE(10), TRIPLE(11), TRIPLE(12), TRIPLE(13));
    out = r14.out;
    result = r14.layer;
  } else {
    ((Fn1)sym(a->lib, entry))(&r1, &a->f, &a->q, input, TRIPLE(0));
    out = r1.out;
    result = &r1.layer;
  }
  for (int i = 0; i < a->layers; i++) {
    D4 cache[] = {result[i].k, result[i].v};
    for (int j = 0; j < 2; j++) {
      float *src = (float *)cache[j].data + cache[j].off;
      if (src != a->kv[2 * i + j].data)
        memcpy(a->kv[2 * i + j].data, src, KV * 4);
    }
  }
  return (float *)out.data + out.off;
}
typedef int32_t (*Embed)(const float *, const int64_t *, float *);
typedef int32_t (*Prepare)(const float *, int64_t, float *, float *, void *);
typedef int32_t (*Layer)(const BJ47WeightsV1 *, const float *, float *, float *,
                         const BJ47FrameV1 *, float *);
typedef int32_t (*Head)(const BJ47WeightsV1 *, const float *, float *);
static int count_runtime(struct dl_phdr_info *info, size_t size, void *count) {
  (void)size;
  if (strstr(info->dlpi_name, "libbj47_kernel_runtime.so")) {
    (*(int *)count)++;
    printf("shared runtime %s\n", info->dlpi_name);
  }
  return 0;
}
int main(int argc, char **argv) {
  REQUIRE(argc == 5);
  const char *bundle = argv[1];
  setvbuf(stdout, NULL, _IOLBF, 0);
  unsigned long vlenb;
  __asm__ volatile("csrr %0, 0xc22" : "=r"(vlenb));
  printf("cpu=%d vlen_bits=%lu\n", sched_getcpu(), 8 * vlenb);
  REQUIRE(vlenb == 128);
  void *libs[] = {library(bundle, "embed.so"),
                  library(bundle, "decoder_layer.so"),
                  library(bundle, "lm_head.so")};
  const BJ47KernelInfoV1 *info[3];
  const char *role_names[] = {"embed", "decoder_layer", "lm_head"};
  for (int i = 0; i < 3; i++) {
    info[i] = ((const BJ47KernelInfoV1 *(*)(void))sym(libs[i],
                                                      "bj47_kernel_info_v1"))();
    REQUIRE(info[i]->abi_version == (i == 2 ? 2U : 1U) &&
            info[i]->struct_bytes == sizeof(*info[i]) &&
            info[i]->role == (unsigned)i + 1 && info[i]->chunk == C &&
            info[i]->width == H && info[i]->context == T &&
            info[i]->head_dim == D && info[i]->kv_heads == J &&
            info[i]->vocab_size == V &&
            !memcmp(info[0]->shape_id, info[i]->shape_id, 32));
    printf("info %s shape=", role_names[i]);
    for (int j = 0; j < 32; j++)
      printf("%02x", info[i]->shape_id[j]);
    printf(" layout=");
    for (int j = 0; j < 32; j++)
      printf("%02x", info[i]->layout_id[j]);
    puts("");
  }
  int runtimes = 0;
  dl_iterate_phdr(count_runtime, &runtimes);
  REQUIRE(runtimes == 1);
  const float *table = blob(bundle, "embed.f32.data", 4ULL * V * H),
              *inv = blob(bundle, "rope.f32.data", D * 2);
  BJ47WeightsV1 w[L],
      head = {blob(bundle, "lm_head.f32.data", 4 * H),
              blob(bundle, "lm_head.q4.data", info[2]->i8_elements)};
  float *kv[2 * L];
  for (int i = 0; i < L; i++) {
    char filename[64];
    snprintf(filename, sizeof(filename), "layer_%02d.f32.data", i);
    w[i].f32 = blob(bundle, filename, info[1]->f32_elements * 4);
    snprintf(filename, sizeof(filename), "layer_%02d.q4.data", i);
    w[i].q4 = blob(bundle, filename, info[1]->i8_elements);
    kv[2 * i] = floats(KV);
    kv[2 * i + 1] = floats(KV);
  }
  Reference a0 = reference(argv[2], "stage.so", 14, 233445440, 736985088, NULL);
  Reference a1 = reference(argv[3], "stage.so", 14, 73280, 868257792, NULL);
  Reference a7 = reference(argv[4], "ref.so", 1, 5184, 52641792, w[7].q4);
  float *c = floats(C * D), *s = floats(C * D), *x = floats(C * H),
        *y = floats(C * H), *logits = floats(V);
  float *other_logits = floats(V), *prior_k = floats(KV), *prior_v = floats(KV);
  _Alignas(128) int64_t ids[C];
  int positions[] = {0, 64, 65, T - C, T - 1}, lengths[] = {C, 1, C, C, 1};
  for (int step = 0; step < 5; step++) {
    printf("  caller buffers in=%p out=%p layer7.k=%p layer7.v=%p\n", (void *)x,
           (void *)y, (void *)kv[14], (void *)kv[15]);
    int n = lengths[step], pos = positions[step];
    printf("position=%d rows=%d\n", pos, n);
    for (int r = 0; r < n; r++)
      ids[r] = (151646 + step * 9929 + r * 787) % V;
    D2 idref = {ids, ids, 0, {1, n}, {n, 1}};
    const float *mid = run_a(&a0, n, pos, &idref);
    D3 hidden = {(void *)mid, (void *)mid, 0, {1, n, H}, {n * H, H, 1}};
    const float *gold = run_a(&a1, n, pos, &hidden);
    Embed embed = sym(libs[0], n == 1 ? "embed_decode" : "embed_prefill");
    Prepare prep = sym(libs[1], n == 1 ? "prepare_decode" : "prepare_prefill");
    Layer layer = sym(libs[1], n == 1 ? "layer_decode" : "layer_prefill");
    Head project = sym(libs[2], n == 1 ? "lm_head_decode" : "lm_head_prefill");
    REQUIRE(!embed(table, ids, x));
    for (int r = 0; r < n; r++)
      REQUIRE(!memcmp(x + r * H, (float *)a0.f.data + ids[r] * H, H * 4));
    REQUIRE(prep(inv, T - n + 1, c, s, NULL) < 0);
    REQUIRE(!prep(inv, pos, c, s, NULL));
    float worst = 0;
    for (int r = 0; r < n; r++)
      for (int j = 0; j < D; j++) {
        float v = (pos + r) * inv[j % (D / 2)];
        worst = fmaxf(worst, fmaxf(fabsf(c[r * D + j] - cosf(v)),
                                   fabsf(s[r * D + j] - sinf(v))));
      }
    printf("  prepare libm max_abs=%.9g\n", worst);
    REQUIRE(worst < 2e-6f);
    BJ47FrameV1 frame = {pos, c, s, NULL};
    for (int i = 0; i < L; i++) {
      memcpy(prior_k, kv[2 * i], KV * 4);
      memcpy(prior_v, kv[2 * i + 1], KV * 4);
      REQUIRE(!layer(&w[i], x, kv[2 * i], kv[2 * i + 1], &frame, y));
      for (int h = 0; h < J; h++) {
        size_t begin = h * T * D, after = begin + (pos + n) * D;
        for (int j = 0; j < 2; j++) {
          const float *before = j ? prior_v : prior_k;
          REQUIRE(!memcmp(before + begin, kv[2 * i + j] + begin, pos * D * 4));
          REQUIRE(!memcmp(before + after, kv[2 * i + j] + after,
                          (T - pos - n) * D * 4));
        }
      }
      if (i == 7) {
        D3 in = {x, x, 0, {1, n, H}, {n * H, H, 1}};
        compare("A[7:8] hidden", run_a(&a7, n, pos, &in), y, n * H);
      }
      if (i == 13)
        compare("A[0:14] hidden", mid, y, n * H);
      float *tmp = x;
      x = y;
      y = tmp;
    }
    REQUIRE(!project(&head, x + (n - 1) * H, logits));
    compare("A[14:28] logits", gold, logits, V);
    Head other = sym(libs[2], n == 1 ? "lm_head_prefill" : "lm_head_decode");
    if (n == 1) {
      Head alias = sym(libs[2], "lm_head");
      REQUIRE(!alias(&head, x, other_logits));
      REQUIRE(!memcmp(other_logits, logits, 4 * V));
    }
    REQUIRE(!other(&head, x + (n - 1) * H, other_logits));
    size_t different = 0;
    for (int i = 0; i < V; i++)
      different += memcmp(logits + i, other_logits + i, 4) != 0;
    printf("  alternate head lowering differs on %zu logits\n", different);
    for (int i = 0; i < 2 * L; i++) {
      Reference *ref = i < 28 ? &a0 : &a1;
      if (memcmp(kv[i], ref->kv[i % 28].data, KV * 4)) {
        char name[64];
        snprintf(name, sizeof(name), "KV layer %d %c", i / 2,
                 i % 2 ? 'V' : 'K');
        compare(name, kv[i], ref->kv[i % 28].data, KV);
      }
    }
    printf("  compared all 56 complete KV slots (%d bytes)\n", 2 * L * KV * 4);
  }
  printf("failed comparisons=%u\n", failures);
  return failures != 0;
}
