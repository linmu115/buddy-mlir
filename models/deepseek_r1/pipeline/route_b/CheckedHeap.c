/* Ownership diagnostic for the same component DSOs. Each scratch allocation
 * uses the system heap. A free of a borrowed pointer or a double free aborts.
 * End-of-call cleanup also catches outputs that escape scratch by poisoning
 * and releasing the remaining allocations before the harness reads outputs. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Allocation {
  void *data;
  size_t bytes;
  struct Allocation *next;
} Allocation;

static Allocation *live;
static pthread_mutex_t list_lock = PTHREAD_MUTEX_INITIALIZER;

static void release(Allocation *a) {
  explicit_bzero(a->data, a->bytes);
  free(a->data);
  free(a);
}

void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t bytes) {
  Allocation *a = malloc(sizeof(*a));
  if (!a ||
      posix_memalign(&a->data,
                     alignment < sizeof(void *) ? sizeof(void *) : alignment,
                     bytes ? bytes : 1))
    abort();
  a->bytes = bytes;
  pthread_mutex_lock(&list_lock);
  a->next = live;
  live = a;
  pthread_mutex_unlock(&list_lock);
  return a->data;
}

void *_mlir_memref_to_llvm_alloc(size_t bytes) {
  return _mlir_memref_to_llvm_aligned_alloc(128, bytes);
}

void _mlir_memref_to_llvm_free(void *ptr) {
  if (!ptr)
    return;
  pthread_mutex_lock(&list_lock);
  Allocation **it = &live;
  while (*it && (*it)->data != ptr)
    it = &(*it)->next;
  if (!*it) {
    fprintf(stderr,
            "component kernel freed borrowed or previously freed storage %p\n",
            ptr);
    abort();
  }
  Allocation *a = *it;
  *it = a->next;
  pthread_mutex_unlock(&list_lock);
  release(a);
}

void buddy_arena_reset(void) {
  /* Worker calls are joined, and KernelCall still owns the call mutex. */
  while (live) {
    Allocation *next = live->next;
    release(live);
    live = next;
  }
}
