#include <pthread.h>
extern void buddy_arena_reset(void);
static pthread_mutex_t arena_lock = PTHREAD_MUTEX_INITIALIZER;
void bj47_begin_call(void) {
  pthread_mutex_lock(&arena_lock);
  buddy_arena_reset();
}
void bj47_end_call(void) {
#ifdef BJ47_CHECKED_HEAP
  buddy_arena_reset();
#endif
  pthread_mutex_unlock(&arena_lock);
}
