#include <assert.h>
#include <stdint.h>

static inline uint64_t rol(const uint64_t x, const int64_t shift) {
  uint64_t ret = x;
  __asm__ ("rolq %b1,%q0"
           :"+g" (ret)
           :"Jc" (shift));
  return ret;
}

static inline uint64_t rol2(const uint64_t x, const int64_t shift) {
  return (x >> shift) | (x << (64-shift));
}

int main()
{
  for(uint64_t i = 0; i < (1LL<<54); i += (1<<28))
    for(uint64_t j = 0; j < 64; j++)
      assert(rol(i, j) == rol(i, j));
  return 0;
}
