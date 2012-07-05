#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _rng_t {
  uint64_t s[3];
} rng_t;

void init_rng64(rng_t* state, uint64_t x)
{
  state->s[0] = 0;
  state->s[1] = 0;
  state->s[2] = x;
}

static uint64_t rng64(rng_t *_s)
{
  uint64_t c = 7319936632422683419ULL;
  uint64_t* s = _s->s;
  uint64_t x = s[1];
  
  /* Increment 128bit counter */
  s[0] += c;
  s[1] += c + (s[0] < c);
  
  /* Two h iterations */
  x ^= (x >> 32) ^ s[2];
  x *= c;
  x ^= x >> 32;
  x *= c;
  
  /* Perturb result */
  return x + s[0];
}

int test_rng(int64_t n)
{
  printf("test_rng(n=%ld)\n", n);
  rng_t state;
  uint64_t s = 0;
  init_rng64(&state, 1234);
  for(int64_t i = 0; i < n; i++)
    s ^= rng64(&state);
  printf("s=%lu\n", s);
  return 0;
}

int main(int argc, char *argv[])
{
  int64_t n = 0;
  if (argc != 2)
  {
    fprintf(stderr, "%s n\n", argv[0]);
  }
  else
  {
    test_rng(1LL<<atoll(argv[1]));
  }
  return 0;
}
