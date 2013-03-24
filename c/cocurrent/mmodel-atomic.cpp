#include <stdint.h>

int main()
{
  uint64_t x = 0;
  __atomic_load_n(&x);
  return 0;
}
