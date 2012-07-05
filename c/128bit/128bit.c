#include <stdio.h>

#define str_bool(x) (x)? "true": "false"
int main(int argc, char *argv[])
{
  __uint64_t max64 = ~0;
  __uint128_t i128 = max64;
  max64++;
  i128++;
  printf("max64 eval to %s, i128 eval to %s\n", str_bool(max64), str_bool(i128));
  return 0;
}
