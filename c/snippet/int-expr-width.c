#include <stdio.h>
#include <stdint.h>

int main()
{
  uint16_t x = 0;
  uint16_t y = 0;
  //x += 1; // error
  //x += (uint16_t)1 // error
  x = (uint16_t) (x + 1);
  printf("sizeof(uint16_t) = %ld, sizeof(uint16_t++) = %ld, sizeof(uint16_t + 1) = %ld\n"
         "sizeof(uint16_t + uint16_t) = %ld, sizeof(char) = %ld, sizeof(char+char) = %ld\n",
         sizeof(x), sizeof(x++), sizeof(x + 1),
         sizeof(x+y), sizeof('c'), sizeof('c' + 'd'));
  return 0;
}
