#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(int argc, char** argv)
{
  int i = 1;
  printf("%d\n", strlen("\("));
  assert("\("[0] == '(');
  printf("large int: %ld\n", atoll("123456789103837375894358925235"));
  printf("large int: %ld\n", atoll("2123456789103837375894358925235"));
  return 0;
}
