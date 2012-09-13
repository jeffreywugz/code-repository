#include <stdio.h>
#include <string.h>
#include <assert.h>

int main(int argc, char** argv)
{
  int i = 1;
  printf("%d\n", strlen("\("));
  assert("\("[0] == '(');
  return 0;
}
