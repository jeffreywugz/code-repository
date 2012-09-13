#include <stdio.h>

int main()
{
  char p0 = 0;
  const char const_p0 = p0;
  char* p1 = NULL;
  const char* const_p1 = p1;
  char* p2 = NULL;
  const char*& const_p2 = p2;
  char** p3 = NULL;
  const char** const_p3 = p3;

  const char const_p5 = 0;
  char p5 = const_p5;
  const char* const_p6 = 0;
  char* p6 = const_cast<char*>(const_p6);
  return 0;
}
