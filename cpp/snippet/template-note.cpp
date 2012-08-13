#include <errno.h>
#include <stdio.h>

/*
// to see whether different template args produce different class
// even the template class do not use the template args
template<typename T>
struct TS1
{
  static int static_func()
  {
  }
  static int* get_static_var_addr()
  {
    static int var_x;
    return &var_x;
  }
};

int main()
{
  int err = 0;
  printf("&TS1<int>::static_func() == %p, &TS1<char>::static_func() == %p\n", TS1<int>::static_func, TS1<char>::static_func);
  printf("&TS1<int>::get_static_var_addr() == %p, &TS1<char>::get_static_var_addr() == %p\n", TS1<int>::get_static_var_addr(), TS1<char>::get_static_var_addr());
  return err;
}
*/

template<const char* id>
struct TS2
{
};

int main()
{
  TS2<"default"> a1;
  return 0;
}
