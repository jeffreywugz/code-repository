#include <stdio.h>

class A
{
  virtual void hello()
  {
    printf("hello from A\n");
  }
  void hello(char* msg)
  {
    printf("hello from %s\n", msg);
  }
};

class B
{
  virtual void hello()
  {
    printf("hello from A");
    A::hello("Bx");
  }
  void hello(char* msg)
  {
    printf("hello from %s\n", msg);
  }
};

int main()
{
  A* p = new B();
  B b;
  p->hello();
  delete p;
  b->hello("BB");
  return 0;
}
