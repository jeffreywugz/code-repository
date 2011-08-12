#include <stdio.h>

class A
{
public:
  virtual void hello()
  {
    printf("hello from A\n");
  }
  void hello(char* msg)
  {
    printf("hello from %s\n", msg);
  }
};

class B: public A
{
public:
  virtual void hello()
  {
    printf("hello from B\n");
    A::hello("Bx");
  }
  void hello(char* msg)
  {
    printf("hello from %s\n", msg);
  }
  void hello(char* msg, int i)
    {
      printf("hello from %s.%d\n", msg, i);
    }
};

int main()
{
  A* p = new B();
  B b;
  p->hello();
  delete p;
  b.hello("BB");
  b.hello("BB", 2);
  return 0;
}
