struct Nil{};
Nil nil;
#define head3(a1, a2, a3,...) a1, a2, a3
template<typename Ret, typename Func, typename T1, typename T2, typename T3>
Ret call(Func func, T1 a1, T2 a2, T3 a3)
{
  return func(a1, a2, a3);
}

template<typename Func, typename T1, typename T2>
int call(Func func, T1 a1, T2 a2, Nil a3)
{
  return func(a1, a2);
}

template<typename Func, typename T1>
int call(Func func, T1 a1, Nil a2, Nil a3)
{
  return func(a1);
}

template<typename Func>
int call(Func func, Nil a1, Nil a2, Nil a3)
{
  return func();
}
#define Call(func, ...) call(func, head3(__VA_ARGS__, nil, nil, nil))

struct Closure
{
  Closure(){}
  ~Closure(){}
  virtual int call() = 0;
};
#if 1
template<typename Func, typename T1, typename T2, typename T3>
struct ClosureImp: public Closure
{
  ClosureImp(Func& func, T1& a1, T2& a2, T3& a3): func(func), a1(a1), a2(a2), a3(a3)
  {}
  ~ClosureImp() {}
  int call() { return ::call(func, a1, a2, a3); }
  Func func;
  T1 a1;
  T2 a2;
  T3 a3;
};

template<typename Func, typename T1, typename T2, typename T3>
ClosureImp<Func, T1, T2, T3>& make_closure_(Func func, T1 a1, T2 a2, T3 a3)
{
  static ClosureImp<Func, T1, T2, T3> closure(func, a1, a2, a3);
  return closure;
}

#define make_closure(func, ...) make_closure_(func, head3(__VA_ARGS__, nil, nil, nil))
#endif

int add(int x, int y){return x + y; }

#include <stdio.h>

int main()
{
  Closure* x  = &make_closure(add, 2, 3);
  int s = x->call();
  printf("s=%d\n", s);
  return 0;
}
