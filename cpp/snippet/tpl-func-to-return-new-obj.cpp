#include <stdio.h>

template<typename T>
void print(T t, int depth) { t.print(depth); }

template<>
void print(int x, int depth) { printf("%*s%ld\n", 8*depth, "", x); }

struct EmptyStruct{};
template<>
void print(EmptyStruct x, int depth) {};
  

template<typename T1, typename T2, typename T3, typename T4, typename T5>
struct Struct
{
  Struct(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5): t1(t1), t2(t2), t3(t3), t4(t4), t5(t5) {}
  ~Struct() {}
  void print(int d=0) { ::print(t1, d); ::print(t2, d+1); ::print(t3, d+1); ::print(t4, d+1); :: print(t5, d+1); }
  T1 t1; T2 t2; T3 t3; T4 t4; T5 t5;
};


template<typename T1, typename T2, typename T3, typename T4, typename T5>
Struct<T1, T2, T3, T4, T5> make_struct(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5)
{
  return Struct<T1, T2, T3, T4, T5>(t1, t2, t3, t4, t5);
}

#define head5(x1, x2, x3, x4, x5, ...) x1, x2, x3, x4, x5
#define S(...) make_struct(head5(__VA_ARGS__, EmptyStruct(), EmptyStruct(), EmptyStruct(), EmptyStruct(), EmptyStruct()))
int main()
{
  //S(1, 2).print();
  S(S(3, 4, 5), S(1, S(2,10, 12)), S(6, 7, 8)).print();
  return 0;
}
