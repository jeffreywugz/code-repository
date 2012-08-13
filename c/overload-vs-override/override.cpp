class A
{
  public:
    int foo(int x) {
      return 1;
    }
};

class B: public A
{
  public:
    int foo(int x, int y){
      return 2;
    }
};
  
int main(int argc, char *argv[])
{
  B b;
  //b.foo(1);
  b.A::foo(1);
  return 0;
}
