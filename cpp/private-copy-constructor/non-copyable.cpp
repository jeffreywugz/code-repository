#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
  void operator=(const TypeName&)

class B
{
  private:
    DISALLOW_COPY_AND_ASSIGN(B);
};

class NonCopyable
{
  public:
    NonCopyable(){}
    ~NonCopyable(){}
  private:
    NonCopyable(const NonCopyable&);
    void operator=(const NonCopyable&);
};

class A: NonCopyable
{};

void foo(A a){}

int main()
{
  A a;
  //foo(a);
  return 0;
}
