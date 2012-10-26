
void foo();

volatile extern int x;
volatile int x = 0;
int main(int argc, char *argv[])
{
  foo();
  x = 1;
  foo();
  return 0;
}
