#include <set>
#include <errno.h>
#include <sys/time.h>

using namespace std;

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

int64_t insert_test(int64_t n, bool is_seq)
{
  int64_t start = 0;
  set<int> s;
  int* buf = new int[n];
  for(int64_t i = 0; i < n; i++){
    buf[i] = is_seq? i: random();
  }
  start = get_usec();
  for(int64_t i = 0; i < n; i++){
    s.insert(buf[i]);
  }
  return get_usec() - start;
}

int main(int argc, char* argv[])
{
  int err = 0;
  int64_t n = 0;
  if (argc != 2)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n\n", argv[0]);
  }
  else
  {
    n = atoll(argv[1]);
    printf("rand insert %ld, time=%ldus\n", n, insert_test(n, false));
    printf("seq insert %ld, time=%ldus\n", n, insert_test(n, true));
  }
  return err;
}
