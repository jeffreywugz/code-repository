#include <errno.h>
#include <stdio.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

int int_cmp(void* x, void* y)
{
  return *((int64_t*)x) - *((int64_t*)y);
}

int merge(int64_t* merged, const int64_t** streams, const int64_t n_stream, const int64_t* stream_lens)
{
  int err = 0;
  int64_t lt[n_stream * 2];
  for(int64_t i = 0; i < n_stream * 2; i++){
    lt[i] = -1;
  }
  return err;
}

int test_merge(const int64_t n_stream, const int64_t stream_len)
{
  int64_t streams[n_stream][stream_len];
  int64_t merged[n_stream * stream_len];
  int64_t stream_lens[n_stream];
  for(int64_t i = 0; i < n_stream; i++){
    stream_lens[i] = stream_len;
  }
  for(int64_t i = 0; i < n_stream; i++){
    for(int64_t j = 0; j < stream_len; j++){
      streams[i][j] = random();
    }
  }
  for(int64_t i = 0; i < n_stream; i++){
    qsort(streams[i], stream_len, sizeof(int64_t), int_cmp);
  }
  return merge(merged, streams, n_stream, n_stream_lens);
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc != 3)
  {
    fprintf(stderr, "%s n_stream stream_len\n", argv[0]);
  }
  else
  {
    err = test_merge(atoll(argv[1]), atoll(argv[2]));
  }
  return err;
}
