
struct Callable
{
  void* (*func)(void*);
  void* arg;
};

int delayed_call(struct Callable* callback, int64_t delay)
{
};
