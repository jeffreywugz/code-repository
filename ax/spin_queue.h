#include "ax_common.h"

class SpinQueue
{
public:
  SpinQueue(){}
  ~SpinQueue(){}
private:
  int64_t push_;
  int64_t pop_;
  void* item_;
};
