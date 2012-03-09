#include <stdint.h>

int64_t x = 0;
int64_t sig = 0;
int64_t y = 0;
int64_t y2 = 0;
int64_t n_err = 0;

void add() {
    x++;
    sig++;
    y = x;
    y2 = x;
}

void check() {
    int64_t sig_ = sig;
    int64_t x_ = x;
    if(sig_ > x_)
      n_err++;
}
