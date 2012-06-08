#if SIMPLE
#define open(...) openx(__VA_ARGS__, default_arg)
#define openx(a, b, c, ...) open3(a, b, c)
#else
#define open(...) openx(__VA_ARGS__, 3, 2)
#define openx(a, b, c, d, ...) open##d(a, b, c)
#define open2(a, b, ...) open2_func(a, b)
#define open3(a, b, c, ...) open3_func(a, b, c)
#endif

open(x, y)
open(x, y, z)
