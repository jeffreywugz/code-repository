#ifndef _ASSERT_H
#define _ASSERT_H

#include <debug.h>
#define assert(exp) _dbg((exp)? (void)0 : panic("assert fail: %s\n", #exp))

#endif  /* NDEBUG */
