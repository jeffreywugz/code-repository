#ifndef __OB_AX_AX_COMMON_H__
#define __OB_AX_AX_COMMON_H__

#include "a0.h"
#include "lock.h"
#include "spin_queue.h"
#include "futex_queue.h"
#include "id_map.h"
#include "cache_index.h"

#include "malloc.h"
#include "printer.h"
#include "mlog.h"
#define DLOG(prefix, format, ...) {get_tl_printer().reset(); get_mlog().append("[%ld] %s %s:%ld [%ld] " format "\n", get_us(), #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__); }

#include "utils.h"
#include "types.h"
#endif /* __OB_AX_AX_COMMON_H__ */
