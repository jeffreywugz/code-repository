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
#define MLOG(prefix, format, ...) {get_tl_printer().reset(); int64_t cur_ts = get_us(); get_mlog().append("[%ld.%.6ld] %s %s:%d [%ld] " format "\n", cur_ts/1000000, cur_ts%1000000, #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__); }

#define ERR(err) MLOG(ERROR, "err=%d syscall: %s", err, strerror(errno))
#include "debug.h"
#endif /* __OB_AX_AX_COMMON_H__ */
