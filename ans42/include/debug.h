#ifndef DEBUG_H
#define DEBUG_H

#ifdef  NDEBUG
#define _dbg(exp) ((void)0)
#else
#define _dbg(exp) (exp)
#endif

#define _dbg_msg(...) _dbg(printf(__VA_ARGS__))
#define _dbg_pos() _dbg_msg("%s@%s# ", __FILE__, __func__)
#define debug(...) ({_dbg_msg("[debug] "); _dbg_pos(); _dbg_msg(__VA_ARGS__);})
#define panic(...) ({_dbg_msg("[panic] "); _dbg_pos(); _dbg_msg(__VA_ARGS__); halt();})
#define watch(i) debug("%s=0x%x\n", #i, i)
#define trace() debug("code reach here\n")
#endif //DEBUG_H
