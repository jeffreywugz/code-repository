#ifndef DEBUG_H
#define DEBUG_H

#ifdef  NDEBUG
#define _dbg(exp) ((void)0)
#else
#define _dbg(exp) (exp)
#endif

#define _dbg_msg(...) _dbg(fprintf(stderr, __VA_ARGS__))
#define debug(...) ({_dbg_msg("[%s@%s] ", __FILE__, __func__); _dbg_msg(__VA_ARGS__);})
#define panic(...) ({debug(__VA_ARGS__); exit(1);})
#endif //DEBUG_H
