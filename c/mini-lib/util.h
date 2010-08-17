#ifndef _UNTIL_H_
#define _UNTIL_H_
#include <stdio.h>
#include <stdlib.h>
#include <check.h>

#define array_len(a) (sizeof(a)/sizeof(a[0]))
#define swap(x, y) {typeof(x) t; t=x; x=y; y=t;}

#ifdef  NDEBUG
#define _dbg(exp) ((void)0)
#else
#define _dbg(exp) (exp)
#endif

#define _dbg_msg(...) _dbg(fprintf(stderr, __VA_ARGS__))
#define _dbg_pos() _dbg_msg("%s:%d:%s: ", __FILE__, __LINE__, __func__)
#define debug(...) {_dbg_pos(); _dbg_msg("DEBUG: "); _dbg_msg(__VA_ARGS__);}
#define panic(...) {_dbg_pos(); _dbg_msg("PANIC: ");_dbg_msg(__VA_ARGS__); exit(-1);}
#define sys_panic(...) {_dbg_pos(); _dbg_msg("PANIC:"); _dbg_msg(__VA_ARGS__); perror(" "); exit(-2);}
#define quick_define_tcase_reg(tcase) void tcase##_tcase_reg(Suite* s) { tcase_reg(s, tcase##_test, #tcase);}

void tcase_reg(Suite* s, void* func, const char* name);
void* self_symbol(const char* name);
#endif /* _UNTIL_H_ */
