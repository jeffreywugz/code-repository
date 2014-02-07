#ifndef __OB_AX_ERRNO_H__
#define __OB_AX_ERRNO_H__
#ifdef ERRNO_DEF
ERRNO_DEF(AX_SUCCESS, 0, "success")
ERRNO_DEF(AX_FATAL_ERR, -1, "fatal")
ERRNO_DEF(AX_CMD_ARGS_NOT_MATCH, -2, "cmd args not match")
#endif

#define ERRNO_DEF(name, value, desc) const static int name = value; // desc
#include __FILE__
#undef ERRNO_DEF

#endif /* __OB_AX_ERRNO_H__ */
