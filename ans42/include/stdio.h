#ifndef STDIO_H
#define STDIO_H
int printf(const char *fmt, ...);
int sprintf(char *buf, const char *fmt, ...);
void message(int level, const char *fmt, ...);

#ifdef __LIBRARY__

#else

enum msg_level {MSG_ERROR, MSG_WARNING, MSG_INFO, MSG_LOG};
#define MSG_LEVEL MSG_INFO

#define log(...) message(MSG_LOG, __VA_ARGS__)
#define info(...) message(MSG_INFO, __VA_ARGS__)
#define warning(...) message(MSG_WARNING, __VA_ARGS__)
#define error(...) message(MSG_ERROR, __VA_ARGS__)

#endif

#endif //STDIO_H
