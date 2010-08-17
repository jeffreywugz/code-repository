#ifndef  __BITS_IO_H__
#define  __BITS_IO_H__

#include <stdio.h>
struct BitStream {
	FILE* fp;
	int byte;
	int pos;
};

struct Code {
	int code;
	int len;
};

void bits_read_init(struct BitStream* bits, FILE* fp);
void bits_write_init(struct BitStream* bits, FILE* fp);
int bits_flush(struct BitStream* bits);
int bits_read(struct BitStream* bits, struct Code* code);
int bits_write(struct BitStream* bits, struct Code* code);


#endif  /*__BITS_IO_H__*/
