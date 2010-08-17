#ifndef  __BITS_IO_H__
#define  __BITS_IO_H__

#ifndef  BITSTREAM_MAX_BITS
#define  BITSTREAM_MAX_BITS (1<<20)
#endif
#include <stdio.h>
struct BitStream {
	unsigned char bits[BITSTREAM_MAX_BITS/8];
	int n;
	int pos;
};

struct Code {
	int code;
	int len;
};

void init_bits(struct BitStream* bits);
int load_bits(struct BitStream* bits, FILE* fp, int len);
int unload_bits(struct BitStream* bits, FILE* fp);
int get_bit(struct BitStream* bits);
int put_bit(struct BitStream* bits, int bit);
int read_bits(struct BitStream* bits, struct Code* code);
int write_bits(struct BitStream* bits, struct Code* code);
int getchar_bits(struct BitStream* bits);
int putchar_bits(struct BitStream* bits, int c);


#endif  /*__BITS_IO_H__*/
