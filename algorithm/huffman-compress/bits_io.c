#include "bits_io.h"
/*
#include <stdio.h>

int main()
{
	int c, n;
	struct BitStream bits1, bits2;
	init_bits(&bits1);
	init_bits(&bits2);
	while((c=getchar())!='\n')
		write_bits(&bits1, c, 8);
	write_bits(&bits1, 0, 8);

	while((n=read_bits(&bits1, &c, 5))){
		write_bits(&bits2, c, n);
	}
	
	printf("%s\n", bits2.bits);
	return 0;
}
*/
void init_bits(struct BitStream* bits)
{
	bits->n=0;
	bits->pos=0;
}

int get_bit(struct BitStream* bits)
{
	int bit;
	if(bits->pos >= bits->n)return -1;
	bit=(bits->bits[bits->pos/8]>>(bits->pos%8)) & 1;
	bits->pos++;
	return bit;
}

int put_bit(struct BitStream* bits, int bit)
{
	if(bits->n >= BITSTREAM_MAX_BITS)return -1;
	bits->bits[bits->n/8] &= ~(1<<(bits->n%8));
	bits->bits[bits->n/8] |= bit<<(bits->n%8);
	bits->n++;
	return 0;
}

int read_bits(struct BitStream* bits, struct Code* code)
{
	int i, bit;
	code->code=0;
	for(i=0; i<code->len; i++){
		if((bit=get_bit(bits))==-1)break;
		(code->code)|=bit<<i;
	}
	return i;
}

int write_bits(struct BitStream* bits, struct Code* code)
{
	int i;
	for(i=0; i < code->len; i++){
		if(put_bit(bits, (code->code >> i)&1)==-1)break;
	}
	return i;
}

int load_bits(struct BitStream* bits, FILE* fp, int len)
{
	int n=fread(bits->bits, 1, len, fp); 
	bits->n=n*8;
	return n;
}

int unload_bits(struct BitStream* bits, FILE* fp)
{
	int n=fwrite(bits->bits, (bits->n+7)/8, 1, fp);
	return n;
}

int getchar_bits(struct BitStream* bits)
{
	if(bits->pos >= bits->n)return -1;
	int c=bits->bits[bits->pos/8];
	bits->pos += 8;
	return c;
}

int putchar_bits(struct BitStream* bits, int c)
{
	if(bits->n >= BITSTREAM_MAX_BITS)return -1;
	bits->bits[bits->n/8]=c;
	bits->n += 8;
	return 0;
}

