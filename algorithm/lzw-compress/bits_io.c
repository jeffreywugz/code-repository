#include "bits_io.h"
/*
int main()
{
	int n;
	struct BitStream bits1, bits2;
	struct Code code;
	bits_read_init(&bits1, stdin);
	bits_write_init(&bits2, stdout);
	code.len=5;
	while((n=bits_read(&bits1, &code))){
		code.len=n;
		bits_write(&bits2, &code);
	}
	bits_flush(&bits2);	
	return 0;
}
*/
void bits_read_init(struct BitStream* bits, FILE* fp)
{
	bits->fp=fp;
	bits->pos=8;
	bits->byte=0;
}

void bits_write_init(struct BitStream* bits, FILE* fp)
{
	bits->fp=fp;
	bits->pos=0;
	bits->byte=0;
}

static int bits_get(struct BitStream* bits)
{
	int bit;
	if(bits->pos >= 8){
	       bits->pos=0;
	       if((bits->byte=fgetc(bits->fp))==EOF)return -1;
	}
	bit=(bits->byte >> bits->pos) & 1;
	bits->pos++;
	return bit;
}

static int bits_put(struct BitStream* bits, int bit)
{
	if(bits->pos >= 8){
		bits->pos=0;
		if(fputc(bits->byte, bits->fp)==EOF)
			return -1;
		bits->byte=0;
	}
	bits->byte |= bit << bits->pos;
	bits->pos++;
	return 0;
}

int bits_read(struct BitStream* bits, struct Code* code)
{
	int i, bit;
	code->code=0;
	for(i=0; i<code->len; i++){
		if((bit=bits_get(bits))==-1)break;
		(code->code)|=bit<<i;
	}
	return i;
}

int bits_write(struct BitStream* bits, struct Code* code)
{
	int i;
	for(i=0; i < code->len; i++){
		if(bits_put(bits, (code->code >> i)&1)==-1)break;
	}
	return i;
}

int bits_flush(struct BitStream* bits)
{
	if(bits->pos==0)return 0;
	if(fputc(bits->byte, bits->fp)==EOF)
		return -1;
	return 0;
}
