#include <stdio.h>
#include "bits_io.h"
#include "str_table.h"

struct StrTable str_table;

void lzwc(struct StrTable* table);
void ulzwc(struct StrTable* table);

int main(int argc, char* argv[])
{
	if(argc==1)lzwc(&str_table);
	else ulzwc(&str_table);
	return 0;
}

void lzwc(struct StrTable* table)
{
	int c, index;
	struct Code code;
	struct BitStream bits;
	struct ConStr con_str;
	str_table_init(table);
	bits_write_init(&bits, stdout);
	con_str.prefix=CLEAR;
	code.len=ORIG_CODE_LEN+1;
	while((c=getchar())!=EOF){
		con_str.suffix=c;
		if((index=str_table_lookup(table, &con_str))!=CLEAR){
			con_str.prefix=index;
			continue;
		}
		code.code=con_str.prefix;
		bits_write(&bits, &code);
		if(table->pos >= 1<<code.len){
			code.len++;
			if(code.len > MAX_CODE_LEN){
				str_table_init(table);
				con_str.prefix=c;
				code.len=ORIG_CODE_LEN+1;
				continue;
			}
		}
		str_table_add(table, &con_str);
		con_str.prefix=c;
	}
	code.code=con_str.prefix;
	bits_write(&bits, &code);
	code.code=END;
	bits_write(&bits, &code);
	bits_flush(&bits);
}

void ulzwc(struct StrTable* table)
{
	int len;
	char str[MAX_STR_LEN];
	char *str_head, *str_tail=str+MAX_STR_LEN;
	unsigned short prefix;
	struct Code code;
	struct BitStream bits;
	struct ConStr con_str;
	str_table_init(table);
	bits_read_init(&bits, stdin);
	code.len=ORIG_CODE_LEN+1;
	prefix=CLEAR;
	while(1){
		if(table->pos >= 1<<code.len){
			code.len++;
			if(code.len > MAX_CODE_LEN){
				str_table_init(table);
				code.len=ORIG_CODE_LEN+1;
				prefix=CLEAR;
			}
		}
		bits_read(&bits, &code);
		if(code.code==END)
			break;

		if(code.code == table->pos){
			memmove(str_head-1, str_head, len);
			str_head--;
			*(str_tail-1)=*str_head;
			len++;
		} else {
			con_str=table->str[code.code];
			len=str_table_get(table, &con_str, str_tail); 
			str_head=str_tail-len;
		}
		con_str.prefix=prefix;
		con_str.suffix=str_head[0];
		if(prefix!=CLEAR)
			str_table_add(table, &con_str);
		fwrite(str_head, len, 1, stdout);
		prefix=code.code;
	}
}
