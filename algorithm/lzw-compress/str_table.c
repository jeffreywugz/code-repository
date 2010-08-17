#include <string.h>
#include "str_table.h"

void str_table_init(struct StrTable* table)
{
	int i, j;
	for(i=0; i<N_CHAR; i++){
		table->str[i].prefix=CLEAR;
		table->str[i].suffix=i;
	}
	for(i=0; i<N_PREFIX; i++)
		for(j=0; j<N_SUFFIX; j++)
			table->index[i][j]=CLEAR;
	for(j=0; j<N_SUFFIX; j++)
		table->index[CLEAR][j]=j;
	table->pos=START;
}

void str_table_add(struct StrTable* table, struct ConStr* con_str)
{
	table->str[table->pos]=*con_str;
	table->index[con_str->prefix][con_str->suffix]=table->pos;
	table->pos++;
}

int str_table_lookup(struct StrTable* table, struct ConStr* con_str)
{
	return table->index[con_str->prefix][con_str->suffix];
}

int str_table_get(struct StrTable* table, struct ConStr* con_str,
		char* str)
{
	int len;
	for(len=1; ; len++){
		*(--str)=con_str->suffix;
		if(con_str->prefix==CLEAR)break;
		con_str= table->str + con_str->prefix;
	}
	return len;
}
