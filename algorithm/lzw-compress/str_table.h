#ifndef  __STR_TABLE_H__
#define  __STR_TABLE_H__

#define  ORIG_CODE_LEN 8
#define  MAX_CODE_LEN 16
#define  N_PREFIX (1<<MAX_CODE_LEN)
#define  N_SUFFIX (1<<ORIG_CODE_LEN)
//#define  N_SUFFIX (1<<8)
#define  N_STR  N_PREFIX
#define  N_CHAR N_SUFFIX
#define  CLEAR 256
#define  END 257
#define  START 258
#define  MAX_STR_LEN 256

struct ConStr {
	unsigned short prefix;
	unsigned char suffix;
};
struct StrTable {
	struct ConStr str[N_STR];
	unsigned short index[N_PREFIX][N_SUFFIX];
	int pos;
};

void str_table_init(struct StrTable* table);
void str_table_add(struct StrTable* table, struct ConStr* con_str);
int str_table_lookup(struct StrTable* table, struct ConStr* con_str);
int str_table_get(struct StrTable* table, struct ConStr* con_str,
		char* str);

#endif  /*__STR_TABLE_H__*/
