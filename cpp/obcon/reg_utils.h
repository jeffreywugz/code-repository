#include "regex.h"
#include "common/ob_define.h"
#include "common/data_buffer.h"
using namespace oceanbase::common;

int extract_reg_match(const char* str, ObDataBuffer& buf, int max_n_match, int& n_match, regmatch_t* regmatch, char** match);
int reg_match(const char* pat, const char* str, ObDataBuffer& buf, int max_n_match, int& n_match, char** match);
int reg_helper(const char* pat, const char *str, int& n_match, char**& match);
int reg_test(const char* pat, const char* str);
int __reg_parse(const char* pat, const char* str, ObDataBuffer& buf, ...);
#define reg_parse(pat, str, buf, ...) __reg_parse(pat, str, buf, ##__VA_ARGS__, NULL)
