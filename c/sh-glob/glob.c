#include <stdbool.h>
#define _GNU_SOURCE
#include <string.h>
#include <assert.h>
#include <stdio.h>

// prerequisite: pat != NULL && str != NULL
bool glob(char* _pat, char* str)
{
  char* pat = strdupa(_pat);
  char* end_pat = pat;
  char* start_str = NULL;
  char end_pat_char = 0;
  bool star_prefix = false;
  printf("glob('%s', '%s')\n", pat, str);
  while(*pat)
  {
    end_pat = strchrnul(pat, '*');
    end_pat_char = *end_pat;
    *end_pat = 0;
    start_str = strstr(str, pat);
    *end_pat = end_pat_char;
    if (!start_str || (!star_prefix && start_str != str))break;
    str = start_str + (end_pat - pat);
    if (end_pat_char){
      star_prefix = true;
      end_pat++;
      if (!*end_pat)str += strlen(str);
    }
    pat = end_pat;
  }
  return !*pat && !*str;
}

int test_glob()
{
  assert(glob("*", ""));
  assert(glob("*", "a"));
  assert(glob("*", "ab"));
  assert(!glob("a*b", "cab"));
  assert(glob("a*b", "ab"));
  assert(glob("a*b", "acb"));
  assert(glob("a*b*", "acb"));
  assert(glob("*a*b*", "acb"));
  assert(glob("*a**b*", "acb"));
}

int main()
{
  return test_glob();
}
