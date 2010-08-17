#include <stdio.h>
#include <string.h>

inline int prefix_match(char* p, int j, char c, int* b)
{
	while(j!=-1 && p[j]!=c)
		j=b[j];
	return j;
}

void get_backward_array(char* p, int* b)
{
	int j;
	for(j=1,b[0]=-1; p[j]; j++)
		b[j]=prefix_match(p, b[j-1], p[j-1], b)+1;
}

int search(char* s, char* p)
{
	int i, j, len=strlen(p), b[len];
	get_backward_array(p, b);
	for(i=0,j=0; p[j] && s[i]; i++,j++)
		j=prefix_match(p, j, s[i], b);
	return p[j]? -1: i-len;
}

void print_(int* b, int n)
{
	int i;
	for(i=0; i<n; i++)
		printf("%d: %d\n", i, b[i]);
}

int power_of_str(char* s)
{
	int m, n=strlen(s), b[n+1];
	s[n]=-1; s[n+1]=0;
	get_backward_array(s, b);
	m=n-b[n];
	if(n%m)return 1;
	return n/m;
}

int	main(int argc,char*argv[])
{
	char s[100], p[100];
	while(scanf("%s", s)!=EOF)
		printf("%d\n", power_of_str(s));
	return 0;
}

