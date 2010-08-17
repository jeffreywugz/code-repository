#include <iostream>

using namespace std;

const int MAX_LEN=300;

char* str2li(char* li, char* str)
{
	memset(li, 0, MAX_LEN);
	int i=strlen(str)-1;
	for(int j=0; i>=0; j++,i--)
		li[j]=str[i]-'0';
	return li;
}

char* li2str(char* str, char* li)
{
	int i,j;
	for(i=MAX_LEN-1; i && li[i]==0; i--)
		;
	for(j=0; i>=0; i--,j++)
		str[j]='0'+li[i];
	str[j]=0;
	return str;
}

char* li_cpy(char* dest, char* src)
{
	memcpy(dest, src, MAX_LEN);
	return dest;
}

char* li_add(char* sum, char* a1, char* a2)
{
	memset(sum, 0, MAX_LEN);
	for(int c=0,i=0; i<MAX_LEN; i++){
		sum[i]=a1[i]+a2[i]+c;
		c=sum[i]/10;
		sum[i]%=10;
	}
}

char* li_mul(char* pro, char* a1, char* a2)
{
	memset(pro, 0, MAX_LEN);
	for(int i=0; i<MAX_LEN; i++){ //遍历a2的每一位
		char tmp[MAX_LEN],tmp_pro[MAX_LEN];
		for(int c=0,j=0; j<MAX_LEN; j++){ //遍历a1的每一位
			tmp[j]=a1[j]*a2[i]+c;
			c=tmp[j]/10;
			tmp[j]%=10;
		}
		memmove(tmp+i, tmp, MAX_LEN-i);
		memset(tmp, 0, i);
		li_add(tmp_pro, pro, tmp);
		li_cpy(pro,tmp_pro);
	}
	return pro;
}

int main()
{
	char a1[MAX_LEN], a2[MAX_LEN], 
	     sum[MAX_LEN], pro[MAX_LEN], str[MAX_LEN];
	str2li(a1, "31415926");
	str2li(a2, "314159265358794");
	//li_add(sum, a1, a2);
	li_mul(pro, a1, a2);
	printf("%s\n", li2str(str, pro));
	return 0;
}
