#include <stdio.h>

#define swap(a, b) {typeof(a) t; t=a; a=b; b=t;}

void perm_recursive(char* list, int n, int k)
{
        int i;
	if(k == n)printf("%s\n", list);
	for(i = k; i < n; i++){
		swap(list[k], list[i]);
		perm_recursive(list, n, k+1);
		swap(list[k], list[i]);
	}
}

void perm(char* list, int n)
{
	perm_recursive(list, n, 0);
}

#define N 128
/*
 *   parent <- depth 
 *   /     \
 *child1 -> child2
 * ^  
 * |
 *iter
 *  */
void perm2(char* list, int n)
{
        int d = 0;
        int pos[N];
        pos[0] = 0;
        while(1){
                if(d == n)printf("%s\n", list);
                if(d == n || pos[d] >= n){ /* backtrace */
                        d--;
                        if(d < 0)break;
                        swap(list[d], list[pos[d]]);
                        pos[d]++;
                        if(pos[d] >= n)continue;
                        swap(list[d], list[pos[d]]);
                } else {
                        d++;
                        if(d == n)continue;
                        pos[d] = d;
                }
        }
}

int main()
{
	char list[]="abc";
        printf("perm([%s])\n", list);
	perm(list, sizeof(list)-1);
        printf("perm2([%s])\n", list);
        perm2(list, sizeof(list)-1);
        printf("list=[%s]\n", list);
	return 0;
}
