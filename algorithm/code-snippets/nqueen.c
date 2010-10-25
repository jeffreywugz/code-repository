#include <stdio.h>

#define swap(a, b) {typeof(a) t; t=a; a=b; b=t;}
#define N 128

void print_array(int* pos, int n)
{
        int i;
        for(i = 0; i < n; i++)
                printf("%d ", pos[i]);
        printf("\n");
}

int compatible(int* pos, int k)
{
        int i;
        for(i = 0; i < k; i++){
                if(pos[i] == pos[k] || pos[k]-pos[i] == k-i || pos[k]-pos[i] == i-k)return 0;
        }
        return 1;
}

int nqueen(int n)
{
        int total = 0;
        int d = 0;
        int pos[N];
        pos[0] = 0;
        while(1){
                if(d == n){
                        printf("%dth solution\n", ++total);
                        print_array(pos, n);
                }
                if(d == n || pos[d] == n){
                        d--;
                        if(d < 0)break;
                        pos[d]++;
                } else if(!compatible(pos, d)){
                        pos[d]++;
                } else {
                        pos[++d] = 0;
                }
        }
}

int nqueen_recursive(int* pos, int n, int d)
{
        static int total = 0;
        int i;
        if(!compatible(pos, d-1))return;
        if(d == n){
                printf("%dth solution\n", ++total);
                print_array(pos, n);
        }
        for(i = 0; i < n; i++){
                pos[d] = i;
                nqueen_recursive(pos, n, d+1);
        }
}

int nqueen2(int n)
{
        int pos[N];
        nqueen_recursive(pos, n, 0);
}

int main()
{
        nqueen(8);
        nqueen2(8);
        return 0;
}
