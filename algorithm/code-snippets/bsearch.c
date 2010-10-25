#include <stdio.h>
#include <assert.h>

int bsearch(int* d, int n, int x)
{
        int s, e, m;
        for(s=0, e=n-1; s < e;){
                m = s + (e-s)/2;
                if(x <= d[m]) e = m;
                else s = m+1;
        }
        return x == d[m]? m: -1;
}

int main()
{
        int d[] = {1, 2, 3, 3, 5, 6, 7, 8};
        assert(bsearch(d, sizeof(d)/sizeof(int), 3) == 2);
        assert(bsearch(d, sizeof(d)/sizeof(int), 4) == -1);
        return 0;
}
