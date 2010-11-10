#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#define swap(x, y) {typeof(x) t; t=x; x=y; y=t;}
typedef void (*Sorter)(int *d, int len);

typedef struct Map {
        const char* name;
        void* data;
} Map;

void* map_get(Map* map, const char* name)
{
        Map* m;
        for(m = map; m->name; m++){
                if(strcmp(m->name, name) == 0)return m->data;
        }
        return NULL;
}

bool check_sort(int *d, int len)
{
        int i;
        for(i = 0; i < len-1; i++)
                if(d[i] > d[i+1])return false;
        return true;
}

int test_sorter(Sorter sorter, int n_items, int repeat)
{
        int *d;
        int i, j;
        srandom(time(NULL));
        d = malloc(sizeof(int) * n_items);
        assert(d);
        for(i = 0; i < repeat; i++){
                for(j = 0; j < n_items; j++)
                        d[j] = random();
                sorter(d, n_items);
                assert(check_sort(d, n_items));
        }
        free(d);
        return 0;
}

void bubble_sort(int *d, int len)
{
        int i, j;
        for(i = len-1; i > 0; i--)
                for(j = 0; j < i; j++){
                        if(d[j] > d[j+1])
                                swap(d[j], d[j+1]);
                }
}

void selection_sort(int *d, int len)
{
        int i, j, i_min;
        for(i = 0; i < len-1; i++){
                i_min = i;
                for(j = i+1; j < len; j++)
                        if(d[j] < d[i_min])
                                i_min = j;
                swap(d[i], d[i_min]);
        }
}

void insertion_sort(int *d, int len)
{
        int i, j, cur;
        for(i = 1; i < len; i++){
                cur = d[i];
                for(j = i-1; d[j] > cur && j >= 0; j--)
                        d[j+1] = d[j];
                d[j+1] = cur;
        }
}

int partion(int *d, int low, int high)
{
        int pivot = d[low];
        int left, right;
        for(left=low, right=high; left < right;){
                while(left < right && d[right] >= pivot)
                        right--;
                d[left] = d[right];
                while(left < right && d[left] <= pivot)
                        left++;
                d[right] = d[left];
        }
        d[left] = pivot;
        return left;
}

void quickSort(int arr[], int left, int right) {
      int i = left, j = right;
      int tmp;
      int pivot = arr[(left + right) / 2];
      while (i <= j) {
            while (arr[i] < pivot)
                  i++;
            while (arr[j] > pivot)
                  j--;
            if (i <= j) {
                  tmp = arr[i];
                  arr[i] = arr[j];
                  arr[j] = tmp;
                  i++;
                  j--;
            }
      };

      if (left < j)
            quickSort(arr, left, j);
      if (i < right)
            quickSort(arr, i, right);

}

void _quick_sort(int *d, int low, int high)
{
        int pivot;
        if(low >= high)return;
        pivot = partion(d, low, high);
        _quick_sort(d, low, pivot-1);
        _quick_sort(d, pivot+1, high);
}

void quick_sort(int *d, int len)
{
        _quick_sort(d, 0, len-1);
}

#define max2(a, e, x, m) ((x<e && a[x]>=a[m])? x: m)
/*     0
 *   1  2
 * 3 4 5 6
 *  */
int sift_down(int* h, int s, int e)
{
        int m;
        while(s < e){
                m = s;
                m = max2(h, e+1, 2*s+1, m);
                m = max2(h, e+1, 2*s+2, m);
                if(s == m)break;
                swap(h[s], h[m]);
                s = m;
        }
}

void heapify(int* h, int n)
{
        int s;
        for(s = (n-2)/2; s >= 0; s--)
                sift_down(h, s, n-1);
}

void heap_sort(int* h, int n)
{
        int i;
        heapify(h, n);
        for(i = n-1; i > 0; i--){
                swap(h[0], h[i]);
                sift_down(h, 0, i-1);
        }
}

int main(int argc, char* argv[])
{
        int i;
        Map sorters[] = {{"bubble", bubble_sort},
                         {"selection", selection_sort},
                         {"insertion", insertion_sort},
                         {"quick", quick_sort},
                         {"heap", heap_sort},
                         {NULL, NULL}};
        Sorter sorter;
        if(argc != 4){
                fprintf(stderr, "%s sorter n_items repeat\n", argv[0]);
                fprintf(stderr, "sorters: ");
                for(i = 0; sorters[i].name; i++)
                        fprintf(stderr, "%s ", sorters[i].name);
                fprintf(stderr, "\n");
                return 1;
        }
        sorter = map_get(sorters, argv[1]);
        if(!sorter)return -2;
        test_sorter(sorter, atoi(argv[2]), atoi(argv[3]));
        return 0;
}
