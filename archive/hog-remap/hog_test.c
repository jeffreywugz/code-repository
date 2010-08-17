#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hog_user.h"
#include "util.h"


static void seq_acc(int* hot, int* hog, int hot_size, int hog_size)
{
        int i, j;
        for(i = 0; i < hog_size; i += hot_size){
#pragma omp parallel for
                for(j = 0; j < hot_size; j++){
                        hot[j] += hog[i+j];
                }
        }
}

#define HOT_SIZE (1<<19)
#define HOG_SIZE (HOT_SIZE*512)
int hot_array[HOT_SIZE/sizeof(int)];
int hog_array[HOG_SIZE/sizeof(int)];

void prepare(int* hot, int* hog, int hot_size, int hog_size)
{
        int i;
        for(i = 0; i < hog_size/sizeof(int); i++)
                hog[i] = 1;
        for(i = 0; i < hot_size/sizeof(int); i++)
                hot[i] = 0;
}

void profile_run(int* hot, int* hog, int hot_size, int hog_size)
{
        double t;
        printf("hot size: %s\n", i2q(hot_size));
        printf("hog size: %s\n", i2q(hog_size));
        
        t =  time_it(
                seq_acc(hot, hog, hot_size/sizeof(int), hog_size/sizeof(int)),
                1);
        printf("time: %f ms\n", t);
}

static void test_static()
{
        printf("test static:\n");
        printf("start: %p, end: %p\n", hog_array, hog_array + HOG_SIZE/sizeof(int));
        
        prepare(hot_array, hog_array, HOT_SIZE, HOG_SIZE);
        profile_run(hot_array, hog_array, HOT_SIZE, HOG_SIZE);
}

static void test_dynamic(int hot_size, int hog_size)
{
        int* hog;
        int* hot;
        printf("test dynamic:\n");
        hot = malloc(hot_size);
        assert(hot);
        hog = hog_alloc(hog_size);
        assert(hog);

        prepare(hot, hog, hot_size, hog_size);
        profile_run(hot, hog, hot_size, hog_size);
        free(hot);
        hog_free(hog, hog_size);
}

int main(int argc, char** argv)
{
        int hot_size;
        int hog_size;
        assert(argc == 2);
        hot_size = q2i(argv[1]);
        assert(hot_size >= (1<<10) && hot_size <= (1<<20));
        hog_size = hot_size * 512;
        
        test_static();
        test_dynamic(hot_size, hog_size);
        return 0;
}
