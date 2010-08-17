#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "dict.h"
#ifndef _DICT_H_
#define _DICT_H_
//<<<header


#include <stdint.h>

#define MAX_DICT_CAPACITY 1024

struct DictItem
{
        const char* key;
        void* val;
        struct DictItem* next;
};
typedef struct DictItem DictItem;

struct Dict
{
    uint32_t n_items;
    uint32_t capacity;
    DictItem* buckets[MAX_DICT_CAPACITY];
};
typedef struct Dict Dict;

//header>>>
#endif  /* _DICT_H_ */

//<<<func_list|get_func_list
static uint32_t hash_idx(const char *str)
{
        uint32_t h = 5381;
        int c;
        while ((c = *str++))
                h = ((h << 5) + h) + c; /* hash * 33 + c */
        return h;
}

static uint32_t hash_verify(const char* str)
{
        uint32_t h = 0;
        int c;
        while ((c = *str++))
                h = c + (h << 6) + (h << 16) - h;
        return h;
}

int dict_init(Dict* dict, int capacity)
{
        assert(capacity < MAX_DICT_CAPACITY);
        dict->capacity = capacity;
        dict->n_items = 0;
        memset(dict->buckets, 0, sizeof(dict->buckets));
        return 0;
}

int dict_destroy(Dict* dict)
{
        int i;
        DictItem* item;
        for(i = 0; i < dict->capacity; i++){
                for(item = dict->buckets[i]; item; item = item->next){
                        free((void*)item->key);
                        free(item);
                }
        }
        dict->n_items = 0;
        return 0;
}

int dict_size(Dict* dict)
{
        return dict->n_items;
}

int dict_capacity(Dict* dict)
{
        return dict->capacity;
}

static DictItem* dict_item_find(Dict* dict, uint32_t idx, const char* key)
{
        DictItem* item;
        for(item = dict->buckets[idx % dict->capacity]; item; item = item->next)
                if(strcmp(item->key, key) == 0)break;
        return item;
}

static DictItem* dict_item_add(Dict* dict, uint32_t idx, const char* key)
{
        DictItem** bucket;
        DictItem* item;
        bucket = &dict->buckets[idx % dict->capacity];
        item = malloc(sizeof(DictItem));
        if(item == NULL)panic("no mem!");
        item->key = strdup(key);
        item->next = *bucket;
        *bucket = item;
        dict->n_items++;
        return item;
}

int dict_set(Dict* dict, const char* key, void* val)
{
        uint32_t idx,verify;
        DictItem* item;
        idx = hash_idx(key);
        verify = hash_verify(key);

        item = dict_item_find(dict, idx, key);
        if(item == NULL){
                item = dict_item_add(dict, idx, key);
        }
        item->val = val;
        return 0;
}

int dict_get(Dict* dict, const char* key, void** val)
{
        uint32_t idx,verify;
        DictItem* item;
        
        idx=hash_idx(key);
        verify=hash_verify(key);
        item = dict_item_find(dict, idx, key);
        if(item == NULL)
                return -1;
        *val = item->val;
        return 0;
}
//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

START_TEST(dict_test)
{
        Dict dict;
        int err;
        char* key = "key1";
        char* val;
        err = dict_init(&dict, 2);
        fail_unless(err == 0);
        err = dict_get(&dict, key, (void**)&val);
        fail_unless(err < 0);
        err = dict_set(&dict, key, "val1");
        err = dict_set(&dict, key, "val1 change");
        err = dict_set(&dict, "key2", "val2");
        err = dict_set(&dict, "key3", "val3");
        err = dict_get(&dict, "key not exist", (void**)&val);
        fail_unless(err < 0);
        err = dict_get(&dict, "key1", (void**)&val);
        fail_unless(err == 0 && strcmp(val, "val1 change") == 0);
        err = dict_get(&dict, "key3", (void**)&val);
        fail_unless(err == 0 && strcmp(val, "val3") == 0);
        fail_unless(dict_capacity(&dict) == 2 && dict_size(&dict) == 3);
        err = dict_destroy(&dict);
        fail_unless(err == 0);
}END_TEST
quick_define_tcase_reg(dict)
#endif /* NOCHECK */
//test>>>
