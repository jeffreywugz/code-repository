#ifndef _CONTAINER_H_
#define _CONTAINER_H_
#include <stdint.h>
#include "core.h"

struct list_head {
	struct list_head *next, *prev;
};

#define LIST_HEAD_INIT(name) { &(name), &(name) }

#define LIST_HEAD(name) \
	struct list_head name = LIST_HEAD_INIT(name)

#define INIT_LIST_HEAD(ptr) do { \
	(ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)

static inline void __list_add(struct list_head *newnode,
			      struct list_head *prev,
			      struct list_head *next)
{
	next->prev = newnode;
	newnode->next = next;
	newnode->prev = prev;
	prev->next = newnode;
}

static inline void list_add(struct list_head *newnode, struct list_head *head)
{
	__list_add(newnode, head, head->next);
}

static inline void list_add_tail(struct list_head *newnode, struct list_head *head)
{
	__list_add(newnode, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next)
{
	next->prev = prev;
	prev->next = next;
}

static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	entry->next = (struct list_head *) 0;
	entry->prev = (struct list_head *) 0;
}

static inline int list_empty(struct list_head *head)
{
	return head->next == head;
}

#define list_entry(ptr, type, member) \
	((type *)((char *)(ptr)-(unsigned long)(&((type *)0)->member)))

#define list_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); \
        	pos = pos->next)
#define list_for_each_prev(pos, head) \
	for (pos = (head)->prev; pos != (head); \
        	pos = pos->prev)
        	
#define list_for_each_entry(pos, head, member)				\
	for (pos = list_entry((head)->next, typeof(*pos), member);	\
	     &pos->member != (head);                                    \
	     pos = list_entry(pos->member.next, typeof(*pos), member))

typedef struct _mystack_t
{
        int capacity;
        int top;
        void** buf;
} mystack_t;

int stack_init(mystack_t* stack, void** buf, int capacity)
{
        stack->buf = buf;
        stack->capacity = capacity;
        stack->top = 0;
        return 0;
}

int stack_size(mystack_t* stack)
{
        return stack->top;
}

int stack_is_empty(mystack_t* stack)
{
        return stack_size(stack) == 0;
}

int stack_is_full(mystack_t* stack)
{
        return stack_size(stack) == stack->capacity;
}

int stack_push(mystack_t* stack, void* data)
{
        if(stack_is_full(stack))
                return -1;
        stack->buf[stack->top++] = data;
        return 0;
}

int stack_pop(mystack_t* stack, void** data)
{
        if(stack_is_empty(stack))
                return -1;
        *data = stack->buf[--stack->top];
        return 0;
}

int stack_destroy(mystack_t* stack)
{
        return 0;
}

#ifdef TEST
long test_stack()
{
        mystack_t stack;
        char* s1 = "just ";
        char* s2 = "for ";
        char* s3 = "fun!";
        char* s;
        char output[128] = "";
        char* buf[128];
        stack_init(&stack, (void**)buf, 20);
        stack_push(&stack, s3);
        stack_push(&stack, s2);
        stack_push(&stack, s1);
        while(!stack_is_empty(&stack)){
                stack_pop(&stack, (void**)&s);
                strcat(output, s);
        }
        cktrue(strcmp(output, "just for fun!") == 0, "stack test");
        return 0;
}
#endif

typedef struct _queue_t
{
        int capacity;
        int front, rear;
        void** buf;
} queue_t;

int queue_init(queue_t* queue, void** buf, int capacity)
{
        queue->buf = buf;
        queue->capacity = capacity;
        queue->front = 0;
        queue->rear = 0;
        return 0;
}

int queue_destroy(queue_t* queue)
{
        return 0;
}

int queue_size(queue_t* queue)
{
        return (queue->rear + queue->capacity - queue->front) % queue->capacity;
}

int queue_is_empty(queue_t* queue)
{
        return queue_size(queue) == 0;
}

int queue_is_full(queue_t* queue)
{
        return queue_size(queue) == queue->capacity-1;
}

int queue_push(queue_t* queue, void* data)
{
        if(queue_is_full(queue))
                return -1;
        queue->buf[queue->rear] = data;
        queue->rear = (queue->rear + 1) % queue->capacity;
        return 0;
}

int queue_pop(queue_t* queue, void** data)
{
        if(queue_is_empty(queue))
                return -1;
        *data = queue->buf[queue->front];
        queue->front = (queue->front + 1) % queue->capacity;
        return 0;
}

#ifdef TEST
long test_queue()
{
        queue_t queue;
        char* s1 = "just ";
        char* s2 = "for ";
        char* s3 = "fun!";
        char* s;
        char output[128] = "";
        char* buf[3];
        queue_init(&queue, (void**)buf, array_len(buf));
        ckerr(queue_push(&queue, s1));
        ckerr(queue_push(&queue, s2));
        ckerr(!queue_push(&queue, s3));
        while(!queue_is_empty(&queue)){
                queue_pop(&queue, (void**)&s);
                strcat(output, s);
        }
        cktrue(strcmp(output, "just for ") == 0, "queue test");
        return 0;
}
#endif

typedef struct _dict_item_t
{
        const char* key;
        void* val;
        struct _dict_item_t* next;
} dict_item_t;

typedef struct _dict_t
{
        int n_items;
        int capacity;
        dict_item_t** buckets;
} dict_t;

static uint32_t hash_idx(const char *str)
{
        uint32_t h = 5381;
        int c;
        while ((c = *str++))
                h = ((h << 5) + h) + c; /* hash * 33 + c */
        return h;
}

/*
static uint32_t hash_verify(const char* str)
{
        uint32_t h = 0;
        int c;
        while ((c = *str++))
                h = c + (h << 6) + (h << 16) - h;
        return h;
}
*/

int dict_init(dict_t* dict, dict_item_t** buf, int capacity)
{
        dict->buckets = buf;
        dict->capacity = capacity;
        dict->n_items = 0;
        memset(dict->buckets, 0, capacity * sizeof(dict_item_t*));
        return 0;
}

int dict_destroy(dict_t* dict)
{
        dict_item_t* item;
        for(int i = 0; i < dict->capacity; i++){
                for(item = dict->buckets[i]; item; item = item->next){
                        free((void*)item->key);
                        free(item);
                }
        }
        dict->n_items = 0;
        return 0;
}

int dict_size(dict_t* dict)
{
        return dict->n_items;
}

int dict_capacity(dict_t* dict)
{
        return dict->capacity;
}

static dict_item_t* dict_item_find(dict_t* dict, uint32_t idx, const char* key)
{
        dict_item_t* item;
        for(item = dict->buckets[idx % dict->capacity]; item; item = item->next)
                if(strcmp(item->key, key) == 0)break;
        return item;
}

static dict_item_t* dict_item_add(dict_t* dict, uint32_t idx, const char* key)
{
        dict_item_t** bucket;
        dict_item_t* item;
        bucket = &dict->buckets[idx % dict->capacity];
        item = (dict_item_t*)malloc(sizeof(dict_item_t));
        if(item == NULL)return NULL;
        item->key = strdup(key);
        item->next = *bucket;
        *bucket = item;
        dict->n_items++;
        return item;
}

int dict_set(dict_t* dict, const char* key, void* val)
{
        uint32_t idx;
        dict_item_t* item;
        idx = hash_idx(key);

        item = dict_item_find(dict, idx, key);
        if(item == NULL){
                item = dict_item_add(dict, idx, key);
        }
        item->val = val;
        return 0;
}

int dict_get(dict_t* dict, const char* key, void** val)
{
        uint32_t idx;
        dict_item_t* item;
        
        idx=hash_idx(key);
        item = dict_item_find(dict, idx, key);
        if(item == NULL)
                return -1;
        *val = item->val;
        return 0;
}

#ifdef TEST
long test_dict()
{
        dict_t dict;
        int err;
        char* key = "key1";
        char* val;
        dict_item_t* buf[2];
        ckerr(dict_init(&dict, buf, array_len(buf)));
        ckerr(!dict_get(&dict, key, (void**)&val));
        ckerr(dict_set(&dict, key, (void*)"val1"));
        ckerr(dict_set(&dict, key, (void*)"val1 change"));
        ckerr(dict_set(&dict, "key2", (void*)"val2"));
        ckerr(dict_set(&dict, "key3", (void*)"val3"));
        ckerr(!dict_get(&dict, "key not exist", (void**)&val));
        err = dict_get(&dict, "key1", (void**)&val);
        cktrue(err == 0 && strcmp(val, "val1 change") == 0);
        err = dict_get(&dict, "key3", (void**)&val);
        cktrue(err == 0 && strcmp(val, "val3") == 0);
        cktrue(dict_capacity(&dict) == array_len(buf) && dict_size(&dict) == 3);
        ckerr(dict_destroy(&dict));
        return 0;
}
#endif

typedef struct _IMAP {
        int64_t id;
        const void* value;
} IMAP;

int imap_fill(IMAP* table, int table_len, IMAP* defs, int defs_len)
{
        memset(table, 0, sizeof(IMAP)*table_len);
        for(int i = 0; i < defs_len; i++){
                table[defs[i].id] = defs[i];
        }
        return 0;
}

const void* imap_get_value(IMAP* table, int table_len, int id)
{
        return (id >= 0 || id < table_len)? table[id].value: NULL;
}

const void* imap_get_value_with_default(IMAP* table, int table_len, int id, const void* default_)
{
        const void* value = imap_get_value(table, table_len, id);
        return value? value: default_;
}

const char* imap_get_name(IMAP* table, int table_len, int id, const char* default_)
{
        return (const char*)imap_get_value_with_default(table, table_len, id, default_);
}

typedef struct pair_t {
        const char* key;
        void* value;
} pair_t;

pair_t* assoc_find(pair_t* li, int n, const char* key)
{
        int i;
        for(i=0; i < n && li->key; i++, li++){
                if(!strcmp(key, li->key))break;
        }
        return i == n? NULL: li;
}

void* assoc_get(pair_t* li, int n, const char* key)
{
        pair_t* p = assoc_find(li, n, key);
        if(p == NULL)return NULL;
        if(p->key == NULL)return NULL;
        return (void*)(p->value);
}

void* assoc_set(pair_t* li, int n, const char* key, const void* value)
{
        pair_t* p = assoc_find(li, n, key);
        if(p == NULL)return NULL;
        p->key = key;
        p->value = (void*)value;
        return (void*)value;
}

void assoc_print(pair_t* li, int n)
{
        for(int i=0; i < n && li->key; i++, li++){
                fprintf(stderr, "%s = %s\n", li->key, (char*)li->value);
        }
}

#ifdef TEST
int test_assoc()
{
        pair_t li[32] = {{"abc", (void*)"cba"}, {"def", (void*)"fed"}, {"ghi", (void*)"ihg"}, {0,}};
        cktrue(strcmp("cba", (const char*)assoc_get(li, array_len(li), "abc")) == 0);
        cktrue(assoc_get(li, array_len(li), "xyz") == NULL);
        return 0;
}
#endif

#endif /* _CONTAINER_H_ */
