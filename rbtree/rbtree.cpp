#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <map>
using namespace std;

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

const bool RBT_BLACK = true;
const bool RBT_RED = false;
typedef struct RbtNode
{
  bool color;
  void* key;
  void* value;
  struct RbtNode* left;
  struct RbtNode* right;
} RbtNode;

bool is_red(const RbtNode* node)
{
  return (NULL == node)? false: (RBT_RED == node->color);
}

typedef int64_t (*comp_func_t)(const void*, const void*);
const RbtNode* search(comp_func_t compare, const RbtNode* root, const void* key)
{
  const RbtNode* target = NULL;
  int64_t cmp = 0;
  if (NULL == root) {
    target = NULL;
  } else if ((cmp = compare(key, root->key)) == 0) {
    target = root;
  } else if (cmp < 0) {
    target = search(compare, root->left, key);
  } else {
    target = search(compare, root->right, key);
  }
  return target;
}

typedef void* (*handle_func_t)(RbtNode* node, void* args);
void* rbt_map(handle_func_t func, RbtNode* root, void* args)
{
  void* result = NULL;
  if (NULL == root)
    return NULL;
  result = func(root, args);
  rbt_map(func, root->left, args);
  rbt_map(func, root->right, args);
  return result;
}

RbtNode* insert(comp_func_t compare, RbtNode* root, RbtNode* target, RbtNode** discarded)
{
  int64_t cmp = 0;
  if (NULL == root) {
    root = target;
  } else if ((cmp = compare(target->key, root->key)) == 0) {
    target->left = root->left;
    target->right = root->right;
    *discarded = root;
    root = target;
  } else if (cmp < 0) {
    root->left = insert(compare, root->left, target, discarded);
  } else {
    root->right = insert(compare, root->right, target, discarded);
  }
  return root;
}

RbtNode* rotR(RbtNode* root)
{
  RbtNode* new_root = root->left;
  root->left = new_root->right;
  new_root->right = root;
  new_root->color = root->color;
  root->color = RBT_RED;
  return new_root;
}

RbtNode* rotL(RbtNode* root)
{
  RbtNode* new_root = root->right;
  root->right = new_root->left;
  new_root->left = root;
  new_root->color = root->color;
  root->color = RBT_RED;
  return new_root;
}

RbtNode* colorFlip(RbtNode* root)
{
  root->color = !root->color;
  root->left->color = !root->left->color;
  root->right->color = !root->right->color;
  return root;
}

RbtNode* fixup(RbtNode* root)
{
  if (is_red(root->right)) root = rotL(root);
  if(is_red(root->left) && is_red(root->left->left)) root = rotR(root);
  if (is_red(root->left) && is_red(root->right))colorFlip(root);
  return root;
}

RbtNode* _rbt_insert(comp_func_t compare, RbtNode* root, RbtNode* target, RbtNode** discarded)
{
  int64_t cmp = 0;
  if (NULL == root)return target;

  if ((cmp = compare(target->key, root->key)) == 0) {
    root->key = target->key;
    root->value = target->value;
    *discarded = target;
  } else if (cmp < 0) {
    root->left = _rbt_insert(compare, root->left, target, discarded);
  } else {
    root->right = _rbt_insert(compare, root->right, target, discarded);
  }

  root = fixup(root);
  return root;
}

RbtNode* rbt_insert(comp_func_t compare, RbtNode* root, RbtNode* target, RbtNode** discarded)
{
  *discarded = NULL;
  root = _rbt_insert(compare, root, target, discarded);
  if (NULL != root)
    root->color = RBT_BLACK;
  return root;
}

RbtNode* move_red_right(RbtNode* root)
{
  root = colorFlip(root);
  if (is_red(root->left->left)) {
    root = rotR(root);
    root =colorFlip(root);
  }
  return root;
}

RbtNode* move_red_left(RbtNode* root)
{
  root = colorFlip(root);
  if (is_red(root->right->left)) {
    root->right = rotR(root->right);
    root = rotL(root);
    root =colorFlip(root);
  }
  return root;
}

RbtNode* _rbt_delete_max(RbtNode* root, RbtNode** discarded)
{
  if (NULL == root)
    return NULL;
  if (is_red(root->left))
    root = rotR(root);
  if (NULL == root->right){
    *discarded = root;
    return NULL;
  }
  if (!is_red(root->right) && !is_red(root->right->left))
    root = move_red_right(root);
  root->right = _rbt_delete_max(root->right, discarded);
  root = fixup(root);
  return root;
}

RbtNode* rbt_delete_max(RbtNode* root, RbtNode** discarded)
{
  *discarded = NULL;
  root = _rbt_delete_max(root, discarded);
  if (NULL != root)
    root->color = RBT_BLACK;
  return root;
}

RbtNode* _rbt_delete_min(RbtNode* root, RbtNode** discarded)
{
  if (NULL == root)
    return NULL;
  if (NULL == root->left) {
    *discarded = root;
    return NULL;
  }
  if (!is_red(root->left) && !is_red(root->left->left))
    root = move_red_left(root);
  root->left = _rbt_delete_min(root->left, discarded);
  root = fixup(root);
  return root;
}

RbtNode* rbt_delete_min(RbtNode* root, RbtNode** discarded)
{
  *discarded = NULL;
  root = _rbt_delete_min(root, discarded);
  if (NULL != root)
    root->color = RBT_BLACK;
  return root;
}

RbtNode* _rbt_delete(comp_func_t compare, RbtNode* root, void* key, RbtNode** discarded)
{
  if (NULL == root) return NULL;
  if (compare(key, root->key) <= 0) {
     if (root->left && !is_red(root->left) && !is_red(root->left->left))
      root = move_red_left(root);
    if (compare(key, root->key) == 0) {
      root->left = _rbt_delete_max(root->left, discarded);
      if (*discarded){
        root->key = (*discarded)->key;
        root->value = (*discarded)->value;
      } else {
        *discarded = root;
        return NULL;
      }
    } else {
     root->left = _rbt_delete(compare, root->left, key, discarded);
    }
  } else {
    if (is_red(root->left)) root = rotR(root);
     if (root->right && !is_red(root->right) && !is_red(root->right->left))
      root = move_red_right(root);
    root->right = _rbt_delete(compare, root->right, key, discarded);
  }
  root = fixup(root);
  return root;
}

RbtNode* rbt_delete(comp_func_t compare, RbtNode* root, void* key, RbtNode** discarded)
{
  *discarded = NULL;
  root = _rbt_delete(compare, root, key, discarded);
  if (NULL != root)
    root->color = RBT_BLACK;
  return root;
}

int64_t int_compare(const int64_t x, const int64_t y)
{
  return x - y;
}

RbtNode* rbt_new_node_for_test(int64_t key)
{
  RbtNode* node = NULL;
  if (NULL == (node = (RbtNode*)malloc(sizeof(RbtNode))))
  {}
  else
  {
    node->color = RBT_RED;
    node->key = (void*)key;
    node->value = (void*)key;
    node->left = NULL;
    node->right = NULL;
  }
  return node;
}

void* rbt_free_node(RbtNode* node, void* args)
{
  free(node);
  return NULL;
}

void _dump_rb_tree(const RbtNode* root, int black_depth, int red_depth)
{
  printf("\n%*s", black_depth * 8 + red_depth*2, "");
  if (NULL == root) {
    printf("Nil,");
    return;
  }
  printf("([%s,%ld],", is_red(root)? "R":"B", (int64_t)root->value);
  if (is_red(root))
    red_depth++;
  else {
    black_depth++;
    red_depth = 0;
  }
  _dump_rb_tree(root->left, black_depth, red_depth);
  _dump_rb_tree(root->right, black_depth, red_depth);
  printf("),");
}

void dump_rb_tree(const char* msg, const RbtNode* root)
{
  printf("--------------rb_tree_begin[%s]---------------", msg);
  _dump_rb_tree(root, 0, 0);
  printf("\n--------------rb_tree_end[%s]---------------\n", msg);
}

char* sf(const char* format, ...)
{
  static char buf[1<<10];
  va_list ap;
  int64_t len = 0;
  va_start(ap, format);
  len = vsnprintf(buf, sizeof(buf), format, ap);
  assert(len > 0 && len < (int64_t)sizeof(buf));
  va_end(ap);
  return buf;
}

int dump_insert_delete_flow(const int64_t n, int64_t* items)
{
  RbtNode* root = NULL;
  RbtNode* discarded = NULL;
  printf("dump_insert_flow(n=%ld)\n", n);
  for(int64_t i = 0; i < n; i++) {
    root = rbt_insert((comp_func_t)int_compare, root, rbt_new_node_for_test(items[i]), &discarded);
    if (NULL != discarded) {
      free(discarded);
      discarded = NULL;
    }
    dump_rb_tree(sf("insert %ld", items[i]), root);
  }
  for(int64_t i = 0; i < n; i++) {
    root = rbt_delete((comp_func_t)int_compare, root, (void*)items[i], &discarded);
    if (NULL != discarded) {
      free(discarded);
      discarded = NULL;
    }
    dump_rb_tree(sf("delete %ld", items[i]), root);
  }
  return 0;
}

int test_rb_tree(const int64_t n, const int64_t* items)
{
  int err = 0;
  RbtNode* root = NULL;
  RbtNode* discarded = NULL;
  int64_t start_time = 0;
  int64_t n_hit = 0;

  printf("--------------------rb-tree--------------------\n");
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    root = rbt_insert((comp_func_t)int_compare, root, rbt_new_node_for_test(items[i]), &discarded);
    if (NULL != discarded) {
      //free(discarded);
      discarded = NULL;
    }
  }
  printf("insert %ld nodes: time=%fs\n", n, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    if (NULL != search((comp_func_t)int_compare, root, (void*)rand())) {
      n_hit++;
    }
  }
  printf("search %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    if (NULL != search((comp_func_t)int_compare, root, (void*)items[i])) {
      n_hit++;
    }
  }
  printf("search %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    root = rbt_delete((comp_func_t)int_compare, root, (void*)rand(), &discarded);
    if (NULL != discarded) {
      n_hit++;
      //free(discarded);
      discarded = NULL;
    }
  }
  printf("delete %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    root = rbt_delete((comp_func_t)int_compare, root, (void*)items[i], &discarded);
    if (NULL != discarded) {
      n_hit++;
      //free(discarded);
      discarded = NULL;
    }
  }
  printf("delete %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);
  return err;
}

int test_stl_map(const int64_t n, const int64_t* items)
{
  int64_t start_time = 0;
  int64_t n_hit = 0;
  map<int64_t, int64_t> int_map;
  printf("--------------------stl-map--------------------\n");
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    int_map[items[i]] = items[i];
  }
  printf("insert %ld nodes: time=%fs\n", n, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    map<int64_t, int64_t>::iterator it= int_map.find(rand());
    if (it != int_map.end()){
      n_hit++;
    }
  }
  printf("search %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    map<int64_t, int64_t>::iterator it= int_map.find(items[i]);
    if (it != int_map.end()){
      n_hit++;
    }
  }
  printf("search %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    if (int_map.erase(rand()) > 0)
      n_hit++;
  }
  printf("delete %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  n_hit = 0;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    if (int_map.erase(items[i]) > 0)
      n_hit++;
  }
  printf("delete %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);
  return 0;
}

int profile(const char* msg, const int64_t n, const int64_t* items)
{
  printf("--------------------%s--------------------\n", msg);
  test_rb_tree(n, items);
  test_stl_map(n, items);
  return 0;
}

int64_t* gen_seq_items(int64_t n, int64_t* items)
{
  for(int64_t i = 0; i < n; i++)
    items[i] = i;
  return items;
}

int64_t* gen_rand_items(int64_t n, int64_t* items)
{
  for(int64_t i = 0; i < n; i++)
    items[i] = rand();
  return items;
}

int main(int argc, char *argv[])
{
  int err = 0;
  const char* cmd = NULL;
  int64_t *items = NULL;
  int64_t n = 0;

  cmd = (argc > 1)? argv[1]: "profile";
  n = argc > 2 ? atoi(argv[2]): 10;
  srand(time(NULL));
  printf("cmd=%s n=%ld\n", cmd, n);

  if (NULL == (items = (int64_t*)malloc(sizeof(int64_t)*n))){
    err = -1;
  }else if (0 == strcmp("profile", cmd)) {
    profile("rand-seq", n, gen_rand_items(n, items));
    profile("ordered-seq", n, gen_seq_items(n, items));
  } else if (0 == strcmp("dump_insert_delete_flow", cmd)) {
    err = dump_insert_delete_flow(n, gen_rand_items(n, items));
  }
  if (NULL != items) {
    free(items);
  }
  return err;
}
