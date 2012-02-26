#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
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

bool is_red(RbtNode* node)
{
  return NULL == node? false: RBT_RED == node->color;
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
  return new_root;
}

RbtNode* rotL(RbtNode* root)
{
  RbtNode* new_root = root->right;
  root->right = new_root->left;
  new_root->left = root;
  return new_root;
}

RbtNode* rbt_insert(comp_func_t compare, RbtNode* root, RbtNode* target, RbtNode** discarded)
{
  int64_t cmp = 0;
  if (NULL == root) {
    return target;
  }

  if (is_red(root->left) && is_red(root->left->left)) {
    root = rotR(root);
    root->left->color  = RBT_BLACK;
  }

  if ((cmp = compare(target->key, root->key)) == 0) {
    root->key = target->key;
    root->value = target->value;
    *discarded = target;
  } else if (cmp < 0) {
    root->left = insert(compare, root->left, target, discarded);
  } else {
    root->right = insert(compare, root->right, target, discarded);
  }

  if (is_red(root->right)) {
    root = rotL(root);
    root->color = root->left->color;
    root->left->color = RBT_RED;
  }
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
    node->value = (void*)~key;
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

void* dump_rb_tree(RbtNode* root, int64_t black_depth, int64_t red_depth)
{
  printf("\n%*s", black_depth * 8 + red_depth*2, "");
  if (NULL == root) {
    printf("Nil,");
    return NULL;
  }
  printf("([%s,%ld],", is_red(root)? "R":"B", root->value);
  if (is_red(root))
    red_depth++;
  else {
    black_depth++;
    red_depth = 0;
  }
  dump_rb_tree(root->left, black_depth, red_depth);
  dump_rb_tree(root->right, black_depth, red_depth);
  printf("),");
}

int test_rb_tree(int64_t n)
{
  int err = 0;
  RbtNode* root = NULL;
  RbtNode* discarded = NULL;
  int64_t start_time = 0;
  int64_t n_hit = 0;

  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    root = rbt_insert((comp_func_t)int_compare, root, rbt_new_node_for_test(i), &discarded);
    printf("--------------rb_tree_begin---------------");
    dump_rb_tree(root, 0, 0);
    printf("\n--------------rb_tree_end---------------\n");
    if (NULL != discarded) {
      free(discarded);
      discarded = NULL;
    }
  }
  printf("insert %ld nodes: time=%fs\n", n, (get_usec() - start_time)/1e6);

  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    if (NULL != search((comp_func_t)int_compare, root, (void*)rand())) {
      n_hit++;
    }
  }
  printf("search %ld keys: n_hit=%ld, time=%fs\n", n, n_hit, (get_usec() - start_time)/1e6);

  start_time = get_usec();
  rbt_map(rbt_free_node, root, NULL);
  printf("destroy %ld node: time=%fs\n", n, (get_usec() - start_time)/1e6);
  return err;
}

int test_map(int64_t n)
{
  int64_t start_time = 0;
  map<int64_t, int64_t> int_map;
  start_time = get_usec();
  for(int64_t i = 0; i < n; i++) {
    int_map[i] = i;
  }
  printf("insert %ld nodes: time=%fs\n", n, (get_usec() - start_time)/1e6);
  return 0;
}

int main(int argc, char *argv[])
{
  int64_t n = 0;
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1 << (n > 0 ? n: 10));
  srand(time(NULL));
  test_rb_tree(n);
  test_map(n);
  return 0;
}
