#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <new>

enum {
  BTREE_SUCCESS = 0,
  BTREE_ERROR = 1,
  BTREE_DUPLICATE = 2,
  BTREE_NOENT = 3,
  BTREE_EAGAIN = 4,
  BTREE_NOMEM = 5,
  BTREE_DEPTH_OVERFLOW = 6,
  BTREE_NOT_INIT = 7,
};
#define AL(x) __atomic_load_n((x), __ATOMIC_SEQ_CST)
#define AS(x, v) __atomic_store_n((x), (v), __ATOMIC_SEQ_CST)
#define CAS(x, ov, nv) __sync_bool_compare_and_swap((x), (ov), (nv))
#define FAA(x, i) __sync_fetch_and_add((x), (i))
#define AAF(x, i) __sync_add_and_fetch((x), (i))
#define PAUSE() asm("pause;\n")
#define WEAK_SYM __attribute__((weak))
#define strbool(x) ((x)?"true":"false")

#define BTREE_LOG(prefix, format, ...) if (__enable_dlog__) {int64_t cur_ts = get_us(); fprintf(stderr, "[%ld.%.6ld] %s %s:%d [%ld] " format "\n", cur_ts/1000000, cur_ts%1000000, #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__); }
bool __enable_dlog__ WEAK_SYM = true;
#define BTREE_ASSERT_EQ(val, expr) { int real_val = expr; if (val != real_val) { printf("expected %d real %d %s\n", val, real_val, #expr); } };
inline int64_t get_us()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

class SpinLock
{
public:
  SpinLock(): lock_(0) {}
  ~SpinLock() {}
  bool try_lock() {
    return CAS(&lock_, 0, 1);
  }
  void unlock() {
    lock_ = 0;
  }
private:
  int64_t lock_;
};

class RWLock
{
public:
  RWLock(): writer_id_(0), read_ref_(0) {}
  ~RWLock() {}
  bool try_rdlock() {
    bool lock_succ = true;
    if (AL(&writer_id_) != 0)
    {
      lock_succ = false;
    }
    else
    {
      FAA(&read_ref_, 1);
      if (AL(&writer_id_) != 0)
      {
        FAA(&read_ref_, -1);
        lock_succ = false;
      }
    }
    return lock_succ;
  }
  bool try_wrlock() {
    bool lock_succ = true;
    if (!CAS(&writer_id_, 0, 1))
    {
      lock_succ = false;
    }
    else
    {
      while(AL(&read_ref_) > 0)
      {
        PAUSE();
      }
    }
    return lock_succ;
  }
  void rdunlock() { FAA(&read_ref_, -1); }
  void wrunlock() { AS(&writer_id_, 0); }
  bool try_rd2wrlock() {
    bool lock_succ = true;
    if (!CAS(&writer_id_, 0, 1))
    {
      lock_succ = false;
    }
    else
    {
      FAA(&read_ref_, -1);
      while(AL(&read_ref_) > 0)
      {
        PAUSE();
      }
    }
    return lock_succ;
  }
private:
  int32_t writer_id_;
  int32_t read_ref_;
};

#include "allocator.h"

class Btree
{
public:
  typedef int64_t key_t;
  typedef void* val_t;
  enum { MAX_DEPTH = 32, NODE_KEY_COUNT = 30};
  const static key_t MIN_KEY = -INT64_MAX;
  struct Node: public BaseNode
  {
  public:
    RWLock lock_;
    bool is_root_;
    bool is_leaf_;
    int32_t count_; 
    key_t keys_[NODE_KEY_COUNT];
    val_t vals_[NODE_KEY_COUNT];
  public:
    Node(): is_root_(false), is_leaf_(false), count_(0) {}
    ~Node() {}
    int try_rdlock() { return lock_.try_rdlock()? BTREE_SUCCESS: BTREE_EAGAIN; }
    int try_wrlock() { return lock_.try_wrlock()? BTREE_SUCCESS: BTREE_EAGAIN; }
    void rdunlock() { lock_.rdunlock(); }
    void wrunlock() { lock_.wrunlock();}
    int try_rd2wrlock() { return lock_.try_rd2wrlock()? BTREE_SUCCESS: BTREE_EAGAIN; }
    bool is_leaf() const { return is_leaf_; }
    void set_is_root(bool is_root) { is_root_ = is_root; }
    void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }
    key_t get_key(int64_t pos) const { return pos < count_? keys_[pos]: NULL; }
    val_t get_val(int64_t pos) const { return pos < count_? vals_[pos]: NULL; }
    void set_key(int64_t pos, key_t key) { if (pos < count_) keys_[pos] = key; }
    void set_val(int64_t pos, val_t val) { if (pos < count_) vals_[pos] = val; }
    void append_child(key_t key, val_t val) {
      if (count_ < NODE_KEY_COUNT)
      {
        keys_[count_] = key;
        vals_[count_] = val;
        count_++;
      }
    }
    int64_t get_count() const { return count_; }
    bool is_overflow(int64_t delta) {
      return count_ + delta > NODE_KEY_COUNT;
    }
    bool is_underflow(int64_t delta) {
      return !is_root_ && (count_ + delta < NODE_KEY_COUNT/2);
    }
    void print(FILE* file, int depth) {
      if (is_leaf_)
      {
        fprintf(file, "L%dK: ", count_);
      }
      else
      {
        fprintf(file, "%*s %dK:\n", depth * 4, "|-", count_);
      }
      for(int64_t i = 0; i < count_; i++) {
        if (!is_leaf_)
        {
          fprintf(file, "%*s %lx:%p ", depth * 4, "|-", keys_[i], vals_[i]);
          Node* child = (Node*)vals_[i];
          child->print(file, depth + 1);
        }
        else
        {
          fprintf(file, "%lx:%p ",keys_[i], vals_[i]);
        }
      }
      fprintf(file, "\n");
    }
    int64_t find_upper_bound(key_t key) { // 比key小的最大元素
      int64_t pos = -1;
      bool is_found = false;
      int64_t start_pos = 0;
      int64_t end_pos = count_;
      while(start_pos + 1 < end_pos) {
        pos = (start_pos + end_pos)/2;
        if (key >= keys_[pos])
        {
          start_pos = pos;
          is_found = true;
        }
        else
        {
          end_pos = pos;
        }
      }
      if (is_found)
      {
        pos = start_pos;
      }
      else if (start_pos + 1 == end_pos && key >= keys_[start_pos])
      {
        pos = start_pos;
      }
      else
      {
        pos = -1;
      }
      return pos;
    }
  protected:
    inline void copy(Node* dest, int64_t dest_start, int64_t start, int64_t end) {
      memcpy(dest->keys_ + dest_start, this->keys_ + start, sizeof(key_t) * (end - start));
      memcpy(dest->vals_ + dest_start, this->vals_ + start, sizeof(val_t) * (end - start));
    }
    inline void copy_and_split_child(Node* dest_node, int64_t dest_start, int64_t start, int64_t end, int64_t pos, val_t val_1, key_t key, val_t val_2) {
      copy(dest_node, dest_start, start, pos + 1);
      dest_node->vals_[dest_start + pos - start] = val_1;
      dest_node->keys_[dest_start + pos + 1 - start] = key;
      dest_node->vals_[dest_start + pos + 1 - start] = val_2;
      copy(dest_node, dest_start + (pos + 2) - start, pos + 1, end);
    }
    inline void copy_and_merge_child(Node* dest_node, int64_t dest_start, int64_t start, int64_t end, int64_t pos, key_t key, val_t val) {
      copy(dest_node, dest_start, start, pos);
      dest_node->keys_[dest_start + pos - start] = key;
      dest_node->vals_[dest_start + pos - start] = val;
      copy(dest_node, dest_start + pos + 1 - start, pos + 2, end);
    }
  public:
    int split_child_no_overflow(Node* new_node, int64_t pos, val_t val_1, key_t key, val_t val_2) {
      int err = BTREE_SUCCESS;
      new_node->is_leaf_ = is_leaf_;
      new_node->count_ = count_ + 1;
      copy_and_split_child(new_node, 0, 0, count_, pos, val_1, key, val_2);
      return err;
    }
    int split_child_cause_recursive_split(Node* new_node_1, Node* new_node_2, int64_t pos, val_t val_1, key_t key, val_t val_2) {
      int err = BTREE_SUCCESS;
      const int64_t half_limit = NODE_KEY_COUNT/2;
      new_node_1->is_leaf_ = is_leaf_;
      new_node_2->is_leaf_ = is_leaf_;

      if (pos < half_limit)
      {
        new_node_1->count_ = half_limit + 1;
        new_node_2->count_ = count_ - half_limit;
        copy_and_split_child(new_node_1, 0, 0, half_limit, pos, val_1, key, val_2);
        copy(new_node_2, 0, half_limit, count_);
      }
      else
      {
        new_node_1->count_ = half_limit;
        new_node_2->count_ = count_ + 1 - half_limit;
        copy(new_node_1, 0, 0, half_limit);
        copy_and_split_child(new_node_2, 0, half_limit, count_, pos, val_1, key, val_2);
      }
      return err;
    }
    int64_t get_merge_pos(int64_t pos) {
      return pos > 0? pos - 1: pos;
    }
    int64_t get_brother_pos(int64_t pos) {
      return pos > 0? pos - 1: pos + 1;
    }
    int replace_child_and_brother(Node* new_node, int64_t pos, Node* child, Node* brother)
    {
      int err = BTREE_SUCCESS;
      new_node->is_leaf_ = is_leaf_;
      new_node->count_ = count_;
      copy(new_node, 0, 0, count_);
      if (pos > 0)
      {
        new_node->keys_[pos] = child->get_key(0);
      }
      else
      {
        new_node->keys_[pos + 1] = brother->get_key(0);
      }
      new_node->vals_[pos] = child;
      new_node->vals_[get_brother_pos(pos)] = brother;
      return err;
    }
    int merge_child_no_underflow(Node* new_node, int64_t pos, key_t key, val_t val)
    {
      int err = BTREE_SUCCESS;
      new_node->is_leaf_ = is_leaf_;
      new_node->count_ = count_ - 1;
      copy_and_merge_child(new_node, 0, 0, count_, get_merge_pos(pos), key, val);
      return err;
    }
    int merge_child_cause_recursive_merge_with_right_brother(Node* new_node, Node* brother, int64_t pos, key_t key, val_t val)
    {
      int err = BTREE_SUCCESS;
      new_node->is_leaf_ = is_leaf_;
      new_node->count_ = count_ + brother->get_count() - 1;
      copy_and_merge_child(new_node, 0, 0, count_, get_merge_pos(pos), key, val);
      brother->copy(new_node, count_ - 1, 0, brother->count_);
      return err;
    }
    int merge_child_cause_recursive_merge_with_left_brother(Node* new_node, Node* brother, int64_t pos, key_t key, val_t val)
    {
      int err = BTREE_SUCCESS;
      new_node->is_leaf_ = is_leaf_;
      new_node->count_ = count_ + brother->get_count() - 1;
      brother->copy(new_node, 0, 0, brother->count_);
      copy_and_merge_child(new_node, brother->count_, 0, count_, get_merge_pos(pos), key, val);
      return err;
    }
    int merge_child_cause_rebalance_with_right_brother(Node* new_node_1, Node* new_node_2, Node* brother, int64_t pos, key_t key, val_t val)
    {
      int err = BTREE_SUCCESS;
      new_node_1->is_leaf_ = is_leaf_;
      new_node_1->count_ = (count_ + brother->count_ - 1)/2;
      new_node_2->is_leaf_ = is_leaf_;
      new_node_2->count_ = (count_ + brother->count_ - 1) - new_node_1->count_;
      copy_and_merge_child(new_node_1, 0, 0, count_, get_merge_pos(pos), key, val);
      brother->copy(new_node_1, count_ - 1, 0, new_node_1->count_ - count_ + 1);
      brother->copy(new_node_2, 0, new_node_1->count_ - count_ + 1, brother->count_);
      return err;
    }
    int merge_child_cause_rebalance_with_left_brother(Node* new_node_1, Node* new_node_2, Node* brother, int64_t pos, key_t key, val_t val)
    {
      int err = BTREE_SUCCESS;
      new_node_1->is_leaf_ = is_leaf_;
      new_node_1->count_ = (count_ + brother->count_ - 1)/2;
      new_node_2->is_leaf_ = is_leaf_;
      new_node_2->count_ = (count_ + brother->count_ - 1) - new_node_1->count_;
      brother->copy(new_node_1, 0, 0, new_node_1->count_);
      brother->copy(new_node_2, 0, new_node_1->count_, brother->count_);
      copy_and_merge_child(new_node_2, brother->count_ - new_node_1->count_, 0, count_, get_merge_pos(pos), key, val);
      return err;
    }
  };

  class Path
  {
  public:
    enum { CAPACITY = MAX_DEPTH };
    struct Item
    {
      Node* node_;
      int64_t pos_;
    };
    Path(): depth_(0) {
      root1_.set_is_root(true);
      root1_.set_is_leaf(false);
      root2_.set_is_root(true);
      root2_.set_is_leaf(false);
      root1_.append_child(MIN_KEY, NULL);
      root2_.append_child(MIN_KEY, NULL);
      push(&root1_, 0);
      push(&root2_, 0);
    }
    ~Path(){}
    bool is_extra_root(Node* node) const {
      return node == &root1_ || node == &root2_;
    }
    Node* get_new_root() const {
      return (Node*)root1_.get_val(0)?: (Node*)root2_.get_val(0);
    }
    int push(Node* node, int64_t pos) {
      int err = BTREE_SUCCESS;
      if (depth_ >= CAPACITY)
      {
        err = BTREE_DEPTH_OVERFLOW;
      }
      else
      {
        path_[depth_].node_ = node;
        path_[depth_].pos_ = pos;
        depth_++;
      }
      return err;
    }
    int pop(Node*& node, int64_t& pos) {
      int err = BTREE_SUCCESS;
      if (depth_ <= 0)
      {
        err = BTREE_DEPTH_OVERFLOW;
      }
      else
      {
        depth_--;
        node = path_[depth_].node_;
        pos = path_[depth_].pos_;
      }
      return err;
    }
    int top(Node*& node, int64_t& pos) {
      int err = BTREE_SUCCESS;
      if (depth_ <= 0)
      {
        err = BTREE_DEPTH_OVERFLOW;
      }
      else
      {
        node = path_[depth_-1].node_;
        pos = path_[depth_-1].pos_;
      }
      return err;
    }
    bool is_empty() const { return 0 == depth_; }
  private:
    Node root1_;
    Node root2_;
    int64_t depth_;
    Item path_[CAPACITY];
  };

  class ReadHandle
  {
  protected:
    RetireList& tree_retire_list_;
    int64_t* ref_;
  public:
    ReadHandle(Btree& tree): tree_retire_list_(tree.get_retire_list()), ref_(NULL) {}
    ~ReadHandle() {}
    void acquire_ref() {
      ref_ = tree_retire_list_.acquire_ref();
    }
    void release_ref() {
      tree_retire_list_.release_ref(ref_);
    }
    int get(Node* root, key_t key, val_t& val) {
      int err = BTREE_SUCCESS;
      Node* leaf = NULL;
      int64_t pos = 0;
      if (BTREE_SUCCESS != (err = find_leaf(root, key, leaf, pos)))
      {}
      else
      {
        key_t prev_key = leaf->get_key(pos);
        if (prev_key != key)
        {
          err = BTREE_NOENT;
        }
        else
        {
          val = leaf->get_val(pos);
        }
      }
      return err;
    }
  private:
    int find_leaf(Node* root, key_t key, Node*& leaf, int64_t& pos){
      int err = BTREE_SUCCESS;
      if (NULL == root)
      {
        err = BTREE_ERROR;
        BTREE_LOG(ERROR, "root=NULL");
      }
      while(BTREE_SUCCESS == err)
      {
        if ((pos = root->find_upper_bound(key)) < 0)
        {
          err = BTREE_ERROR;
          BTREE_LOG(ERROR, "could not find upper_bound: root=%p key=%lx", root, key);
        }
        else if (root->is_leaf())
        {
          leaf = root;
          break;
        }
        else
        {
          root = (Node*)root->get_val(pos);
        }
      }
      return err;
    }
  };

  class WriteHandle: public ReadHandle
  {
  private:
    BaseNodeAllocator& allocator_;
    Path path_;
    BaseNodeList retire_list_;
    BaseNodeList alloc_list_;
  public:
    WriteHandle(Btree& tree): ReadHandle(tree), allocator_(tree.get_allocator()) {}
    ~WriteHandle() {}
    Node* alloc_node() {
      Node* p = NULL;
      if (NULL == (p = (Node*)tree_retire_list_.alloc()))
      {
        p = (Node*)allocator_.alloc();
      }
      if (NULL != p)
      {
        new(p)Node();
        alloc_list_.push(p);
      }
      return p;
    }
    void free_node(Node* p) {
      allocator_.free(p);
    }
    void retire(int btree_err) {
      if (BTREE_SUCCESS != btree_err)
      {
        Node* p = NULL;
        while(NULL != (p = (Node*)retire_list_.pop()))
        {
          p->wrunlock();
        }
        while(NULL != (p = (Node*)alloc_list_.pop()))
        {
          free_node(p);
        }
      }
      else
      {
        tree_retire_list_.retire(retire_list_);
      }
    }
  public:
    int find_path(Node* root, key_t key){
      int err = BTREE_SUCCESS;
      int64_t pos = -1;
      if (NULL == root)
      {
        err = BTREE_ERROR;
        BTREE_LOG(ERROR, "root=NULL");
      }
      while(BTREE_SUCCESS == err)
      {
        if ((pos = root->find_upper_bound(key)) < 0)
        {
          err = BTREE_ERROR;
          BTREE_LOG(ERROR, "could not find upper_bound: root=%p key=%lx", root, key);
        }
        else if (BTREE_SUCCESS != (err = path_.push(root, pos)))
        {}
        else if (root->is_leaf())
        {
          break;
        }
        else
        {
          root = (Node*)root->get_val(pos);
        }
      }
      return err;
    }
    int insert_and_split_upward(key_t key, val_t val, bool overwrite, Node*& new_root)
    {
      int err = BTREE_SUCCESS;
      Node* old_node = NULL;
      int64_t pos = -1;
      Node* new_node_1 = NULL;
      Node* new_node_2 = NULL;
      if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
      {}
      else if (BTREE_SUCCESS != (err = insert_to_leaf(old_node, overwrite, pos, key, val, new_node_1, new_node_2)))
      {}
      while(BTREE_SUCCESS == err)
      {
        if (old_node == new_node_1)
        {
          break;
        }
        else if (NULL != new_node_2)
        {
          if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
          {}
          else if (BTREE_SUCCESS != (err = split_child(old_node, pos, new_node_1, new_node_2->get_key(0), new_node_2, new_node_1, new_node_2)))
          {}
        }
        else
        {
          if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
          {}
          else if (BTREE_SUCCESS != (err = replace_child(old_node, pos, new_node_1)))
          {}
          else
          {
            new_node_1 = old_node;
          }
        }
      }
      if (BTREE_SUCCESS == err)
      {
        new_root = path_.get_new_root();
      }
      return err;
    }

    int delete_and_merge_upward(key_t key, val_t& val, Node*& new_root)
    {
      int err = BTREE_SUCCESS;
      Node* old_node = NULL;
      int64_t pos = -1;
      Node* new_node_1 = NULL;
      Node* new_node_2 = NULL;
      if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
      {}
      else if (BTREE_SUCCESS != (err = delete_from_leaf(old_node, pos, key, val, new_node_1, new_node_2)))
      {}
      while(BTREE_SUCCESS == err)
      {
        if (old_node == new_node_1)
        {
          break;
        }
        else if (NULL == new_node_2)
        {
          if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
          {}
          else if (BTREE_SUCCESS != (err = replace_child(old_node, pos, new_node_1)))
          {}
          else
          {
            new_node_1 = old_node;
          }
        }
        else if (NULL == new_node_1)
        {
          if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
          {}
          else if (BTREE_SUCCESS != (err = merge_child(old_node, pos, new_node_2->get_key(0), new_node_2, new_node_1, new_node_2)))
          {}
        }
        else
        {
          if (BTREE_SUCCESS != (err = path_.pop(old_node, pos)))
          {}
          else if (BTREE_SUCCESS != (err = replace_child_and_brother(old_node, pos, new_node_1, new_node_2, new_node_1)))
          {}
          else
          {
            new_node_2 = NULL;
          }
        }
      }
      if (BTREE_SUCCESS == err)
      {
        if (NULL != (new_root = path_.get_new_root()))
        {
          if (new_root->get_count() == 1 && !new_root->is_leaf())
          {
            retire_list_.push(new_root);
            new_root = (Node*)new_root->get_val(0);
          }
        }
      }
      return err;
    }
  private:
    int try_wrlock(Node* node)
    {
      int err = BTREE_SUCCESS;
      if (0 != node->try_wrlock())
      {
        err = BTREE_EAGAIN;
      }
      else if (!path_.is_extra_root(node))
      {
        retire_list_.push(node);
      }
      return err;
    }
    int replace_child(Node* old_node, int64_t pos, val_t val) {
      int err = BTREE_SUCCESS;
      if (0 != old_node->try_rdlock())
      {
        err = BTREE_EAGAIN;
      }
      else
      {
        old_node->set_val(pos, val);
        old_node->rdunlock();
      }
      return err;
    }
    int replace_child_and_brother(Node* old_node, int64_t pos, Node* child, Node* brother, Node*& new_node) {
      int err = BTREE_SUCCESS;
      if (BTREE_SUCCESS != (err = try_wrlock(old_node)))
      {}
      else if (NULL == (new_node = alloc_node()))
      {
        err = BTREE_NOMEM;
      }
      else
      {
        err = old_node->replace_child_and_brother(new_node, pos, child, brother);
      }
      return err;
    }
    int insert_to_leaf(Node* old_node, bool overwrite, int64_t pos, key_t key, val_t val, Node*& new_node_1, Node*& new_node_2) {
      int err = BTREE_SUCCESS;
      key_t prev_key = old_node->get_key(pos);
      if (prev_key != key)
      {
        err = split_child(old_node, pos, old_node->get_val(pos), key, val, new_node_1, new_node_2);
      }
      else if (overwrite)
      {
        new_node_1 = old_node;
        new_node_2 = NULL;
        err = replace_child(old_node, pos, val);
      }
      else
      {
        err = BTREE_DUPLICATE;
      }
      return err;
    }
    int split_child(Node* old_node, int64_t pos, val_t val_1, key_t new_key, val_t val_2, Node*& new_node_1, Node*& new_node_2) {
      int err = BTREE_SUCCESS;
      if (BTREE_SUCCESS != (err = try_wrlock(old_node)))
      {}
      else if (old_node->is_overflow(1))
      {
        if (NULL == (new_node_1 = alloc_node()) || NULL == (new_node_2 = alloc_node()))
        {
          err = BTREE_NOMEM;
        }
        else
        {
          err = old_node->split_child_cause_recursive_split(new_node_1, new_node_2, pos, val_1, new_key, val_2);
        }
      }
      else
      {
        new_node_2 = NULL;
        if (NULL == (new_node_1 = alloc_node()))
        {
          err = BTREE_NOMEM;
        }
        else
        {
          err = old_node->split_child_no_overflow(new_node_1, pos, val_1, new_key, val_2);
        }
      }
      return err;
    }
    
    int delete_from_leaf(Node* old_node, int64_t pos, key_t key, val_t& val, Node*& new_node_1, Node*& new_node_2)
    {
      int err = BTREE_SUCCESS;
      key_t prev_key = old_node->get_key(pos);
      if (prev_key != key)
      {
        err = BTREE_NOENT;
      }
      else
      {
        val = old_node->get_val(pos);
        int64_t brother_pos = old_node->get_brother_pos(pos);
        err = merge_child(old_node, pos, old_node->get_key(brother_pos), old_node->get_val(brother_pos), new_node_1, new_node_2);
      }
      return err;
    }
    int merge_child(Node* old_node, int64_t pos, key_t key, val_t val, Node*& new_node_1, Node*& new_node_2)
    {
      int err = BTREE_SUCCESS;
      Node* parent_node = NULL;
      Node* brother_node = NULL;
      int64_t parent_pos = -1;
      if (BTREE_SUCCESS != (err = try_wrlock(old_node)))
      {}
      else if (!old_node->is_underflow(-1))
      {
        new_node_2 = NULL;
        if (NULL == (new_node_1 = alloc_node()))
        {
          err = BTREE_NOMEM;
        }
        else
        {
          err = old_node->merge_child_no_underflow(new_node_1, pos, key, val);
        }
      }
      else if (BTREE_SUCCESS != (err = path_.top(parent_node, parent_pos)))
      {}
      else if (NULL == (brother_node = (Node*)parent_node->get_val(parent_node->get_brother_pos(parent_pos))))
      {
        err = BTREE_ERROR;
      }
      else if (BTREE_SUCCESS != (err = try_wrlock(brother_node)))
      {}
      else if (!brother_node->is_overflow(old_node->get_count()))
      {
        new_node_1 = NULL;
        if (NULL == (new_node_2 = alloc_node()))
        {
          err = BTREE_NOMEM;
        }
        else if (parent_pos > 0)
        {
          err = old_node->merge_child_cause_recursive_merge_with_left_brother(new_node_2, brother_node, pos, key, val);
        }
        else
        {
          err = old_node->merge_child_cause_recursive_merge_with_right_brother(new_node_2, brother_node, pos, key, val);
        }
      }
      else
      {
        if (NULL == (new_node_1 = alloc_node()) || NULL == (new_node_2 = alloc_node()))
        {
          err = BTREE_NOMEM;
        }
        else if (parent_pos > 0)
        {
          err = old_node->merge_child_cause_rebalance_with_left_brother(new_node_2, new_node_1, brother_node, pos, key, val);
        }
        else
        {
          err = old_node->merge_child_cause_rebalance_with_right_brother(new_node_1, new_node_2, brother_node, pos, key, val);
        }
      }
      return err;
    }
  };

  Btree(): allocator_(sizeof(Node)), root_(NULL)
  {
    if (NULL != (root_ = (Node*)allocator_.alloc()))
    {
      root_->set_is_root(true);
      root_->set_is_leaf(true);
      root_->append_child(MIN_KEY, NULL);
    }
  }
  ~Btree(){}
  void print(FILE* file)
  {
    fprintf(file, "--------------------------\n");
    fprintf(file, "|root=%p\n", root_);
    root_->print(file, 0);
  }

  RetireList& get_retire_list() { return retire_list_; }
  BaseNodeAllocator& get_allocator() { return allocator_; }

  int set(key_t key, val_t val, bool overwrite = true)
  {
    int err = BTREE_SUCCESS;
    Node* root = NULL;
    Node* new_root = NULL;
    WriteHandle handle(*this);
    handle.acquire_ref();
    if (NULL == (root = AL(&root_)))
    {
      err = BTREE_NOT_INIT;
    }
    else if (BTREE_SUCCESS != (err = handle.find_path(root, key)))
    {
      BTREE_LOG(ERROR, "path.search(%p, %lx)=>%d", root, key, err);
    }
    else if (BTREE_SUCCESS != (err = handle.insert_and_split_upward(key, val, overwrite, new_root)))
    {
      if (BTREE_DUPLICATE != err && BTREE_EAGAIN != err)
      {
        BTREE_LOG(ERROR, "insert_upward(%p, %lx)=>%d", val, key, err);
      }
    }
    else if (NULL != new_root)
    {
      new_root->set_is_root(true);
      if (!CAS(&root_, root, new_root))
      {
        err = BTREE_EAGAIN;
      }
    }
    handle.release_ref();
    handle.retire(err);
    return err;
  }

  int del(key_t key, val_t& val)
  {
    int err = BTREE_SUCCESS;
    Node* root = NULL;
    Node* new_root = NULL;
    WriteHandle handle(*this);
    handle.acquire_ref();
    if (NULL == (root = AL(&root_)))
    {
      err = BTREE_NOT_INIT;
    }
    else if (BTREE_SUCCESS != (err = handle.find_path(root, key)))
    {
      BTREE_LOG(ERROR, "path.search(%p, %lx)=>%d", root, key, err);
    }
    else if (BTREE_SUCCESS != (err = handle.delete_and_merge_upward(key, val, new_root)))
    {
      if (BTREE_NOENT != err && BTREE_EAGAIN != err)
      {
        BTREE_LOG(ERROR, "delete_upward(%p, %lx)=>%d", val, key, err);
      }
    }
    else if (NULL != new_root)
    {
      new_root->set_is_root(true);
      if (!CAS(&root_, root, new_root))
      {
        err = BTREE_EAGAIN;
      }
    }
    handle.release_ref();
    handle.retire(err);
    return err;
  }

  int get(key_t key, val_t& val)
  {
    int err = BTREE_SUCCESS;
    Node* root = NULL;
    ReadHandle handle(*this);
    handle.acquire_ref();
    if (NULL == (root = AL(&root_)))
    {
      err = BTREE_NOT_INIT;
    }
    else if (BTREE_SUCCESS != (err = handle.get(root, key, val)))
    {
      if (BTREE_NOENT != err)
      {
        BTREE_LOG(ERROR, "handle.get(%p, %lx)=>%d", root, key, err);
      }
    }
    handle.release_ref();
    return err;
  }
private:
  RetireList retire_list_;
  BaseNodeAllocator allocator_;
  Node* root_;
};
