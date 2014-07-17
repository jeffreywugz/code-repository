#ifndef __OB_BTREE_ALLOCATOR_H__
#define __OB_BTREE_ALLOCATOR_H__

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <pthread.h>
#include "utility.h"

class HazardRef
{
public:
  enum { MAX_REF = 1024 };
  const static int64_t INVALID_VERSION = INT64_MAX;
public:
  HazardRef(): cur_ver_(1), used_ref_(0) {
    pthread_key_create(&key_, NULL);
  }
  ~HazardRef() {
    pthread_key_delete(key_);
  }
  int64_t* acquire_ref() {
    int64_t* ref = (int64_t*)pthread_getspecific(key_);
    if (NULL == ref)
    {
      int64_t new_ref_idx = FAA(&used_ref_, 1);
      if (new_ref_idx < MAX_REF)
      {
        ref = ref_array_ + new_ref_idx;
      }
      pthread_setspecific(key_, ref);
    }
    if (NULL != ref)
    {
      // while(true)
      // {
      //   int64_t cur_ver = AL(&cur_ver_);
      //   AS(ref, cur_ver);
      //   if (cur_ver >= AL(&cur_ver_))
      //   {
      //     break;
      //   }
      //   else
      //   {
      //     PAUSE();
      //   }
      // }
      int64_t cur_ver = AL(&cur_ver_);
      AS(ref, cur_ver);
    }
    return ref;
  }
  void release_ref(int64_t* ref) {
    *ref = INVALID_VERSION;
  }
  int64_t new_version() {
    return AAF(&cur_ver_, 1);
  }
  int64_t get_hazard_version() {
    int64_t min_version = AL(&cur_ver_);
    for (int64_t i = 0; i < used_ref_; i++)
    {
      int64_t ver = AL(ref_array_ + i);
      if (ver < min_version)
      {
        min_version = ver;
      }
    }
    return min_version;
  }
private:
  pthread_key_t key_;
  int64_t cur_ver_;
  int64_t used_ref_;
  int64_t ref_array_[MAX_REF];
};

struct BaseNode
{
  BaseNode(): next_(NULL), version_(0) {}
  ~BaseNode() {}
  BaseNode* next_;
  int64_t version_;
};

class BaseNodeList
{
public:
  BaseNodeList(): count_(0), tail_(&head_) {
    head_.next_ = &head_;
  }
  ~BaseNodeList() {}
  int64_t get_count() const { return count_; }
  void push(BaseNode* node) {
    count_++;
    node->next_ = tail_->next_;
    tail_->next_ = node;
    tail_ = node;
  }
  BaseNode* pop() {
    BaseNode* p = head_.next_;
    if (&head_ == p)
    {
      p = NULL;
    }
    else
    {
      head_.next_ = p->next_;
      if (p == tail_)
      {
        tail_ = p->next_;
      }
      count_--;
    }
    return p;
  }
  void clear() {
    count_ = 0;
    tail_ = &head_;
    head_.next_ = tail_;
  }
  void concat(BaseNodeList& list) {
    if (list.get_count() > 0)
    {
      count_ += list.get_count();
      list.tail_->next_ = tail_->next_;
      tail_->next_ = list.head_.next_;
      tail_ = list.tail_;
      list.clear();
    }
  }
  BaseNode* head() {
    BaseNode* p = head_.next_;
    return &head_ == p? NULL: p;
  }
  void set_version(int64_t version) {
    if (get_count() > 0)
    {
      head_.next_->version_ = version;
    }
  }
private:
  int64_t count_;
  BaseNode head_;
  BaseNode* tail_;
};

class RetireList
{
public:
  enum { PREPARE_THRESHOLD = 64, RETIRE_THRESHOLD = 1024 };
  struct ThreadRetireList
  {
    ThreadRetireList() {}
    ~ThreadRetireList() {}
    BaseNodeList retire_list_;
    BaseNodeList prepare_list_;
  };
public:
  RetireList(): hazard_version_(0) {
    pthread_key_create(&key_, NULL);
  }
  ~RetireList() {
    pthread_key_delete(key_);
  }
  BaseNode* alloc() {
    BaseNode* p = NULL;
    ThreadRetireList* retire_list = NULL;
    if (NULL == (retire_list = get_thread_retire_list()))
    {}
    else if (retire_list->retire_list_.get_count() <= RETIRE_THRESHOLD)
    {}
    else if (NULL == (p = retire_list->retire_list_.head()))
    {}
    else if (hazard_version_ <= p->version_
             && (hazard_version_ = hazard_ref_.get_hazard_version()) <= p->version_)
    {
      p = NULL;
    }
    else
    {
      retire_list->retire_list_.pop();
    }
    return p;
  }
  void retire(BaseNodeList& list) {
    ThreadRetireList* retire_list = NULL;
    if (NULL == (retire_list = get_thread_retire_list()))
    {}
    else
    {
      retire_list->prepare_list_.concat(list);
      if (retire_list->prepare_list_.get_count() > PREPARE_THRESHOLD)
      {
        retire_list->prepare_list_.set_version(hazard_ref_.new_version());
        retire_list->retire_list_.concat(retire_list->prepare_list_);
      }
    }
  }
  int64_t* acquire_ref() {
    return hazard_ref_.acquire_ref();
  }
  void release_ref(int64_t* ref) {
    return hazard_ref_.release_ref(ref);
  }
private:
  ThreadRetireList* get_thread_retire_list() {
    ThreadRetireList* list = NULL;
    if (NULL == (list = (ThreadRetireList*)pthread_getspecific(key_)))
    {
      list = alloc_thread_retire_list();
      pthread_setspecific(key_, (void*)list);
    }
    return list;
  }
  ThreadRetireList* alloc_thread_retire_list() {
    return new ThreadRetireList();
  }
private:
  pthread_key_t key_;
  int64_t hazard_version_;
  HazardRef hazard_ref_;
};

class BaseNodeAllocator
{
public:
  enum { BLOCK_SIZE = (1<<22) };
  typedef BaseNode Node;
  struct Block
  {
    Block(int64_t block_size): used_(0), len_(block_size - sizeof(*this)) {}
    ~Block(){}
    void* alloc(int64_t size) {
      void* p = NULL;
      if (used_ + size <= len_)
      {
        p = buf_ + used_;
        used_ += size;
      }
      return p;
    }
    int64_t used_;
    int64_t len_;
    char buf_[];
  };
  struct ThreadAlloc
  {
    ThreadAlloc(int64_t node_size): node_size_(node_size), cur_block_(NULL), free_list_() {}
    ~ThreadAlloc() {}
    Node* alloc() {
      Node* p = free_list_.pop();
      if (NULL == p)
      {
        if (NULL == cur_block_ || NULL == (p = (Node*)cur_block_->alloc(node_size_)))
        {
          cur_block_ = alloc_block(BLOCK_SIZE);
          p = (Node*)cur_block_->alloc(node_size_);
        }
        if (NULL != p)
        {
          new(p)Node();
        }
      }
      return p;
    }
    void free(Node* p) {
      if (NULL != p)
      {
        free_list_.push(p);
      }
    }
    Block* alloc_block(int64_t size) {
      Block* p = (Block*)malloc(size);
      return new(p)Block(size);
    }
    int64_t node_size_;
    Block* cur_block_;
    BaseNodeList free_list_;
  };
  BaseNodeAllocator(int64_t size): node_size_(size) {
    pthread_key_create(&key_, NULL);
  }
  ~BaseNodeAllocator() {
    pthread_key_delete(key_);
  }
  Node* alloc() {
    Node* p = NULL;
    ThreadAlloc* alloc = NULL;
    if (NULL != (alloc = get_thread_alloc()))
    {
      p = alloc->alloc();
    }
    return p;
  }
  void free(Node* p) {
    ThreadAlloc* alloc = NULL;
    if (NULL != (alloc = get_thread_alloc()))
    {
      alloc->free(p);
    }
  }
private:
  ThreadAlloc* get_thread_alloc() {
    ThreadAlloc* alloc = NULL;
    if (NULL == (alloc = (ThreadAlloc*)pthread_getspecific(key_)))
    {
      alloc = new_thread_alloc();
      pthread_setspecific(key_, alloc);
    }
    return alloc;
  }
  ThreadAlloc* new_thread_alloc() {
    ThreadAlloc* alloc = (ThreadAlloc*)malloc(sizeof(*alloc));
    if (NULL != alloc)
    {
      new(alloc)ThreadAlloc(node_size_);
    }
    return alloc;
  }
private:
  pthread_key_t key_;
  int64_t node_size_;
};

#endif /* __OB_BTREE_ALLOCATOR_H__ */
