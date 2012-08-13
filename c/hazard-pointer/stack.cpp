struct Node
{
  Node* prev_;
  void* data_;
};

class ThreadId
{
  int64_t get_id();
};

template<int64_t max_n_thread, int64_t n_pointer>
class HazardSet
{
  public:
    HazardSet(): pointers_(NULL) {}
    ~HazardSet()  {}
  public:
    int init(int64_t n_thread, int64_t n_pointer_per_thread, void* pointers);
    int64_t get_id();
    int add(void* p);
    int del(void* p);
    int is_exist(void* p);
  private:
    int64_t n_thread_;
    int64_t n_pointer_;
    void* pointers_;
};

class Allocator
{
  public:
    int alloc(Node*& node);
    int free(Node*& node);
    int add_hazard(Node* node);
    int del_hazard(Node* node);
  private:
    HazardSet hazard_set_;
};

class Stack {
  public:
    Stack(): top_(NULL) {}
    ~Stack() {}
    int push(Node* node) {
      int err = 0;
      node->prev_ = top_;
      if (!__sync_bool_compare_and_swap(&top_, node->prev_, node))
      {
        err = -EAGAIN;
      }
      return err;
    }
    int pop(Allocator* allocator, Node*& node) { 
      int err = 0;
      int tmp_err = 0;
      if (NULL == allocator)
      {
        err = -EINVAL;
      }
      else if (NULL == (node = top_))
      {
        err = -EAGAIN;
      }
      else if (0 != (err = allocator->add_hazard(node)))
      {}
      else
      {
        if (top_ != node || !__sync_bool_compare_and_swap(&top_, node, node->prev_))
        {
          err = -EAGAIN;
        }
        if (0 != (tmp_err = allocator->del_hazard(node)))
        {
          err = tmp_err;
        }
      }
      return err;
    }
  private:
    Node* top_;
};

