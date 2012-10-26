struct BlockRef
{
  int64_t version_; // also used as mutex lock
  BlockRef* next_;
};

struct Node
{
  int64_t version_; // also used as mutex lock
  int64_t hkey_;
  Key key_;
  Value value_;
  BlockRef* block_;
};

template<typename Key, typename Value>
class KVCache
{
  public:
    int init(int64_t n_slot, int64_t n_block, int64_t block_size, char* block_buf, Node* slot_buf);
    int get_node(const Key& key, Node*& node) {
      int err = 0;
      int64_t hkey = hash(key);
      for(idx = hkey; 0 == err && idx < hkey + n_slot_; idx++){
        node = slots_ + (idx%n_slot_);
        if ((node->hkey_ % n_slot_) != (hkey % n_slot_))
        {
          err = -ENOENT;
        }
        else if (node->key_ == key)
        {
          break;
        }
      }
      if (idx == hkey + n_slot_)
      {
        err = -ENOENT;
      }
      return err;
    }
    int put_node(Node*& node) {
    }
    int alloc_node(Node*& node) {
    }
    int revert_node(Node* node);
  private:
    int64_t n_slot_;
    int64_t n_block_;
    int64_t block_size_;
    Node* slots_;
    char* blocks_;
};

