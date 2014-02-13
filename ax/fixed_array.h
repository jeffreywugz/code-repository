#include "ax_common.h"

template<typename DataT>
class ObFixedArray
{
  typedef DataT Item;
  enum {MIN_ARRAY_LEN = 4};
public:
  ObFixedArray(): pos_mask_(0), items_(NULL) {}
  ~ObFixedArray() { destroy(); }
  int init(int64_t len)
  {
    int err = AX_SUCCESS;
    if (len < MIN_ARRAY_LEN || 0 != ((len-1) & len))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL != items_)
    {
      err = AX_INIT_TWICE;
    }
    else if (NULL == (items_ = (Item*)ax_malloc(sizeof(Item) * len, ObModIds::AX_SEQ_QUEUE)))
    {
      err = AX_NOMEM;
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < len; i++)
    {
      new(items_ + i)Item(i);
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    else
    {
      pos_mask_ = len - 1;
    }
    return err;
  }

  bool is_inited() const { return NULL != items_; }
  Item* get(int64_t seq){ return items_? items_ + (seq & pos_mask_): NULL; }
  int64_t len() const { return pos_mask_ + 1; }
  void destroy()
  {
    if (NULL != items_)
    {
      for(int64_t i = 0; i < pos_mask_ + 1; i++)
      {
        items_[i].~Item();
      }
      ob_free(items_);
      items_ = NULL;
    }
  }
protected:
  int64_t pos_mask_;
  Item* items_;
};
