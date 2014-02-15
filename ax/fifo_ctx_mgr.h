#ifndef __OB_AX_CTX_MGR_H__
#define __OB_AX_CTX_MGR_H__
#include "fixed_array.h"

template<typename DataT>
class ObCtxMgr
{
public:
  enum {READY = 1 };
  struct Item
  {
    Item(int64_t seq): seq_(seq), data_() {}
    ~Item(){}
    void wait(int64_t seq) {
      while (AL(&seq_) != seq)
      {
        PAUSE();
      }
    }
    int64_t seq_ CACHE_ALIGNED;
    DataT data_;
  };
public:
  ObCtxMgr(): push_(0) {}
  ~ObCtxMgr(){}
public:
  int init(int64_t len){
    return array_.init(len);
  }

  DataT* alloc(int64_t& seq){
    int err = AX_SUCCESS;
    Item* item = NULL;
    DataT* ret = NULL;
    int64_t push = FAA(&push_, 1);
    if (NULL == (item = array_.get(push)))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      item->wait(push);
      ret = &item->data_;
      seq = push;
    }
    return ret;
  }
  int add(int64_t seq){
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = array_.get(seq)))
    {
      err = AX_NOT_INIT;
    }
    else if (!CAS(&item->seq_, seq, seq + READY))
    {
      err = AX_STATE_NOT_MATCH;
    }
    return err;
  }
  DataT* get(int64_t seq) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    DataT* ret = NULL;
    if (NULL == (item = array_.get(seq)))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      item->wait(seq + READY);
      ret = &item->data_;
    }
    return ret;
  }
  int free(int64_t seq) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = array_.get(seq)))
    {
      err = AX_NOT_INIT;
    }
    else if (!CAS(&item->seq_, seq + READY, seq + array_.len()))
    {
      err = AX_STATE_NOT_MATCH;
    }
    return err;
  }
protected:
  int64_t push_ CACHE_ALIGNED;
  ObFixedArray<Item> array_;
};

#endif /* __AX_AX_CTX_MGR_H__ */

