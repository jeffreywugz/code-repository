#ifndef __OB_AX_CTX_MGR_H__
#define __OB_AX_CTX_MGR_H__
#include "fixed_array.h"

template<typename Item>
class CtxMgr
{
public:
  CtxMgr(){}
  ~CtxMgr(){}
public:
  int init(int64_t len){
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = free_list_.init(len)))
    {}
    else if (AX_SUCCESS != (err = array_.init(len)))
    {}
    return err;
  }

  int alloc(int64_t& id){
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (AX_SUCCESS != (err = free_list_.pop(item)))
    {}
    else
    {
      id = item->born();
    }
    return err;
  }
  int fetch(int64_t id, Item*& item){
    int err = AX_SUCCESS;
    if (NULL == (item = array_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = item->inc_ref(id)))
    {}
    return err;
  }
  int revert(Id id) {
    int err = AX_SUCCESS;
    if (NULL == (item = array_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = item->dec_ref(id)))
    {}
    return err;
  }
  int free(int64_t id) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = array_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = item->reclaim(id, id + array_.len())))
    {}
    else if (AX_SUCCESS != (err = free_list_.push(item)))
    {}
    return err;
  }
protected:
  FreeList free_list_;
  FixedArray<Item> array_;
};

#endif /* __AX_AX_ITEM_MGR_H__ */

