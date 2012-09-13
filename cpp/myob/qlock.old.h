/**
 * (C) 2007-2010 Taobao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * Authors:
 *   yuanqi <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 */
#ifndef __OB_COMMON_QLOCK_H__
#define __OB_COMMON_QLOCK_H__
namespace oceanbase
{
  namespace common
  {
    struct QLock
    {
      enum State {
        IDLE = 0,
        SHARED = 1,
        SHARED_WITH_EXCLUSIVE_INTENT = 2,
        EXCLUSIVE = 3,
      };
      enum {
        STATE_MASK = (1ULL<<2) - 1,
        STATE_SHIFT = 0,
        SHARED_REF_MASK = (1ULL<<16) - 1,
        SHARED_REF_SHIFT = 2,
        UID_MASK = ~0,
        UID_SHIFT = 16,
      };
#define get_bit_field(x, FIELD) (x & FIELD##_MASK) >> FIELD##_SHIFT
#define set_bit_field(x, FIELD, v) (x & ~(FIELD##_MASK>>FIELD##_SHIFT<<FIELD##_SHIFT) | (v << FIELD##_SHIFT))
#define CAS(x, old_v, new_v) __sync_bool_compare_and_swap(x, old_v, new_v)
      QLock(): lock_(0) {}
      ~QLock() {}
      volatile union {
        uint64_t lock_;
        struct {
          uint64_t state_:2;
          uint64_t n_shared_ref_:14;
          uint64_t uid_:48;
        };
      };
      static inline uint64_t get_state(uint64_t lock)
      {
        return get_bit_field(lock, STATE);
      }
      static inline uint64_t set_state(uint64_t lock, uint64_t state)
      {
        return set_bit_field(lock, STATE, state);
      }
      static inline uint64_t get_shared_ref(uint64_t lock)
      {
        return get_bit_field(lock, SHARED_REF);
      }
      static inline uint64_t set_shared_ref(uint64_t lock, uint64_t shared_ref)
      {
        return set_bit_field(lock, SHARED_REF, shared_ref);
      }
      static inline uint64_t add_shared_ref(uint64_t lock)
      {
        return set_shared_ref(lock, get_shared_ref(lock) + 1);
      }
      static inline uint64_t del_shared_ref(uint64_t lock)
      {
        return set_shared_ref(lock, get_shared_ref(lock) - 1);
      }
      static inline uint64_t get_uid(uint64_t lock)
      {
        return get_bit_field(lock, UID);
      }
      static inline uint64_t set_uid(uint64_t lock, uint64_t uid)
      {
        return set_bit_field(lock, UID, uid);
      }

      int try_shared_lock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        UNUSED(uid);
        if (EXCLUSIVE == get_state(lock)
            || !CAS(&lock_, lock, add_shared_ref(set_state(lock, get_state(lock) == IDLE? (uint64_t)SHARED: get_state(lock)))))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_shared_with_exclusive_intent_lock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if ((IDLE != get_state(lock) && SHARED != get_state(lock)))
        {
          err = OB_EAGAIN;
        }
        else if (!CAS(&lock_, lock, add_shared_ref(
                        set_uid(set_state(lock, SHARED_WITH_EXCLUSIVE_INTENT), uid))))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_share2exclusive_lock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if (SHARED_WITH_EXCLUSIVE_INTENT != get_state(lock)
            ||  get_shared_ref(lock) > 1)
        {
          err = OB_EAGAIN;
        }
        else if (set_uid(lock, uid) != lock)
        {
          err = OB_LOCK_NOT_MATCH;
          TBSYS_LOG(ERROR, "try_share2exclusive_lock(lock=%lu, uid=%lu): uid not match", lock, uid);
        }
        else if (get_shared_ref(lock) <= 0)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "try_share2exclusive_lock(lock=%lu, uid=%lu): ref count error", lock, uid);
        }
        else if (!CAS(&lock_, lock, del_shared_ref(set_state(lock, EXCLUSIVE))))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_exclusive_lock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if (IDLE != get_state(lock))
        {
          err = OB_EAGAIN;
        }
        else if (get_shared_ref(lock) > 0)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "try_exclusive_lock(lock=%lu, uid=%lu): shared ref != 0", lock, uid);
        }
        else if (!CAS(&lock_, lock, set_uid(set_state(lock, EXCLUSIVE), uid)))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_shared_unlock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if (SHARED != get_state(lock) && SHARED_WITH_EXCLUSIVE_INTENT != get_state(lock))
        {
          err = OB_LOCK_NOT_MATCH;
          TBSYS_LOG(ERROR, "try_shared_unlock(lock=%lu, uid=%lu): state not match", lock_, uid);
        }
        else if (get_shared_ref(lock) <= 0)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "try_shared_unlock(lock=%lu, uid=%lu): ref count wrong", lock_, uid);
        }
        else if (!CAS(&lock_, lock, del_shared_ref(
                        set_state(lock, get_shared_ref(lock) == 1? (uint64_t)IDLE: get_state(lock)))))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_exclusive_unlock(uint64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if (EXCLUSIVE != get_state(lock) || set_uid(lock, uid) != lock)
        {
          err = OB_LOCK_NOT_MATCH;
          TBSYS_LOG(ERROR, "try_exclusive_unlock(lock=%lu, uid=%lu): state uid not match", lock_, uid);
        }
        else if (get_shared_ref(lock) != 0)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "try_exclusive_unlock(lock=%lu, uid=%lu): ref count wrong", lock_, uid);
        }
        else if (!CAS(&lock_, lock, set_uid(set_state(lock, IDLE), 0ULL)))
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int try_shared_with_exclusive_intent_unlock(int64_t uid)
      {
        int err = OB_SUCCESS;
        uint64_t lock = lock_;
        if (SHARED_WITH_EXCLUSIVE_INTENT != get_state(lock) || set_uid(lock, uid) != lock)
        {
          err = OB_LOCK_NOT_MATCH;
          TBSYS_LOG(ERROR, "try_shex_intent_unlock(lock=%lu, uid=%lu): state or uid not match", lock_, uid);
        }
        else if (get_shared_ref(lock) <= 0)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "try_shex_intent_unlock(lock=%lu, uid=%lu): ref count wrong", lock_, uid);
        }
        else if (!CAS(&lock_, lock, del_shared_ref(
                        set_uid(set_state(lock, get_shared_ref(lock) == 1? IDLE: SHARED), 0ULL))))
        {
          err = OB_EAGAIN;
        }
        return err;
      }
    };
  }; // end namespace common
}; // end namespace oceanbase

#endif /* __OB_COMMON_QLOCK_H__ */
