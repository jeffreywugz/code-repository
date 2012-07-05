#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <new>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, (new_us - old_us)/1000); \
      new_us - old_us; })

inline int64_t get_largest_power_of_2(int64_t n)
{
  int64_t power = 1;
  while(n){
    n >>= 1;
    power <<= 1;
  }
  return power>>2;
}
template<typename T>
class HashMap
{
  public:
    HashMap(int64_t capacity): used_(1), limit_(1), capacity_(capacity) {
      memset(slots_, 0, sizeof(T*) * capacity);
    }
    ~HashMap() {}
    static HashMap* create(int64_t capacity) {
      HashMap* map = (typeof(map))malloc(sizeof(*map) + sizeof(T*) * capacity);
      if (map){
        map = new(map) HashMap(capacity);
      }
      return map;
    }
    int64_t get_slot_idx(const int64_t hash) const {
      return (hash % (2 * limit_) < used_)?  hash % (2 * limit_): hash % limit_;
    }
    T** get_slot(const int64_t hash) const {
      return const_cast<T**>(slots_ + get_slot_idx(hash));
    }
    int insert(T* value) {
      int err = 0;
      T** slot = get_slot(value->hash());
      value->next_ = *slot;
      *slot = value;
      return err;
    }
    int expand() {
      int err = 0;
      T** adjust_slot = NULL;
      T* p = NULL;
      T* q = NULL;
      limit_ <<= (++used_ > 2 * limit_? 1: 0);
      adjust_slot = get_slot(used_ -1);
      p = *adjust_slot;
      *adjust_slot = NULL;
      for(; 0 == err && p; p = q){
        q = p->next_;
        p->next_ = NULL;
        if (0 != (err = insert(p)))
        {
          fprintf(stderr, "insert(p=%p)=>%d\n", p, err);
        }
      }
      return err;
    }
    int lookup(T* key, T*& value) const {
      int err = 0;
      for(value = *get_slot(key->hash()); value; value = value->next_) {
        if (key->equal(value))
          break;
      }
      return err;
    }
  private:
      int64_t used_;
      int64_t limit_;
      int64_t capacity_;
      T* slots_[0];
};

struct pair_t
{
  pair_t* next_;
  int64_t key_;
  int64_t value_;
  int64_t hash() const { return key_; }
  bool equal(pair_t* that) const { return this->key_ == that->key_; }
};

const int64_t MAX_N_SLOTS = 1<<30;
int test_hash(int64_t n)
{
  int err = 0;
  pair_t* pairs = (typeof(pairs))malloc(sizeof(pair_t) * n);
  HashMap<pair_t>* hash = HashMap<pair_t>::create(MAX_N_SLOTS);
  pair_t* value = NULL;
  int64_t start_ts = 0;
  int64_t end_ts = 0;
  if (NULL == pairs || NULL == hash)
  {
    err = -ENOMEM;
  }
  else
  {
    memset(pairs, 0, sizeof(pairs) * n);
  }
  for (int64_t i = 0; 0 == err && i < n; i++) {
    pairs[i].key_ = random();
    pairs[i].value_ = ~pairs[i].key_;
  }
  start_ts = get_usec();
  for(int64_t i = 0; 0 == err && i < n; i++) {
    if (0 != (err = hash->expand()))
    {
      fprintf(stderr, "hash->expand()=>%d\n");
    }
    else if (0 != (err = hash->insert(pairs + i)))
    {
      fprintf(stderr, "hash->insert(i=%ld)=>%d\n", i, err);
    }
  }
  end_ts = get_usec();
  printf("insert: %ldms\n", (end_ts - start_ts)/1000);
  start_ts = get_usec();
  for(int64_t i = 0; 0 == err && i < n; i++) {
    if (0 != (err = hash->lookup(pairs + i, value)))
    {
      fprintf(stderr, "hash->insert(i=%ld)=>%d\n", i, err);
    }
  }
  end_ts = get_usec();
  printf("lookup: %ldms\n", (end_ts - start_ts)/1000);
  if (NULL != pairs)
  {
    free(pairs);
  }
  if (NULL != hash)
  {
    free(hash);
  }
  return err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 2)
  {
    fprintf(stderr, "%s n_items", argv[0]);
  }
  else
  {
    profile(test_hash(atoll(argv[1])));
  }
  return 0;
}
