class LinkQueue
{
public:
  struct Item
  {
    Item(): next_(NULL){}
    ~Item() {}
    Item* next_;
  };
public:
  LinkQueue(): head_(NULL), tail_(NULL) {}
  ~LinkQueue(){}
  int init(Item* item) {
    int err = 0;
    head_ = tail_ = item;
    return err;
  }
  inline int push(Item* p) {
    int err = AX_SUCCESS;
    Item* tail = NULL;
    if (NULL == p)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else
    {
      p->next_ = NULL;
      tail = FAS(&tail_, p);
      tail->next_ = p;
    }
    return err;
  }
  inline int pop(Item*& p, Item*& prev) {
    int err = AX_SUCCESS;
    Item* head = NULL;
    while(NULL == (head = FAS(&head_, NULL)))
      ;
    if (NULL == head->next_)
    {
      head_ = head;
      err = AX_EAGAIN;
    }
    else
    {
      head_ = head->next_;
      prev = head;
      p = head->next_;
    }
    return err;
  }
private:
  Item* head_ CACHE_ALIGNED;
  Item* tail_ CACHE_ALIGNED;
};
