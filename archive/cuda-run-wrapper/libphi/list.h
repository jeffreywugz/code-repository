#ifndef _LIST_H_
#define _LIST_H_
/**
 * @file   list.h
 * @author unknown kernel developer
 * @date   Mon Jul 20 10:08:45 2009
 * 
 * @brief  define list operations copy from linux kernel source.
 * 
 * @ingroup libphi
 */

struct list_head {
	struct list_head *next, *prev;
};

#define LIST_HEAD_INIT(name) { &(name), &(name) }

#define LIST_HEAD(name) \
	struct list_head name = LIST_HEAD_INIT(name)

#define INIT_LIST_HEAD(ptr) do { \
	(ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)

static inline void __list_add(struct list_head *newnode,
			      struct list_head *prev,
			      struct list_head *next)
{
	next->prev = newnode;
	newnode->next = next;
	newnode->prev = prev;
	prev->next = newnode;
}

static inline void list_add(struct list_head *newnode, struct list_head *head)
{
	__list_add(newnode, head, head->next);
}

static inline void list_add_tail(struct list_head *newnode, struct list_head *head)
{
	__list_add(newnode, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next)
{
	next->prev = prev;
	prev->next = next;
}

static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	entry->next = (struct list_head *) 0;
	entry->prev = (struct list_head *) 0;
}

static inline int list_empty(struct list_head *head)
{
	return head->next == head;
}

#define list_entry(ptr, type, member) \
	((type *)((char *)(ptr)-(unsigned long)(&((type *)0)->member)))

#define list_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); \
        	pos = pos->next)
#define list_for_each_prev(pos, head) \
	for (pos = (head)->prev; pos != (head); \
        	pos = pos->prev)
        	
#define list_for_each_entry(pos, head, member)				\
	for (pos = list_entry((head)->next, typeof(*pos), member);	\
	     &pos->member != (head);                                    \
	     pos = list_entry(pos->member.next, typeof(*pos), member))

#endif    

