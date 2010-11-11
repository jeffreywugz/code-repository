#!/usr/bin/python
import time
import threading
import Queue

def async_call(func):
    threading.Thread(target=func).start()
    
def par(func, n=1):
    [async_call(func) for i in range(n)]

class AQueue:
    def __init__(self, limit = 1024):
        self.limit = limit
        self.queue = []
        self.resource_sem = threading.Semaphore(0)
        self.space_sem = threading.Semaphore(limit)

    def get(self):
        self.resource_sem.acquire()
        x = self.queue.pop()
        self.space_sem.release()
        return x
    
    def put(self, x):
        self.space_sem.acquire()
        self.queue.insert(0, x)
        self.resource_sem.release()
        
q = AQueue(4)
def producer():
    for i in range(10):
        time.sleep(0.1)
        q.put(i)

def consumer():
    for i in range(10):
        print "q.get() => %d" % q.get()
    
par(producer, 8)
par(consumer, 8)
