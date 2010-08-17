#!/usr/bin/python

import threading
import time

counter = 0
lock = threading.Lock()
class Counter:
    def __init__(self, initial=0):
        self.count = initial
        self.lock = threading.Lock()

    def inc(self):
        self.lock.acquire()
        self.count += 1
        self.lock.release()

    def __str__(self):
        return 'counter:%d'% self.count

counter = Counter()
def inc():
    global counter
    while True:
        for i in range(10000):
            pass
        counter.inc()

threads = [threading.Thread(target=inc) for i in range(8)]
for i in threads:
    i.setDaemon(True)
print 'setDaemon Done'
[i.start() for i in threads]
print 'start Done'
time.sleep(1)
# [i.join() for i in threads]
print counter
