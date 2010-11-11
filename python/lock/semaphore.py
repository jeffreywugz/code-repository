#!/usr/bin/python
import time
import threading

def async_call(func):
    threading.Thread(target=func).start()
    
def par(func, n=1):
    [async_call(func) for i in range(n)]

class Sem:
    def __init__(self, counter=0):
        self.counter, self.waiter = counter, 0
        self.cond = threading.Condition() # Condition will create an associated Lock automatically

    def get(self):
        with self.cond:
            if self.counter <= 0:
                self.waiter += 1
                while self.counter <= 0:
                    self.cond.wait()
                self.waiter -= 1
            self.counter -= 1

    def post(self):
        with self.cond:
            self.counter += 1
            if self.waiter >= 0:
                self.cond.notify()

sem = Sem(2)
def process():
    print "sem.counter = %d" % sem.counter
    sem.get()
    print "get"
    time.sleep(0.1)
    sem.post()
    print "post"
    
par(process, 8)
