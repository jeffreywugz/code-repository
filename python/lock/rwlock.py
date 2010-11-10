#!/usr/bin/python
import time
import threading

def async_call(func):
    threading.Thread(target=func).start()
    
def par(func, n=1):
    [async_call(func) for i in range(n)]

class RWLock:
    def __init__(self):
        write_lock = threading.Lock()
        self.write_lock = RWLock.WriteLock(write_lock)
        self.read_lock = RWLock.ReadLock(write_lock)

    class WriteLock:
        def __init__(self, write_lock):
            self.write_lock = write_lock

        def acquire(self):
            self.write_lock.acquire()

        def release(self):
            self.write_lock.release()

        def __enter__(self):
            self.acquire()

        def __exit__(self, type, value, traceback):
            self.release()

    class ReadLock:
        def __init__(self, write_lock):
            self.write_lock = write_lock
            self.counter_lock = threading.Lock()
            self.counter = 0

        def acquire(self):
            with self.counter_lock:
                if self.counter == 0:
                    self.write_lock.acquire()
                self.counter += 1

        def release(self):
            with self.counter_lock:
                self.counter -= 1
                if self.counter == 0:
                    self.write_lock.release()

        def __enter__(self):
            self.acquire()

        def __exit__(self, type, value, traceback):
            self.release()

class State:
    def __init__(self):
        self.rwlock = RWLock()
        self.x, self.y = 0, 0

    def set(self, x, y):
        with self.rwlock.write_lock:
            self.x = x
            time.sleep(0.01)
            self.y = y

    def get(self):
        with self.rwlock.read_lock:
            time.sleep(0.01)
            return self.x, self.y

state = State()
def read():
    for i in range(100):
        print state.get()

def write():
    for i in range(100):
        state.set(i, i*i)
        
par(read, 2)
par(write, 2)
