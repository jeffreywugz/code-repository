#!/usr/bin/python

import os, exceptions
import subprocess
import fcntl
import rpc

def makeItNonBlocking(f):
    fd = f.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

class RLog:
    def __init__(self):
        self.pipes = []
    
    def shell(self, cmd):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        makeItNonBlocking(p.stdout)
        makeItNonBlocking(p.stderr)
        self.pipes += [p.stdout, p.stderr]
        return p

    @rpc.methodWrapper
    def get(self):
        def safeRead(f):
            try:
                content = f.read()
            except exceptions.IOError:
                content = ""
            if content == None: content = ""
            return content
        self.pipes = [p for p in self.pipes if not p.closed]
        return ''.join([safeRead(f) for f in self.pipes])
