#!/usr/bin/python
import sys, os
from common import *

class Store:
    def __init__(self, path, default):
        self.path, self.default = path, default
        self.file = posixfile.open(self.path, 'w')

    def __del__(self):
        self.file.close()
        
    def lock(self):
        self.file.lock('w|')

    def unlock(self):
        self.file.lock('u')

    def get(self):
        try:
            f = open(self.path)
            content = f.read()
            f.close()
            return eval(content)
        except exceptions.Exception:
            return self.default
        
    def set(self, value):
        f = open(self.path, 'w')
        f.write(repr(value))
        f.close()

class DictStore(Store):
    def __init__(self, path):
        Store.__init__(self, path, {})

    def __setitem__(self, k, v):
        self.lock()
        d = self.get()
        d[k] = v
        self.set(d)
        self.unlock()

    def __getitem__(self, k):
        return self.getAll()[k]

    def getAll(self):
        self.lock()
        d = self.get()
        self.unlock()
        return d

class Agent(CmdObjProxy):
    def __init__(self, ent_cmd, url, guestUser, guestPasswd, session='', user='', passwd='', log_dir='log'):
        self.ent_cmd, self.log_dir = ent_cmd, log_dir
        self.url, self.guestUser, self.guestPasswd = url, guestUser, guestPasswd
        self.session, self.user, self.passwd = session, user, passwd
        ent_ctor = """ENT(url="%s", session="%s", user="%s", passwd="%s", guestUser="%s", guestPasswd="%s")""" % (self.url, self.session, self.user, self.passwd, self.guestUser, self.guestPasswd)
        CmdObjProxy.__init__(self, 'ent', init='ent=%s'%(ent_ctor))

    def check(self, host1, pg1, host2, pg2):
        file_name = '%s-%s-%s-%s'%(host1, pg1, host2, pg2)
        log = '%s/%s.check'%(self.log_dir, file_name.replace('/', '_'))
        return self.asyncall('check', log, host1, pg1, host2, pg2)

    def getCheckingStatus(self):
        logs = os.listdir(self.log_dir)
        def safe_eval(x):
            try:
                return eval(x)
            except exceptions.Exception:
                return None
        def toCheckingStatus(log):
            path = os.path.join(self.log_dir, log)
            try:
                f = open(path)
                lines = f.readlines()
                status = map(safe_eval, lines[1:])
                status = filter(None, status)
                if not status: return None
                return status[-1]
            except exceptions.IOError:
                return None
        status = map(toCheckingStatus, logs)
        status = filter(None, status)
        return status

if __name__ == '__main__':
    import config
    agent = Agent(config.ent_cmd, config.url, user=config.user, passwd=config.passwd, guestUser=config.guestUser, guestPasswd=config.guestPasswd)
    result = cmd_app_run(locals(), sys.argv[1], sys.argv[1:])
    print result
