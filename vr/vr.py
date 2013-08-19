#!/usr/bin/env python2
'''
Usages:
  ./vr.py cfg.py
#cat cfg.py
group=[]
self_addr=''
path=''
#end cfg.py

  1. get: (seq) -> (err, content)
  2. propose: (seq, content) -> (err)
  3. reclaim: (seq) -> (err)
  4. reconf: (group) -> (err)
'''
import logging
import json
import socket

class Log:
    def __init__(self, epoch, viewno, log_id, content):
        self.epoch, self.viewno, self.log_id, self.content = epoch, viewno, log_id, content
    def get_epoch(self):
        return self.epoch
    def get_viewno(self):
        return self.viewno
    def get_log_id(self):
        return self.log_id
    def get_content(self):
        return self.get_content
    def __lt__(self, other):
        return [self.get_epoch(), self.get_viewno(), self.get_log_id()]  < [other.get_epoch(), other.get_viewno(), other.get_log_id()]
        
class LogStore:
    def __init__(self, path):
        self.path, self.logs, self.last_log_id = path, [None] * 1000000, 0
    def add(self, log):
        if self.logs[log.log_id] == None or self.logs[log.log_id] < log:
            self.logs[log.log_id] = log
            self.last_log_id = log.log_id
            return True
        else:
            return False
    def get_last_log(self):
        return self.logs[self.last_log_id]
    def get_end_cursor(self):
        last_log = self.get_last_log()
        return last_log.get_epoch(), last_log.get_viewno(), last_log.get_log_id()

class Meta:
    def __init__(self, addr, group):
        self.epoch, self.viewno, self.addr, self.group, self.old_group = 0, 0, addr, group, group

class MSG:
    def __init__(self, src, dest, epoch, viewno, pcode, content):
        self.src, self.dest, self.epoch, self.viewno, self.pcode, self.content = src, dest, epoch, viewno, pcode, content
    def encode(self):
        return json.dumps(dict(src=self.get_src(), dest=self.get_dest(), viewno=self.get_viewno(), pcode=self.get_pcode(), content=self.get_content()))
    @staticmethod
    def decode(buf):
        d = json.loads(buf)
        return MSG(d['src'], d['dest'], d['viewno'], d['pcode'], d['content'])

def get_ip_port(addr):
    ip, port = addr.split(':')
    return ip, int(port)
class RPC:
    PROPOSE_REQ = 0
    PROPOSE_RESP = 1

    PREPARE_REQ = 16
    PREPARE_RESP = 17
    COMMIT_REQ = 18
    COMMIT_RESP = 19
    FETCH_REQ  = 20
    FETCH_RESP  = 21

    PROPOSE_VIEW_CHANGE = 32
    DO_VIEW_CHANGE = 33
    START_VIEW = 34
    def __init__(self, addr, timeout=0.001):
        self.addr = addr
        ip, port = get_ip_port(addr)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.settimeout(timeout)
    def __del__(self):
        self.sock.close()
    def recv(self, timeout):
        data, addr = sock.recvfrom(1024)
        return MSG.decode(data)
    def send(self, msg):
        return self.sock.sendto(get_ip_port(msg.get_dest()), msg.encode())

class VR:
    DEFAULT_CFG = dict(addr='127.0.0.1:1234', path='vr', group=['127.0.0.1:1234'], timeout=0.01)
    def __init__(self, rpc, cfg=VR.DEFAULT_CFG):
        self.stop = False
        self.addr = cfg['addr']
        self.rpc = RPC(cfg['addr'], cfg['timeout'])
        self.meta = Meta(cfg['addr'], cfg['group'])
        self.log_store = LogStore(cfg['path'])
        self.handlers = dict(RPC.PREPARE_REQ=self.handler_prepare, RPC.COMMIT_REQ=self.handle_commit)
    def try_catchup(self):
        pass
    def send(self, dest, pcode, err, content=None):
        return self.rpc.send(MSG(self.addr, dest, self.meta.get_epoch(), self.meta.get_viewno(), content))
    def response(self, msg, err, content=None):
        return self.send(msg.get_src(), msg.get_pcode() + 1, err, content)
    def handle_propose(self, msg):
        pass
    def handle_prepare(self, msg):
        log = json.loads(msg.get_content())
        if self.log_store.add(Log(log['epoch'], log['viewno'], log['log_id'], log['content'])):
            err = RPC.SUCCESS
        else:
            err = RPC.STORE_FAIL
        return self.response(msg, err)
    def handle_commit(self, msg):
        log = json.loads(msg.get_content())
        if self.log_store.add(Log(log['epoch'], log['viewno'], log['log_id'], log['content'])):
            err = RPC.SUCCESS
        else:
            err = RPC.STORE_FAIL
        return self.response(msg, err)
    def handle_pkt(self, msg):
        if self.meta != msg:
            return self.response(msg, VR.NOT_READY)
        handler = self.handlers.get(msg.get_pcode())
        if not handler:
            return self.response(msg, VR.UNKNOWN_PKT)
        return handler(msg)

    def catchup(self):
        self.rpc.send()
        pass
    def keep_ha(self):
        pass
    def main_loop(self):
        logging.warning('mainloop started')
        while not self.stop:
            self.catchup() and self.keep_ha() and self.handle_pkt(self.rpc.recv())
        logging.warning('mainloop stopped')
    def handle_sig(self, signo):
        self.stop = True

def load_cfg(path):
    env = dict()
    try:
        with open(path) as f:
            exec(f.read(), env)
    except IOError,e:
        print traceback.format_exc()
    finally:
        return env

def help(msg):
    sys.stderr.write(msg)
    sys.stderr.write(__doc__)

if __name__ == '__main__':
    len(sys.argv) < 2 or help("invalid argument: need 1 argument") or sys.exit(1)
    vr = VR(load_cfg(sys.argv[1]))
    signal.signal(signal.SIGTERM, vr.handle_sig)
    vr.main_loop()
