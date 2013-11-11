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

class Timer:
    def __init__(self, interval):
        self.interval, self.last_trigger_time = 0, 0
    def get(self):
        cur_time = time.time()
        if self.last_trigger_time + self.interval  > cur_time:
            return False
        self.last_trigger_time = cur_time
        return True
class RPC:
    def __init__(self, addr, timeout=0.5):
        self.addr = addr
        self.timer = Timer(timeout)
        ip, port = get_ip_port(addr)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.settimeout(timeout/10)
        logging.warning('listen at %s, timeout=%f'%(addr, timeout))
    def __del__(self):
        self.sock.close()
        logging.warning('close sock %s'%(addr, timeout))
    def recv(self, timeout):
        if self.timer.get():
            return None
        data, addr = sock.recvfrom(1024)
        if not data:
            logging.error('recv null pkt')
        return MSG.decode(data)
    def send(self, msg):
        return self.sock.sendto(get_ip_port(msg.get_dest()), msg.encode())

class Env(dict):
    def __init__(self, _dict_={}, **kw):
        dict.__init__(self)
        self.update(_dict_, **kw)

    def __getattr__(self, name):
        return self.get(name)
class ViewChangeWorker:
    def __init__(self, meta):
        self.meta = meta
    def handle_pkt(self, msg):
        pass
class NormalWorker:
    def __init__(self):
        pass
    def handle_pkt(self, msg):
        pass
class KeepAliveWorker:
    def __init__(self, rpc, keep_alive_timeout):
        self.rpc = rpc
        self.last_keep_alive_time = 0
        self.keep_alive_timeout = keep_alive_timeout
    def handle_keep_alive(self, msg):
        cur_tiem = time.time()
        if self.last_keep_alive_time + self.keep_alive_timeout < cur_time:
            return True
class VR:
    NOT_READY = 0
    NOTIFY_STATE = 1
    FETCH_LOG = 2
    PUSH_LOG = 3

    NO_KEEP_ALIVE = 16
    DO_VIEW_CHANGE = 17
    START_VIEW = 18

    PREPARE = 32
    PREPARE_OK = 33
    COMMIT = 34
    DEFAULT_CFG = Env(addr='127.0.0.1:1234', path='vr', group=['127.0.0.1:1234'], timeout=0.01, keep_alive_time=0.5)
    def __init__(self, rpc, cfg=VR.DEFAULT_CFG):
        self.stop = False
        self.addr = cfg.addr
        self.rpc = RPC(cfg.addr, cfg.timeout)
        self.meta = Meta(cfg.addr, cfg.group)
        self.log_store = LogStore(cfg.path)
        self.handlers = dict(RPC.PREPARE_REQ=self.handler_prepare, RPC.COMMIT_REQ=self.handle_commit)
        logging.warning('VR init: %s'%(str(cfg)))
    def broadcast(self, pcode, content=None):
        for svr in self.meta.get_group():
            self.send(svr, pcode, content)
    def send(self, dest, pcode, content=None):
        return self.rpc.send(MSG(self.addr, dest, self.meta.get_epoch(), self.meta.get_viewno(), content))
    def response(self, msg, err, content=None):
        return self.send(msg.get_src(), msg.get_pcode() + 1, err, content)
    def handle_keep_alive(self):
        cur_time = time.time()
        if self.last_keep_alive_time + self.keep_alive_timeout > cur_time:
            return True
        self.broadcast(RPC.NO_KEEP_ALIVE)
        return False
    def handle_no_keep_alive(self, msg):
        if enough no_keep_alive collected:
            self.broadcast(RPC.DO_VIEW_CHANGE)
    def handle_do_view_change(self, msg):
        if enough do_view_change collected:
            self.broadcast(RPC.START_VIEW)
    def handle_start_view(self, msg):
        pass
    def handle_prepare(self, msg):
        log = json.loads(msg.get_content())
        self.log_store.add(Log(log['epoch'], log['viewno'], log['log_id'], log['content']))
    def handle_commit(self, msg):
        log = json.loads(msg.get_content())
        if self.log_store.add(Log(log['epoch'], log['viewno'], log['log_id'], log['content'])):
            err = RPC.SUCCESS
        else:
            err = RPC.STORE_FAIL
        return self.response(msg, err)
    def handle_pkt(self, msg):
        if not msg:
            return self.handle_keep_alive()
        if self.meta != msg:
            return self.response(msg, VR.NOT_READY)
        handler = self.handlers.get(msg.get_pcode())
        if not handler:
            return self.response(msg, VR.UNKNOWN_PKT)
        return handler(msg)
    def main_loop(self):
        logging.warning('mainloop started')
        while not self.stop:
            self.handle_pkt(self.rpc.recv())
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
    cfg = load_cfg(sys.argv[1])
    vr = VR(Env(cfg))
    signal.signal(signal.SIGTERM, vr.handle_sig)
    vr.main_loop()
