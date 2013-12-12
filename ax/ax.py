#!/bin/env python3
'''
ax.py: log replication server
'''
from axbase import *
import time
import signal
import os

class MainRecord:
    def __init__(self):
        pass

    def load(self):
        pass

class MRStore:
    def __init__(self, path):
        pass
    def load(self, mr):
        pass
    def store(self, mr):
        pass

class ReadReq:
    def __init__(self):
        pass

class WriteReq:
    def __init__(self):
        pass

class Tape:
    def __init__(self):
        pass
    def read(self, req):
        pass
    def write(self, req):
        pass

class AxHandler:
    def handle_packet(self, pkt):
        pass

class Config:
    default_value = dict(
        self_addr = ('127.0.0.1', 8042),
    )
    def __init__(self, path):
        cfg = load_file(path)
        self.path = path
        self.cfg = {k:cfg.get(k, v) for k, v in Config.default_value.items()}
    def __getattr__(self, key):
        if key in self.cfg:
            return self.cfg.get(key)
        else:
            raise AttributeError('no attribute for {}'.format(key))
    def __str__(self):
        return '{}:{}'.format(self.path, self.cfg)

class AxServer:
    '''
    workdir has file:
    1. config:
    '''
    PING = 1
    APPEND = 2
    VOTE_REQ = 3
    FETCH = 4
    def __init__(self, workdir):
        self.workdir, self.config_file = workdir, '{}/config.py'.format(workdir)
        self.cfg = Config(self.config_file)
        logging.info('load config: %s', self.cfg)
        self.port = AxPort(self.cfg.self_addr)
        self.is_req_stop = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, sig, frame):
        self.req_stop()

    def req_stop(self):
        if self.is_req_stop:
            os.kill(0, signal.SIGKILL)
        self.is_req_stop = True

    def pop(self, cate):
        return self.port.pop({'timer':AxPktCate.Timer, 'net':AxPktCate.Normal}[cate])
    def push(self, pkt):
        self.port.push(pkt)
    def ping(self, pkt):
        pkt.src, pkt.dest = pkt.dest, pkt.src
        yield pkt
    def default_handler(self, pkt):
        logging.warn("unknown pkt: %s", pkt)
    def handle_event(self, pkt):
        if pkt == None: return None
        handler_map = {AxServer.PING:self.ping}
        handler = handler_map.get(pkt.pcode, self.default_handler)
        logging.info('handle %s', pkt)
        for pkt in handler(pkt):
            self.push(pkt)
    def thread_func(self, arg):
        logging.info('start thread: {}'.format(arg))
        while not self.is_req_stop:
            self.handle_event(self.pop(arg))
        logging.info('stop thread: {}'.format(arg))
    def start(self):
        for arg in 'timer net'.split():
            threading.Thread(target=self.thread_func, args=(arg,), name=arg).start()
    def main_loop(self):
        logging.info('start main_loop')
        while not self.is_req_stop:
            time.sleep(1)
        logging.info('stop main_loop')

def mkserver(spec):
    spec_list = spec.split(':')
    if len(spec_list) != 2: raise AxException('invalid server spec')
    ip, port = spec_list
    return ip, int(port)

class AxClient:
    def __init__(self):
        self.port = AxPort(('127.0.0.1', 0))
        self.counter = AtomicCounter()
        self.wait_queue = WaitQueue(1024)
    def send_recv(self, pcode, dest, msg, timeout):
        seq = self.counter.next()
        wait_obj = self.
        logging.info('send seq=%s, dest=%s, msg=%s', seq, dest, msg)
        pkt = Packet(pcode=pcode, dest=dest, msg=(seq,msg))
        self.port.push(pkt)
        return wait_obj.wait(seq, timeout)
        
    def ping(self, dest, msg):
        logging.info('ping dest=%s, msg=%s', dest, msg)
        pkt = Packet(pcode=AxServer.PING, dest=dest, msg=msg)
        self.port.push(pkt)
    def reconfigure(self, old_group, new_group):
        pass
    
class AxApp(AxAppBase):
    '''
    Usages:
    ./ax.py action args_for_action
    '''
    version = 'AxApp 0.01'
    def __init__(self):
        AxAppBase.__init__(self)
        
    @ExportAsCmd
    def echo(self, msg: 'str: msg to echo')->'print msg':
        pinfo(msg)

    @ExportAsCmd
    def ping(self, server:"ip:port", msg: 'str: ping msg to echo')->'server return msg':
        client = AxClient()
        client.ping(mkserver(server), msg)

    @ExportAsCmd
    def start(self, working_dir: 'str: working directory for app')->'start status':
        '''start AxServer'''
        if not working_dir: raise AxCmdArgsError('start: need specify "working_dir"')
        server = AxServer(working_dir)
        server.start()
        server.main_loop()

    @ExportAsCmd
    def reconfigure(self, old_group:'list_of_server_addr', new_group: 'list_of_server_addr'):
        '''change group members: old_group=ip1:port1,ip2:port2... new_group=ip3:port3,ip4:port4...'''
        if not old_group or not new_group: raise AxCmdArgsError('reconfigure: need specify "old_group" and "new_group"')
        client = AxClient()
        client.reconfigure(old_group.split(','), new_group.split(','))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(thread)d %(message)s', level=logging.INFO)
    app = AxApp()
    app.run(sys.argv[1:])
    
