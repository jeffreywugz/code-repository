import sys
import logging
import inspect
import traceback
import re
import json
import threading
import queue
import socket

class AxException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
    def __repr__(self):
        return 'AxException(%s)'%(self.msg)

class AxCmdArgsError(AxException):
    pass

def bind(self, mapping):
    for k, v in mapping.items():
        setattr(self, k, v)

def pcolor(msg, color):
    if sys.stdout.isatty():
        print('\033[{}m{}\033[0m'.format(color, msg))
    else:
        print(msg)
    sys.stdout.flush()

def pinfo(msg):
    return pcolor(msg, 32)

def pwarn(msg):
    return pcolor(msg, 31)

def load_file(path):
    env = dict()
    try:
        with open(path) as f:
            exec(f.read(), env, env)
    except Exception as e:
        print(e)
    finally:
        return env

class AtomicCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = 0
    def next(self):
        with self.lock:
            value = self.value
            self.value++
        return value

class WaitObj:
    def __init__(self):
        self.cond = threading.Condition()
        self.seq, self.value = -1, None

    def is_set(self, seq):
        return self.seq < seq

    def wait(self, seq, timeout):
        with self.cond:
            self.cond.wait_for(self.predict, timeout)
            if self.seq != seq:
                return None
            else:
                return self.value

    def set(self, seq, value):
        with self.cond:
            if self.seq > seq:
                return False
            else:
                self.value = value
                self.seq = seq
                return True

class WaitQueue:
    def __init__(self, qlen):
        self.qlen = qlen
        self.wait_queue = (WaitObj() for i in range(qlen))
    def get(self, i):
        return self.wait_queue[i % self.qlen]

class AxPktCate:
    Timer = 1
    Normal = 2
class Packet:
    def __init__(self, cate=AxPktCate.Normal, pcode=None, src=None, dest=None, deadline=None, msg=None):
        bind(self, locals())
    def serialize(self):
        return bytes(json.dumps(dict(cate=self.cate, pcode=self.pcode, src=self.src, dest=self.dest, deadline=self.deadline, msg=self.msg)), "utf-8")
    def deserialize(self, pktbuf):
        attrs = json.loads(str(pktbuf, "utf-8"))
        self.cate, self.pcode, self.src, self.dest, self.deadline = attrs['cate'], attrs['pcode'], attrs['src'], attrs['dest'], attrs['deadline']
        self.msg = attrs['msg']
    def __str__(self):
        return 'Packet(<{}:{}>, {}->{}, deadline={}, {})'.format(self.cate, self.pcode, self.src, self.dest, self.deadline, self.msg)

class AxTimerQueue:
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.lock = threading.Lock()
    def push(self, deadline, pkt):
        return self.queue.push((deadline, pkt))
    def pop(self):
        with self.lock:
            try:
                deadline, pkt = self.queue.get(True, 1)
                cur_time = time.time()
                if cur_time >= deadline:
                    return pkt
                else:
                    self.queue.put((deadline, pkt))
            except queue.Empty as e:
                return None

class AxUDPSocket:
    def __init__(self, addr):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr)
        self.addr = self.sock.getsockname()
        self.sock.settimeout(1)
        logging.info('bind: %s', self.addr)
    def push(self, pkt):
        pkt.src = self.addr
        pktbuf = pkt.serialize()
        logging.debug('send msg: len={}, src={}, dest={}'.format(len(pktbuf), self.addr, pkt.dest))
        return self.sock.sendto(pktbuf, tuple(pkt.dest))
    def pop(self):
        try:
            pktbuf, addr = self.sock.recvfrom(1024)
            logging.debug('recv msg: len={}, src={}, dest={}'.format(len(pktbuf), addr, self.addr))
            pkt = Packet()
            pkt.deserialize(pktbuf)
            return pkt
        except socket.timeout as e:
            return None

class AxPort:
    def __init__(self, addr):
        self.sock = AxUDPSocket(addr)
        self.timer = AxTimerQueue()
    def get_port(self, cate):
        if cate == AxPktCate.Timer:
            return self.timer
        elif cate == AxPktCate.Normal:
            return self.sock
        else:
            raise Exception('not support pkt_cate: {}'.format(cate))
    def push(self, pkt):
        return self.get_port(pkt.cate).push(pkt)
    def pop(self, cate):
        return self.get_port(cate).pop()
        

def ExportAsCmd(func):
    def wrapper(*args, **kw):
        func.__doc__
        return func(*args, **kw)
    wrapper.inner_func = func
    wrapper.export_as_cmd = True
    return wrapper

class AxAppBase:
    def __init__(self):
        pass
        
    def run(self, argv):
        def get_attr_pairs(self):
            return ((k, getattr(self, k)) for k in dir(self))
        def get_exported_cmd(kv_list):
            return ((k, v) for k, v in kv_list if callable(v) and getattr(v, 'export_as_cmd', None))
        def format_cmd_doc(kv_list):
            return '\n'.join('{} {}\n\t{}'.format(k, str(inspect.signature(v.inner_func)), v.inner_func.__doc__ or '') for k, v in kv_list)
        def show_help():
            cmd_list = get_exported_cmd(get_attr_pairs(self))
            pinfo(self.__doc__)
            pinfo(format_cmd_doc(cmd_list))
        def parse_cmd_args(args):
            return [i for i in args if not re.match('^\w+=', i)], dict(i.split('=', 1) for i in args if re.match('^\w+=', i))
        list_args, kw_args = parse_cmd_args(argv)
        logging.info(self.version)
        logging.info('execute: {} {}'.format(list_args, kw_args))
        try:
            if not list_args: raise AxCmdArgsError('need specify action')
            method = getattr(self, list_args[0], None)
            if not callable(method): raise AxCmdArgsError('not callable action: {}'.format(list_args[0]))
            try:
                inspect.getcallargs(method.inner_func, self, *list_args[1:], **kw_args)
            except TypeError as e:
                raise AxCmdArgsError(str(e))
            return method(*list_args[1:], **kw_args)
        except AxCmdArgsError as e:
            pwarn(e)
            pwarn('list_args={} kw_args={}'.format(list_args, kw_args))
            show_help()
        except Exception as e:
            pwarn(traceback.format_exc())
