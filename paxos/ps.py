#!/usr/bin/python2
'''
./ps.py cfg:start
./ps.py cfg:reconfigure
./ps.py cfg:propose 'content-str'
# cfg example
self='ip:port'
group='ip1:port1 ip2:port2'.split()
'''

import sys
import traceback

class Fail(Exception):
    def __init__(self, msg, obj=None):
        self.msg, self.obj = msg, obj
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return 'Fail(%s, %s)'%(self.msg, self.obj)

def load_cfg(path):
    env = dict()
    try:
        with open(path) as f:
            exec(f.read(), env)
    except IOError,e:
        print traceback.format_exc()
    finally:
        return env

class PaxosInstance:
    def __init__(self, iseq):
        self.iseq, self.mtime, self.final_accept_proposer, self.final_accept_value, self.checksum, self.total_checksum = iseq, 0, None, None, 0, 0 # commitlog info
        self.last_resp_propose_seq, self.last_resp_accept_seq, self.last_resp_accept_value = 0, 0, None # acceptor info
        self.group_members, self.proposer_idx, self.last_used_propose_seq = [], 0,  0 # proposer info
    def catchup(self, end_cursor):
        if self.iseq < end_cursor.iseq:
            pass
    def propose(self, overview):
        if overview.last_update_time + ov_live_time < time.time() or self.iseq < overview.end_cursor.iseq:
            return NEED_CATCHUP
class Overview:
    def __init__(self, max_settle_pi):
        self.max_settle_pi = max_settle_pi

class PaxosInstanceSet:
    def __init__(self):
        self.pis = []
    def get(self, idx):
        if idx < 0 || idx >= len(self.pis):
            return None
        return self.pis[idx]

class PaxosServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.thread = threading.Thread(target=lambda :self.main_loop())
    def main_loop(self):
        while True:
            instance = self.pis.get_next_unsettle_instance()
            instance.catchup(overview()) or instance.propose(self.pkt_queue.pop())
    def do_admin(self, cmd, *args):
        func = getattr(self, cmd)
        if not func: raise Fail('not support cmd', cmd)
        return cmd(*args)
    def start(self):
        print 'start'
    def reconfigure(self):
        print 'leave'
    def propose(self, arg):
        print 'propose'

def help(msg):
    sys.stderr.write(msg)
    sys.stderr.write(__doc__)

if __name__ == '__main__':
    len(sys.argv) < 2 or help("invalid argument: need 1 argument") or sys.exit(1)
    m = re.match('(?:([.+]):)(.+)', sys.argv[1])
    if not m:
        help("invalid argument: should has the form 'config:cmd'")
        sys.exit(2)
    cfg_file, cmd = m.group(1) or 'cfg.py', m.group(2)
    print PaxosServer(load_cfg(cfg_file)).do_admin(cmd, *sys.argv[2:])
