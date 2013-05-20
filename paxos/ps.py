#!/usr/bin/python2
'''
./ps.py cfg:join
./ps.py cfg:leave
./ps.py cfg:status
'''

import sys

class MsgDeliver:
    pass

class PS:
    def __init__(self, cfg):
        pass
    def do_admin(self, cmd):
        
    def propose(self):
        pass

def help(msg):
    sys.stderr.write(msg)
    sys.stderr.write(__doc__)

if __name__ == '__main__':
    len(sys.argv) == 2 or help("invalid argument: need 1 argument") or sys.exit(1)
    m = re.match('(?:([.+]):)(.+)', sys.argv[1])
    if not m:
        help("invalid argument: should has the form 'config:cmd'")
        sys.exit(2)
    cfg_file, cmd = m.group(1) or 'ps.ini', m.group(2)
    print PaxosServer(load_cfg(cfg_file)).do_admin(cmd)
