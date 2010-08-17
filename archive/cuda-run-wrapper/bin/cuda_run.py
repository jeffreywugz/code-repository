#!/usr/bin/python

import os
import sys
import time

bin_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.realpath(os.path.join(bin_dir, ".."))
cudarun_daemon = "bin/cudarun_daemon"
cudarun_daemon_config_file = "etc/config.conf"
cudarun_lib = "lib/libcudarun.so"

def get_server_addr():
    lines = open(os.path.join(base_dir, cudarun_daemon_config_file)).readlines()
    items = [line.strip().split('=') for line in lines]
    return dict(items)['server.addr']

server_addr = get_server_addr()

def run(exe):
    os.system('LD_PRELOAD=%s server_addr=%s %s'%(os.path.join(base_dir, cudarun_lib), server_addr, exe))
    
class Daemon:
    def __init__(self, base_dir, daemon, config_file):
        self.base_dir, self.daemon, self.config_file = base_dir, daemon, config_file
        

    def is_running(self):
        return os.system('ps -C %s -o pid= >/dev/null'%os.path.basename(self.daemon)) == 0

    def check(self):
        if self.is_running():
            print "daemon is running."
        else:
            print "daemon is not running"
            
    def start(self):
        if self.is_running():
            return
        self._start()

    def stop(self):
        if not self.is_running():
            return
        self._stop()
        
    def _start(self):
        daemon = os.path.join(self.base_dir, self.daemon)
        config_file = os.path.join(self.base_dir, self.config_file)
        os.system("%s --config-file %s&" %(daemon, config_file))
        time.sleep(0.1)

    def _stop(self):
        os.system('pkill %s'% os.path.basename(self.daemon))
        time.sleep(0.1)

    def restart(self):
        self.stop()
        self.start()

daemon = Daemon(base_dir, cudarun_daemon, cudarun_daemon_config_file)
action = "%s(*%s)" %(sys.argv[1], sys.argv[2:])
eval(action)
