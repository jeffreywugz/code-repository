#!/bin/env python3
'''
ax.py: log replication server
'''
from axbase import *
import time
import signal

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

class AxServer:
    def __init__(self, cfg):
        self.cfg = cfg;
        self.is_req_stop = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_event(self, arg):
        if arg == 'timer':
            pass
        else:
            pass
    def handle_signal(self, sig, frame):
        logging.warn('receive sig: {}, stop server'.format(sig))
        self.stop()

    def thread_func(self, arg):
        logging.info('start thread: {}'.format(arg))
        while not self.is_req_stop:
            self.handle_event(arg)
        logging.info('stop thread: {}'.format(arg))
    def start(self):
        for arg in 'timer net'.split():
            threading.Thread(target=self.thread_func, args=(arg,), name=arg).start()
    def stop(self):
        self.is_req_stop = True
    def main_loop(self):
        logging.info('start main_loop')
        while not self.is_req_stop:
            time.sleep(1)
        logging.info('stop main_loop')

class AxClient:
    def __init__(self):
        pass
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
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    app = AxApp()
    app.run(sys.argv[1:])
    
