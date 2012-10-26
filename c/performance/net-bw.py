#!/usr/bin/env python2

import socket
import select

def server_start(addr, max_n_conn=10, poll_timeout=1):
    sock = socket.socket()
    sock.bind(addr)
    sock.listen(max_n_conn)
    poll_obj = select.epoll()
    poll_obj.register(sock.fileno(), select.EPOLL_IN|select.EPOLL_OUT)
    while True:
        for fd,events in poll_obj.poll(poll_timeout):
            
