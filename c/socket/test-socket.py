#!/bin/env python2

import socket
import time

HOST,  PORT = '',50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
time.sleep(30)
conn, addr = s.accept()
print 'Accept'
print 'Connected by', addr
while 1:
    data = conn.recv(1024)
    if not data: break
    conn.sendall(data)
conn.close()
