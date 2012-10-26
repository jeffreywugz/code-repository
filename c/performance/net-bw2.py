#!/usr/bin/env python2
'''
./netbw.py server addr packet_size
./netbw.py client addr packet_size
./netbw.py mesure addr packet_size
'''
import sys
import os
import string
import subprocess
import time
import socket

def server_start(addr, max_packet_len=1024):
    sock = socket.socket()
    sock.bind(addr)
    sock.listen(1)
    conn, client_addr = sock.accept()
    print 'conn from %s'%(str(client_addr))
    while True:
        buf = conn.recv(max_packet_len)
        if buf == 'end':
            break

def client_start(addr, max_packet_len=1024, duration=3):
    sock = socket.socket()
    sock.connect(addr)
    buf = '0' * max_packet_len
    end_time = time.time() + duration
    bytes_send = 0
    while time.time() < end_time:
        sock.send(buf)
        bytes_send += max_packet_len
    print 'bandwidth = %dbyte/s'%(bytes_send/duration)
    sock.send('end')
    sock.close()
            
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print __doc__
        sys.exit(1)
    func, addr, packet_len = sys.argv[1:]
    ip, port = addr.split(':')
    port = int(port)
    packet_len = int(port)
    if func == 'server':
        server_start((ip,port), packet_len)
    elif func == 'client':
        client_start((ip,port), packet_len, duration=3)
    elif func == 'mesure':
        cmd = 'scp $full_bin_path $ip: && ssh $ip ./$bin server $addr $packet_size'
        cmd = string.Template(cmd).substitute(full_bin_path=sys.argv[0], bin=os.path.basename(sys.argv[0]), ip=ip, addr=addr, packet_size=packet_len)
        server = subprocess.Popen(cmd, shell=True)
        time.sleep(1)
        try:
            client_start((ip, port), packet_len, duration=3)
        except Exception, e:
            print 'client_start fail', e
        finally:
            server.terminate()
    else:
        print __doc__
        sys.exit(1)
    
