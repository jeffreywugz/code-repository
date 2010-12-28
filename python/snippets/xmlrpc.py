#!/usr/bin/python2

import xmlrpclib

server = xmlrpclib.ServerProxy("http://time.xmlrpc.com")

currentTimeObj = server.currentTime
currtime = currentTimeObj.getCurrentTime()

print currtime
print currtime.value
