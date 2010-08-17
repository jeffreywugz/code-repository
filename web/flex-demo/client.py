#!/usr/bin/env python

import rpc

client = rpc.Client('http://localhost:8080/test/', 'id', 'date', 'add', 'trace')
print client.id()
print client.add(a=2,b=3)
print client.trace(a=2,b=3)
