#!/usr/bin/python
import urllib,urllib2

headers = {
    'Content-Type': 'application/octet-stream',
    }
path = 'admin/ab'
parameters = dict(dcPath='ha-datacenter', dsName='Storage1')
url = 'https://10.117.5.120/folder/%s?%s'%(path, urllib.urlencode(parameters))
req = urllib2.Request(url, data="abcdef", headers=headers)
auth_handler = urllib2.HTTPBasicAuthHandler()
auth_handler.add_password('VMware HTTP server', 'https://10.117.5.120', 'root', 'iambook11')
opener = urllib2.build_opener(auth_handler)
response = opener.open(req)
out = open('index.html', 'w')
out.write(response.read())

def rput(vc, dc, host, ds):
    pass
