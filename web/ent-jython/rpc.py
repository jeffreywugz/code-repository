#!/usr/bin/python
import simplejson as json
import urllib, urllib2
from urlparse import urljoin

def loads_args(**kw):
    return dict([(k, json.loads(v)) for (k, v) in kw.items()])

def dumps_args(**kw):
    return dict([(k, json.dumps(v)) for (k, v) in kw.items()])

def funcWrapper(func):
    def wrapper(**kw):
        kwargs = loads_args(**kw)
        result = func(**kwargs)
        return json.dumps(result)
    wrapper.exposed = True
    return wrapper

def methodWrapper(func):
    def wrapper(self, **kw):
        kwargs = loads_args(**kw)
        result = func(self, **kwargs)
        return json.dumps(result)
    wrapper.exposed = True
    return wrapper

def makeProxy(url, name):
    request = urljoin(url, name)
    def proxy(**kw):
        kwargs = dumps_args(**kw)
        kwargs = urllib.urlencode(kwargs)
        if kwargs:  _request = "%s?%s" % (request, kwargs)
        else: _request = request
        return urllib2.urlopen(_request).read()
    return proxy

class Client(object):
    def __init__(self, url):
        self.url = url
        self.__dict__.update(proxyMethods)

    def __getattr__(self, name):
        return makeProxy(self.url, name)

class Server(object):
    def __init__(self, obj):
        self.obj = obj

    def setObj(self, obj):
        self.obj = obj
        
    def __getattr__(self, name):
        func = getattr(self.obj, name)
        return funcWrapper(func)
                             
class RpcDemo:
    @methodWrapper
    def id(self):
        return "msite"
    
    @methodWrapper
    def trace(self, **kw):
        return kw
    
    @methodWrapper
    def date(self):
        return time.ctime()
    
    @methodWrapper
    def echo(self, msg="*echo*"):
        return msg
    
    @methodWrapper
    def upper(self, msg="*upper*"):
        return msg.upper()
    
    @methodWrapper
    def add(self, a, b):
        return a+b
