#!/usr/bin/env python

import sys, os, re
import exceptions, traceback
import StringIO
import cherrypy
from cherrypy.lib import safemime
safemime.init()
from cherrypy.lib.static import serve_file
import mako.template
import simplejson as json
import rpc
from agent import *
import config

cwd = os.path.dirname(os.path.abspath(__file__))

class GErr(exceptions.Exception):
    def __init__(self, msg, obj=None):
        exceptions.Exception(self)
        self.obj, self.msg = obj, msg

    def __str__(self):
        return str((self.msg, self.obj))

class Template:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def render(self, templateFileName, **kw):
        path = os.path.join(self.base_dir, templateFileName)
        return mako.template.Template(filename=path).render(**kw)

template = Template(cwd)

class FileStore:
    def __init__(self, storeFile, defaultValue=None):
        self.storeFile, self.defaultValue = storeFile, defaultValue

    @rpc.methodWrapper
    def get(self):
        try:
            with open(self.storeFile) as f:
                value = json.loads(f.read())
        except (exceptions.IOError, exceptions.ValueError):
            value = self.defaultValue
        return value

    @rpc.methodWrapper
    def save(self, value):
        with open(self.storeFile, 'w') as f:
            f.write(json.dumps(value))
        return True
    

@rpc.funcWrapper
def shell(expr='"True"'):
    result = None
    exception = None
    try:
        result = eval(expr, globals(), locals())
    except exceptions.Exception,e:
        exception = e
        result = str(exception)
    return result

class AgentRpc(rpc.Server):
    def __init__(self):
        rpc.Server.__init__(self, object())

    def set(self, serviceUrl, sessionId, serverGuid):
        self.serviceUrl, self.sessionId, self.serverGuid = serviceUrl, sessionId, serverGuid
        self.agent = Agent(config.ent_cmd, serviceUrl, session=sessionId, guestUser=config.guestUser, guestPasswd=config.guestPasswd)
        self.setObj(self.agent)

    @rpc.methodWrapper
    def status(self):
        return self.agent.getCheckingStatus()

    @rpc.methodWrapper
    def getAllPortgroups(self):
        return self.agent.getAllPortgroups()
    
    @rpc.methodWrapper
    def inspect(self):
        return dict(serviceUrl=self.serviceUrl, sessionId=self.sessionId)
    
class Root:
    def __init__(self):
        self.agent = AgentRpc()
        
    @cherrypy.expose
    def index(self):
        return str(self)

    @cherrypy.expose
    def ent(self, serviceUrl='', sessionId='', serverGuid='', **kw):
        if serviceUrl:
            self.agent.set(serviceUrl, sessionId, serverGuid)
        return serve_file(os.path.join(cwd, 'ent.html'))
        
    def __str__(self):
        return str(self.__dict__.keys())

default_config = {'/': {'tools.staticdir.on': True, 'tools.staticdir.dir': cwd, 'tools.safe_multipart.on': True},
          }
root = Root()
root.shell = shell
root.test = rpc.RpcDemo()


cherrypy.tree.mount(root, '/', config=default_config)

def server_start(host='0.0.0.0', port=8080):
    cherrypy.server.socket_host = host
    cherrypy.server.socket_port = port
    cherrypy.engine.start()

if __name__ == '__main__':
    server_start()
