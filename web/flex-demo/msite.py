#!/usr/bin/env python

import sys, os, re
import exceptions, traceback
import time, datetime
import stat
import StringIO
import cherrypy
from cherrypy.lib import safemime
import mako.template
import simplejson as json
import rpc
import rlog
import subprocess
import random
safemime.init()

cwd = os.path.dirname(os.path.abspath(__file__))
adminHost = 'admin.local'
hadoopDir = '/home/hadoop'

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
    
template = Template(cwd)

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

class Root:
    @cherrypy.expose
    def index(self):
        return str(self)

    @rpc.methodWrapper
    def genID(self):
        try:
            with open('globals', 'r') as f:
                vars = json.loads(f.read())
        except (exceptions.IOError, exceptions.ValueError):
            vars = {'counter':0}
        id = vars['counter']
        vars['counter'] = id+1
        with open('globals', 'w') as f:
            f.write(json.dumps(vars))
        return id
    
    def __str__(self):
        return str(self.__dict__.keys())

class Preference(FileStore):
    def __init__(self, fileName):
        FileStore.__init__(self, fileName, {'userName':'user', 'userEmail':'user@abc.com', 'defaultClusterSize':4, 'emailMe':"No"})

class ClusterList(FileStore):
    def __init__(self, fileName, log):
        self.log = log
        FileStore.__init__(self, fileName, [])

    @rpc.methodWrapper
    def save(self, value):
        [cluster.update(dict(mpLabel="Not Availible.", mpTotal=-1)) for cluster in value]
        with open(self.storeFile, 'w') as f:
            f.write(json.dumps(value))
        return True
        
    def makeCmd(self, name, operation, restArgs=""):
        return 'ssh root@admin.local "cd /share/vhadoop-scripts; ./vhadoop-scripts.pl --operation=%(operation)s --cluster_conf=../flex-demo/%(name)s.xml %(restArgs)s"'%{
            'name':name, 'operation':operation, 'restArgs':restArgs}
    
    def shell(self, name, operation, restArgs=""):
        cmd = self.makeCmd(name, operation, restArgs)
        print cmd
        p = self.log.shell(cmd)
        p.wait()

    def popen(self, name, operation, restArgs=""):
        cmd = self.makeCmd(name, operation, restArgs)
        print cmd
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        content = p.stdout.read()
        p.wait()
        if p.returncode != 0:
            print "Error: with code %d\n"%p.returncode
        return content
    
    def genConfigFile(self, cluster):
        with open('%(name)s.xml'%cluster, 'w') as f:
            f.write(template.render('config.xml', **cluster))
            
    @rpc.methodWrapper
    def update(self, cluster):
        self.genConfigFile(cluster)
        self.shell(cluster['name'], 'resize', '--num_vms=%s'%cluster['numVMs'])
        cluster.update(mpLabel='Cluster created, Deploy Hadoop Service...', mpTotal = 0)
        return cluster

    @rpc.methodWrapper
    def deploy(self, cluster):
        self.genConfigFile(cluster)
        self.shell(cluster['name'], 'poweron')
        self.shell(cluster['name'], 'deploy')
        cluster.update(mpLabel = "Hadoop Service is Ready.", mpTotal = -1)
        return cluster

    @rpc.methodWrapper
    def getStatus(self):
        return {'hadoop':{'mpLabel':'Creating', 'mpTotal':0}}
    
    def getClusterIp(self, cluster):
        self.genConfigFile(cluster)
        ip = self.popen(cluster['name'], 'getNameNodeIP')
        return ip
        
class Cluster:
    def __init__(self, host, basedir, log):
        self.host, self.basedir, self.log = host, basedir, log

    def makeCmd(self, cmd):
        return 'ssh root@%(host)s "cd %(basedir)s; %(cmd)s"'%{'host':self.host, 'basedir':self.basedir, 'cmd':cmd}
    
    def shell(self, cmd):
        cmd = self.makeCmd(cmd)
        p = self.log.shell(cmd)
        p.wait()
        
    def popen(self, cmd):
        p = subprocess.Popen(self.makeCmd(cmd), shell=True, stdout=subprocess.PIPE)
        return p.stdout.read()

    def listFile(self):
        return self.popen('ls').split()[1:]

    def fileInfo(self, name):
        return dict(name=name)

    def uploadFile(self, local, remote):
        return os.system("scp %(local)s root@%(host)s:%(basedir)s/%(remote)s"%{'host':self.host, 'basedir':self.basedir, 'local':local, 'remote':remote})
    
    def listJob(self):
        return [job.split()[0] for job in self.popen('bin/hadoop job -list all').split('\n')[4:-1] ]

    def jobInfo(self, jobID):
        rawInfo = self.popen("bin/hadoop job -status %s"%(jobID))
        pattern = r"map\(\) completion: ([.0-9]+)\nreduce\(\) completion: ([.0-9]+)" 
        match = re.search(pattern, rawInfo)
        mapCompletion, reduceCompletion = match.groups()
        return dict(jobID=jobID, loaded=float(mapCompletion) + float(reduceCompletion), total=2.0)
    
    def submitJob(self, jar, className, input, output, args):
        self.shell("bin/hadoop jar %s %s %s %s %s" % (jar, className, input, output, args))
        
class JobList(FileStore):
    def __init__(self, fileName, clusterList, log):
        self.clusterList, self.log = clusterList, log
        FileStore.__init__(self, fileName, [])

    @rpc.methodWrapper
    def submit(self, job):
        if not (job['cluster'] and job['jar'] and job['className'] and job['input'] and job['output']):
            raise GErr('illformed job arguments!', job)
        ip = self.clusterList.getClusterIp({'name':job['cluster']})
        if not ip: raise GErr('master ip not configured!', job)
        os.system("scp %(jar)s root@%(host)s:/home/hadoop"%dict(jar=job['jar'], host=ip))
        hadoop = Cluster(ip, hadoopDir, self.log)
        hadoop.submitJob(job['jar'], job['className'], job['input'], job['output'], job['args'])
        job.update(mpLabel="JobFinished.", mpTotal=-1)
        return job

    @rpc.methodWrapper
    def getSubmitted(self):
        return [{'cluster':'hadoop', 'jar':'main', 'className':'grep', 'mpLabel':'Executing', 'mpLoaded':random.randint(0,100), 'mpTotal':100}]
    
    @cherrypy.expose
    def upload(self, fileContent, Filename, **kw):
        with open(Filename,'w') as f:
            f.write(fileContent.file.read())

    @cherrypy.expose
    def report(self, job):
        job = json.loads(job)
        ip = self.clusterList.getClusterIp({'name':job['cluster']})
        if not ip: raise GErr('master ip not configured!', job)
        job.update(nameNode = ip)
        return template.render('report.html', **job)
        
class File:
    def __init__(self, name, owner, size, date):
        self.name, self.owner, self.size, self.date = name, owner, size, date

    @staticmethod
    def new(path):
        st = os.stat(path)
        date = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(st[stat.ST_MTIME]))
        size = st[stat.ST_SIZE]
        owner = st[stat.ST_UID]
        try:
            owner = pwd.getpwuid(owner).pw_name
        except exceptions.KeyError:
            pass
        return File(path, owner, size, date)

    def dict(self):
        return dict(name=self.name, owner=self.owner, size=self.size, date=self.date)
    
    def __str__(self):
        return simplejson.dumps(self.dict())
        
class FileExplorer:
    def __init__(self):
        pass
        
    @rpc.methodWrapper
    def get(self):
        return []

    @cherrypy.expose
    def add(self, fileContent, Filename, **kw):
        with open(Filename,'w') as f:
            f.write(fileContent.file.read())

log = rlog.RLog()
clusterList = ClusterList('cluster.list', log)
root = Root()
root.log = log
root.shell = shell
root.test = rpc.RpcDemo()
root.preference = Preference('preference.opt')
root.clusterList = clusterList
root.jobList = JobList('job.list', clusterList, log)
root.fileExplorer = FileExplorer()

config = {'/': {'tools.staticdir.on': True, 'tools.staticdir.dir': cwd, 'tools.safe_multipart.on': True},
          }

cherrypy.tree.mount(root, '/', config=config)

def server_start(host='0.0.0.0', port=8080):
    cherrypy.server.socket_host = host
    cherrypy.server.socket_port = port
    cherrypy.engine.start()

if __name__ == '__main__':
    server_start()
