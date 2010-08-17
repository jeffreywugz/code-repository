#!/usr/bin/env jython
from java.net import *
from com.vmware.vim25.mo import *
from com.vmware.vix import *

class Ent:
    def __init__(self, cache, url, user, passwd):
        self.cache, self.url, self.user, self.passwd = cache, url, user, passwd
        self.si = ServiceInstance(URL(url), user, passwd, True)
        self.root = InventoryNavigator(self.si.getRootFolder())
        self.hosts = map(lambda x: x.getName(), self.root.searchManagedEntities('HostSystem'))
        self.vms = map(lambda x: x.getName(), self.root.searchManagedEntities('VirtualMachine'))

    def __del__(self):
        self.si.getServerConnection().logout()
        
    def listRunningVMs(self):
        vix = VixVSphereHandle('10.117.5.120', 'root', 'iambook11')
        return vix.getRunningVms()

    def __str__(self):
        return "Ent(cache='%s', url='%s', user='%s', passwd='%s'):\n\tHosts=%s\n\tVMs=%s"%(self.cache, self.url, self.user, self.passwd, self.hosts, self.vms)

if __name__ == '__main__':
    ent = Ent(cache='/tmp/ent', url='https://10.117.4.187/sdk', user='hxi', passwd='assert(ans==42)')
    print ent.listRunningVMs()
