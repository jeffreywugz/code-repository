from java.net import *
from java.util import *
import urlparse
from com.vmware.vim25 import *
from com.vmware.vim25.mo import *
from com.vmware.vix import *
from common import *

class Vix:
    def __init__(self, url, user, passwd, guestUser, guestPasswd):
        self.ids = idSequence()
        self.url, self.user, self.passwd = url, user, passwd
        self.guestUser, self.guestPasswd = guestUser, guestPasswd
        self.vix = VixHostHandle(VixConstants.VIX_API_VERSION,
                                 VixServiceProvider.VIX_SERVICEPROVIDER_VMWARE_VI_SERVER,
                                 url, 0, user, passwd)
        
    def getVixVm(self, vmx):
        return self.vix.openVm(vmx)

    def shell(self, vixVm, cmd):
        vixVm.loginInGuest(self.guestUser, self.guestPasswd, 0)
        vixVm.runScriptInGuest('/bin/bash', cmd, False)

    def put(self, vixVm, local, remote):
        vixVm.loginInGuest(self.guestUser, self.guestPasswd, 0)
        vixVm.copyFileFromHostToGuest(local, remote)
        
    def get(self, vixVm, remote, local):
        vixVm.loginInGuest(self.guestUser, self.guestPasswd, 0)
        vixVm.copyFileFromGuestToHost(remote, local)
        
    def popen(self, vixVm, cmd):
        fileName = '/tmp/out.%d'%(self.ids.next())
        newCmd = '%s >%s 2>&1'%(cmd, fileName)
        self.shell(vixVm, newCmd)
        self.get(vixVm, fileName, fileName)
        f = open(fileName)
        content = f.read()
        f.close()
        return content

    def __repr__(self):
        return """Vix(guestUser='%s', guestPasswd='%s')
"""%(self.guestUser, self.guestPasswd)
    
    
class VC:
    def __init__(self, url, guestUser, guestPasswd, session=None, user="", passwd=""):
        self.url, self.session, self.user, self.passwd = url, session, user, passwd
        if session:
            self.si = ServiceInstance(URL(url), session, True)
        else:
            self.si = ServiceInstance(URL(url), user, passwd, True)
        ticket = self.si.sessionManager.acquireCloneTicket()
        self.vix = Vix(url, None, ticket, guestUser, guestPasswd)
        self.root = InventoryNavigator(self.si.rootFolder)
        self.dc = self.root.searchManagedEntity('Datacenter', 'ent')
        self.fields = self.getCustomFields()
        
    def __del__(self):
        self.si.getServerConnection().logout()

    def getServerConnection(self):
        return self.si.serverConnection
    
    def registerVm(self, host, name, vmx):
        vmFolder = self.dc.getVmFolder()
        pool = host.getParent().getResourcePool()
        vmFolder.registerVM_Task(vmx, name, False, pool, host).waitForMe()

    def listPlugin(self):
        return self.si.extensionManager.getExtensionList()

    def hasPlugin(self, key):
        extList = self.listPlugin()
        keyList = [ext.key for ext in extList]
        return key in keyList
    
    def unregisterPlugin(self, key):
        extMgr = self.si.extensionManager
        extMgr.unregisterExtension(key)
        
    def registerPlugin(self, key, url, adminEmail='hxi@vmware.com', description='no msg.', version='1.0.0', company='VMware, Inc.'):
        extMgr = self.si.extensionManager
        ext0 = extMgr.getExtensionList()[0]
        desc = makeDynamicObj(Description, label=description, summary=description)
        server = makeDynamicObj(ExtensionServerInfo, adminEmail=[adminEmail], type='com.vmware.vim.viClientScripts', url=url, description=desc, company=company)
        client = makeDynamicObj(ExtensionClientInfo, version=version, company=company, description=desc, type='com.vmware.vim.viClientScripts', url='')
        
        ext = makeDynamicObj(Extension, key=key, server=[server], client=[client], version=version, description=desc, subjectName=description, company=company, lastHeartbeatTime=Calendar.getInstance())
        extMgr.registerExtension(ext)
        
    # basic
    def getHosts(self):
        return self.root.searchManagedEntities('HostSystem')
        
    def getHost(self, name):
        return self.root.searchManagedEntity('HostSystem', name)
    
    def getAllVms(self):
        return self.root.searchManagedEntities('VirtualMachine')
        
    def getVms(self, host):
        return  InventoryNavigator(host).searchManagedEntities('VirtualMachine')
    
    def getVm(self, name):
        return self.root.searchManagedEntity('VirtualMachine', name)

    # custom fields
    def getCustomFields(self):
        fieldsManager = self.si.customFieldsManager
        fields = fieldsManager.getField()
        if not fields: return []
        return dict([(f.name, f.key) for f in fields])
        
    def addCustomFieldDef(self, name, type):
        fields = self.si.customFieldsManager
        fields.addCustomFieldDef(name, type, None, None)

    def safeAddCustomFieldDef(self, name, type):
        if name in self.fields.keys():
            return
        self.addCustomFieldDef(name, type)
        self.fields = self.getCustomFields()

    def getCustomFieldKey(self, name):
        return self.fields[name]

    def getCustomFieldName(self, name):
        invert_fields = dict([(key, name) for (name, key) in self.fields])
        return invert_fields[name]

    def setValue(self, mo, name, value):
        mo.setCustomValue(name, value)
        
    def getValue(self, mo, name, default=""):
        key = self.getCustomFieldKey(name)
        values = mo.getCustomValue()
        if not values: values = []
        values = filter(lambda x: x.key == key, values)
        if values: return values[0].value
        else: return default
        
    def setGlobal(self, name, value):
        self.setValue(self.si.rootFolder, name, value)

    def getGlobal(self, name, default=""):
        return self.getValue(self.si.rootFolder, name, default)

    # Vix
    def getVmx(self, vm):
        return vm.getPropertyByPath('summary.config.vmPathName');

    def popen(self, vm, cmd):
        return self.vix.popen(self.vix.getVixVm(self.getVmx(vm)), cmd)
        
    # fileManger
    def mkdir(self, path):
        self.si.fileManager.makeDirectory(path, self.dc, True)
        
    def copy(self, src, dest):
        self.si.fileManager.copyDatastoreFile_Task(src, self.dc, dest, self.dc, True).waitForMe()

    # network
    def getMac(self, vm):
        net_devices = filter(lambda x: type(x) == VirtualVmxnet3, vm.getPropertyByPath('config.hardware.device'))
        assert(len(net_devices) == 1)
        eth0 = net_devices[0]
        return eth0.macAddress
        
    def getPortgroups(self, host):
        pgs = host.hostNetworkSystem.networkInfo.portgroup
        return pgs
        # return [pg.spec.name for pg in pgs] # How to Identify SystemManagement portgroup?

    def getPortgroup(self, host, pgName):
        pgs = self.getPortgroups(host)
        pg = filter(lambda x: x.spec.name == pgName, pgs)
        assert(len(pg) == 1)
        return pg[0]

    def __repr__(self):
        obj = ObjFormatter(self)
        obj.hosts = [h.name for h in self.getHosts()]
        return obj.format(['url', 'user', 'passwd'], ['hosts','vix'])

    def test(self, hostName, vmName):
        def test_msg(msg):
            print '\n==============VC %s test=============='% msg
        # test_msg('basic')
        # print self
        # print self.getHosts()
        # host = self.getHost(hostName)
        # print host
        # print self.getVms(host)
        vm = self.getVm(vmName)
        print vm
        
        test_msg('vix')
        vmx = self.getVmx(vm)
        print vmx
        print self.popen(vm, 'hostname')

        # test_msg('custom field')
        # self.safeAddCustomFieldDef('VCTest', None)
        # self.setGlobal('VCTest', 'vc-test')
        # print self.getGlobal('VCTest')

        # test_msg('network')
        # pgs = self.getPortgroups(host)
        # print [pg.spec.name for pg in pgs]
        # self.getMac(vm)

        # test_msg('plugin')
        # print [ext.key for ext in self.listPlugin()]
        # self.registerPlugin(key='my.first.extension', url='http://10.117.4.52:8000/plugin.xml')
