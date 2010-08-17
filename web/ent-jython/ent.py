#!/usr/bin/env jython
import sys
import exceptions
from pprint import pprint, pformat
import threading
import time
from common import *
from vc import *

class ProbeVmPool:
    def __init__(self, vc, prefix='probe.', src = '[Storage1] probevm'):
        self.vc, self.prefix, self.src = vc, prefix, src

    def cleanup(self):
        def safeDestroy(vm):
            if vm.getPropertyByPath('runtime.powerState') == VirtualMachinePowerState.poweredOn:
                vm.powerOffVM_Task().waitForMe()
            vm.destroy_Task().waitForMe()
        map(safeDestroy, self.getAllProbeVms())

    def probeVmFilter(self, vms):
        return filter(lambda vm: vm.name.startswith(self.prefix), vms)
    
    def getAllProbeVms(self):
        return self.probeVmFilter(vc.getAllVms())
        
    def getProbeVms(self, host):
        return self.probeVmFilter(self.vc.getVms(host))

    def getVmStatus(self, vm):
        name = 'EntProbeVmState'
        return self.vc.getValue(vm, name, 'free')

    def setVmStatus(self, vm, status):
        name = 'EntProbeVmState'
        self.vc.setValue(vm, name, status)
        
    def freeVm(self, vm):
        self.setVmStatus(vm, 'free')

    def refVm(self, vm):
        self.setVmStatus(vm, 'using')

    def isFree(self, vm):
        return self.getVmStatus(vm) == 'free'

    def freeAll(self):
        map(self.freeVm, self.getAllProbeVms())
    
    def newName(self):
        names = ['%s%d'%(self.prefix, i) for i in range(1000)]
        existVmNames = [vm.name for vm in self.getAllProbeVms()]
        return [n for n in names if n not in existVmNames][0]

    def newVm(self, host, name):
        dest = '[%s] %s' %(host.getDatastores()[0].name, name)
        self.vc.copy(self.src, dest)
        self.vc.registerVm(host, name, dest + '/probevm.vmx')
        return self.vc.getVm(name)

    def findFreeVms(self, host):
        vms = self.getProbeVms(host)
        return filter(self.isFree, vms)
        
    def _alloc(self, host):
        freeVms = self.findFreeVms(host)
        if not freeVms:
            newName = self.newName()
            newVm = self.newVm(host, newName)
        freeVms = self.findFreeVms(host)
        assert(len(freeVms) > 0)
        vm = freeVms[0]
        self.refVm(vm)
        return vm

    @staticmethod
    def _configureProbeVm(vm, pg):
        nic = filter(lambda x: x.deviceInfo.label == 'Network adapter 1', vm.getPropertyByPath('config.hardware.device'))[0]
        backing = VirtualEthernetCardNetworkBackingInfo()
        backing.setDeviceName(pg)
        nic.setBacking(backing)
        # connectable = VirtualDeviceConnectInfo()
        # connectable.setConnected(True)
        # nic.setConnectable(connectable)
        nicChange = VirtualDeviceConfigSpec()
        nicChange.setDevice(nic)
        nicChange.setOperation(VirtualDeviceConfigSpecOperation.edit)
        configSpec = VirtualMachineConfigSpec()
        configSpec.setDeviceChange([nicChange])
        vm.reconfigVM_Task(configSpec).waitForMe()

    def alloc(self, host, pg):
        vm = self._alloc(host)
        self._configureProbeVm(vm, pg)
        if vm.getPropertyByPath('runtime.powerState') != VirtualMachinePowerState.poweredOn:
            vm.powerOnVM_Task(None).waitForMe()
        print self.vc.popen(vm, "kill `ps|grep eth_tool|awk '{print $1}'`; /bin/eth_tool -r -e eth0 &")
        return vm

    def free(self, vm):
        self.freeVm(vm)

    def __repr__(self):
        obj = ObjFormatter(self)
        def convertToNames(vms):
            return [vm.name for vm in vms]
        obj.vms = convertToNames(self.getAllProbeVms())
        return obj.format(['prefix', 'src'], ['vms'])
    

class ENT:
    def __init__(self, url, guestUser, guestPasswd, session=None, user='', passwd=''):
        self.url, self.session, self.user, self.passwd = url, session, user, passwd,
        self.guestUser, self.guestPasswd = guestUser, guestPasswd
        self.vc = VC(url, session=session, user=user, passwd=passwd, guestUser=guestUser, guestPasswd=guestPasswd)
        self.probeVms = ProbeVmPool(self.vc)
        self.addCustomFieldDefs()

    def cleanup(self):
        self.set(self.checkingStatusId, {})
        self.probeVms.freeAll()

    def register(self, url, key='my.ent.extension', description='ENT plugin'):
        if self.vc.hasPlugin(key):
            self.vc.unregisterPlugin(key)
        self.vc.registerPlugin(key, url, description=description)
        
    def addCustomFieldDefs(self):
        self.vc.safeAddCustomFieldDef('EntProbeVmState', 'VirtualMachine')

    def set(self, name, value):
        self.store.set(name, value)

    def get(self, name, default=None):
        return self.store.get(name, default)

    def getHosts(self):
        return [h.name for h in self.vc.getHosts()]

    def getPortgroups(self, host):
        return [(host,pg.spec.name) for pg in self.vc.getPortgroups(self.vc.getHost(host))]

    def getPortgroup(self, host, pg):
        host = self.vc.getHost(host)
        pg = self.vc.getPortgroup(host, pg)
        return pg

    def getAllPortgroups(self):
        hosts = self.getHosts()
        pgs = map(self.getPortgroups, hosts)
        return reduce(lambda x, y: x+y, pgs, [])
    
    def getVlanId(self, host, pg):
        return self.getPortgroup(host, pg).spec.vlanId
    
    def _check(self, host1, pg1, host2, pg2):
        def setStatus(status, result=None, progress=-1):
            print repr((host1, pg1, host2, pg2, status, result, progress))
        setStatus('start')
        _host1 = self.vc.getHost(host1)
        _host2 = self.vc.getHost(host1)
        setStatus('alloc vm')
        vm1 = self.probeVms.alloc(_host1, pg1)
        vm2 = self.probeVms.alloc(_host2, pg2)
        mac2 = self.vc.getMac(vm2)
        setStatus('popen')
        output = self.vc.popen(vm1, '/bin/eth_tool -e eth0 -d %s'%mac2)
        setStatus('free vm')
        self.probeVms.free(vm1)
        self.probeVms.free(vm2)
        result = output == 'success\n'
        setStatus('done', result)

    def check(self, host1, pg1, host2, pg2):
        checker = threading.Thread(target=self._check, args=(host1, pg1, host2, pg2))
        checker.start()
        
    def executeCheckPlan(self, plan):
        [self.check(*p) for p in plan]
    
    def shouldCheck(self, host1, pg1, host2, pg2):
        def isSpecialVlan(vlanId):
            # return vlanId == 4095 or vlanId == 1
            return False
        if pg1 == pg2: return True
        vlan1, vlan2 = self.getVlanId(host1, pg1), self.getVlanId(host2, pg2)
        if isSpecialVlan(vlan1) or isSpecialVlan(vlan2): return False
        return vlan1 == vlan2
        
    def checkPlan(self, *pgs):
        pgs = reduce(lambda x,y:x+y, pgs, [])
        plan = [pg1+pg2 for pg1 in pgs for pg2 in pgs if pg2 != pg1]
        plan = filter(lambda x: self.shouldCheck(*x), plan)
        return plan

    def defaultCheckPlan(self, *hosts):
        pgs = map(self.getPortgroups, hosts)
        return self.checkPlan(*pgs)
        
    def __repr__(self):
        obj = ObjFormatter(self)
        return obj.format([],['vc', 'probeVms'])

    def viewResult(self):
        while True:
            print pformat(self.getCheckingStatus())
            time.sleep(1)
            
    def test(self, *hosts):
        def test_msg(msg):
            print '\n==============ENT %s test=============='% msg
        host = hosts[0]

        # test_msg('dict store')
        # print self.checkingStatus.getAll()
        # self.checkingStatus[('h1', 'pg1', 'h2', 'pg2')] = ('start', None, -1)
        # print self.checkingStatus[('h1', 'pg1', 'h2', 'pg2')]
        
        # test_msg('basic')
        # print self
        # print self.getHosts()
        # pgs = self.getPortgroups(host)
        # pgs = [(pg, self.getVlanId(host, pg)) for host, pg in pgs]
        # print pformat(pgs)

        # test_msg('store')
        # print self.setCheckingStatus(host[0], pgs[0], host[1], pgs[1])
        # print self.getCheckingStatus()
        
        # test_msg('probe vm pool')
        # vms = self.probeVms
        # vms.freeAll()
        
        # test_msg('virtual switch checking')
        # print self.check(host, pgs[0], host, pgs[1])

        # test_msg('intra host check plan')
        # plan = self.defaultCheckPlan(*hosts)
        # pprint(plan)

        # test_msg('global storage')
        # name = 'EntID'
        # self.set(name, range(10))
        # print self.get(name)

        # test_msg('check plan')
        # self.cleanup()
        # plan = self.defaultCheckPlan(*hosts)
        # print pformat(plan)
        # self.executeCheckPlan(plan)

if __name__ == '__main__':
    import config
    init = 'ent=ENT(config.url, user=config.user, passwd=config.passwd, guestUser=config.guestUser, guestPasswd=config.guestPasswd)'
    mark = MarkCodec()
    result = cmd_app_run(locals(), sys.argv[1], sys.argv[1:], init=init)
    print mark.dumps(result)
