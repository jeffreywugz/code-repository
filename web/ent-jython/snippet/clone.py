    def clone(self, vm, host, target):
        cloneSpec = VirtualMachineCloneSpec()
        relocateSpec = VirtualMachineRelocateSpec()
        relocateSpec.setHost(host.getMOR())
        relocateSpec.setPool(host.getParent().getResourcePool().getMOR())
        relocateSpec.setDatastore(host.getDatastores()[0].getMOR())
        cloneSpec.setLocation(relocateSpec)
        cloneSpec.setPowerOn(False)
        cloneSpec.setTemplate(False)
        vm.cloneVM_Task(self.getVm('probe.1').getParent(), target, cloneSpec).waitForMe()
        
