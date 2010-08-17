-include config
module-name = hog
ifneq ($(KERNELRELEASE),)
obj-m := $(module-name).o
$(module-name)-objs :=  main.o hog_pages.o
else

kernel-dir = /lib/modules/$(shell uname -r)/build
pwd = $(shell pwd)
cc = gcc

ifeq ($(SUPPORT_OPENMP), 1)
openmp-flags=-fopenmp
endif
ifeq ($(SUPPORT_STATIC_REMAP), 1)
static-remap-ldflags = -T hog.ld
endif

cflags = -Wall $(openmp-flags)
ldflags = -Wall $(openmp-flags)
test-objs := hog_user.o hog_test.o
test-srcs = $(patsubst %.o,%.c,$(test-objs))

.PHONY: all test clean load unload reload dep
all: test

$(module-name).ko: main.c hog_pages.c
ifeq ($(SUPPORT_KERNEL_MODULE), 1)
	make -C $(kernel-dir) M=$(pwd) modules
else
	echo "Do not support kernel module!"
	exit -1
endif

$(module-name)_test: $(test-objs)
	$(cc) $(ldflags) -o $@ $(test-objs) $(static-remap-ldflags)
%.o: %.c
	$(cc) $(cflags) -c -o $@ $<

load:
	sudo insmod ./$(module-name).ko
unload:
	sudo rmmod ./$(module-name).ko
reload:
	sudo rmmod ./$(module-name).ko
	sudo insmod ./$(module-name).ko

ifneq ($(SUPPORT_KERNEL_MODULE), 1)
test: $(module-name)_test
else
test: $(module-name).ko $(module-name)_test
	rm -f ./$(HOG_DEVICE)
	sudo mknod ./$(HOG_DEVICE) c `awk '$$2=="$(module-name)" {print $$1}' /proc/devices` 0
	sudo chgrp wheel ./$(HOG_DEVICE)
	sudo chmod 664 ./$(HOG_DEVICE)
endif
ifeq ($(SUPPORT_PFMON), 1)
	pfmon --trigger-code-start=0x`nm $(module-name)_test|grep 'T profile_run'|awk '{print $$1}'` -e memory_requests:all,requests_to_l2:data,l2_cache_miss:data ./$(module-name)_test 512k
else
	./$(module-name)_test 512k
endif

clean:
	rm -f *.o *.ko Module.symvers Module.markers modules.order *.mod.c .*.cmd $(module-name)_test $(HOG_DEVICE)

config.h: config
	cat $<|sed '/^#/d'|awk -F= 'NF ==2 {printf "#define %s %s\n", $$1, $$2}' > $@
dep:
	(for f in $(test-srcs);do echo -n "";$(cc) -E -M -MM -MG $$f;done) > .dep

### Dependencies:
-include .dep

endif

