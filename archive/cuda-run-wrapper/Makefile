prj.name = cuda-run
prj.version = 0.1
config-file = etc/config.conf
cuda-run-exec-script = bin/cuda_run.py
cuda-run-profile-script = bin/cuda_profile.py
prj.pkg = etc/config.conf $(cuda-run-daemon.exe) $(cuda-run-exec-script) $(cuda-run-lib.lib) doc/reference

CC = gcc -Wall
CPP = gcc -E
CPPFLAGS = -I/usr/local/cuda -Ilibphi -Idevice -Idaemon -Ilibcudarun
NVCC = nvcc
CUDA-PREFIX = /usr/local/cuda

all: doc
.PHONY: pkg test profile doc bak clean etags dep

cuda-run-lib.objs = $(patsubst %.c,%.o,$(wildcard libcudarun/*.c))
cuda-run-lib.private-objs =  $(patsubst %.o,%-cuda-run-lib.o,libphi/unsocket.o device/device_list.o device/device.o device/device_info.o device/device_evaluator.o)
cuda-run-lib.lib = lib/libcudarun.so
$(cuda-run-lib.objs): %.o:%.c
	$(CC) -I$(CUDA-PREFIX)/include -Ilibphi -Ilibcudarun -Idevice -fPIC -c -o $@ $<
$(cuda-run-lib.private-objs): %-cuda-run-lib.o: %.c
	$(CC) -I$(CUDA-PREFIX)/include -Ilibphi -Ilibcudarun -Idevice -fPIC -c -o $@ $<
$(cuda-run-lib.lib): $(cuda-run-lib.objs) $(cuda-run-lib.private-objs)
	$(CC) -L$(CUDA-PREFIX)/lib -lcudart -lcuda --shared -o $@ $^

cuda-run-daemon.objs = $(patsubst %.c,%.o,$(wildcard daemon/*.c)) $(patsubst %.c,%.o,$(wildcard libphi/*.c)) $(patsubst %.c,%.o,$(wildcard device/*.c))
cuda-run-daemon.exe = bin/cudarun_daemon
$(cuda-run-daemon.objs): %.o:%.c
	$(CC) -DDAEMON -I$(CUDA-PREFIX)/include -I/usr/local/include -Idaemon -Ilibphi -Idevice -c -o $@ $<
$(cuda-run-daemon.exe): $(cuda-run-daemon.objs)
	$(CC) -L$(CUDA-PREFIX)/lib -lcudart -lcuda -L/usr/local/lib -lcheck -o $@ $^

cuda-run-example.objs = cuda-example/example.o
cuda-run-example.exe = bin/example
$(cuda-run-example.objs): %.o:%.cu
	$(NVCC) -c -o $@ $<
$(cuda-run-example.exe): $(cuda-run-example.objs)
	$(NVCC) -o $@ $^

pkg: $(prj.pkg)
	tar jcf $(prj.name)-$(prj.version)-pkg.tar.bz2 $^
test: $(prj.pkg) $(cuda-run-example.exe)
	$(cuda-run-exec-script) run $(cuda-run-example.exe)
profile: $(prj.pkg) $(cuda-run-example.exe)
	$(cuda-run-profile-script) $(cuda-run-example.exe)

doc:
	doxygen doc/doxygen.conf
	tar jcf reference.tar.bz2 doc/reference
	pdflatex -output-directory=doc doc/readme.tex
	evince doc/readme.pdf
	# lftp xhf:a@10.10.22.251 -e "put readme.tar.bz2; exit"
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf libphi/*.o device/*.o daemon/*.o $(cuda-run-daemon.exe) libcudarun/*.o $(cuda-run-lib.lib) cuda-example/*.o $(cuda-run-example.exe) \
                 *.linkinfo .dep doc/html doc/readme.aux doc/readme.log doc/readme.out doc/readme.pdf $(prj.name)-$(prj.version)-pkg.tar.bz2
bak: clean
	(top_dir=$(shell basename `pwd`); cd ..; tar jcf $(prj.name)-$(prj.version)-$(shell date +%Y%m%d).tar.bz2 $$top_dir)
dep:
	(for f in $(shell find -name "*.c"); do $(CPP) $(CPPFLAGS) -M -MM -MG -MT $${f/.c/.o} $$f;done; \
         for f in $(cuda-run-lib.private-objs); do $(CPP) $(CPPFLAGS) -M -MM -MG -MT $$f $${f/-cuda-run-lib.o/.c};done;) >.dep
-include .dep