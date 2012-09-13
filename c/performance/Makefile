target = queue-with-sem
queue-with-sem.args = 1 1 10000000
rol.args = 
queue.args = 2 10000000
profile-frame.args = 4 13
sem-vs-pthread-signal.args = 24 1
cas-vs-atomic-add.args = 2 100000000
fadvise-seq.args = 10240000000
memcpy.args = 1024000000
memcmp.args = 
mem-param.args = 1024 8 1000000
gcc-opt.args = 
cc = gcc
cxx = g++
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
cxxflags = -g -O2 -lpthread -lrt

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
