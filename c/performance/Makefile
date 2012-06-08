target = memcpy
memcpy.args = 1024000000
memcmp.args = 
mem-param.args = 1024 8 1000000
gcc-opt.args = 
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O3 -D_GNU_SOURCE
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
