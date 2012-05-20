target = mem-param
mem-param.args = 1024 8 1000000
gcc-opt.args = 
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O2 -D_GNU_SOURCE
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
