target = par-queue
par-queue.args = 4 1024 4 102400
cc = gcc
cxx = g++
cflags = -g -lpthread -std=c99 -O3 -D_GNU_SOURCE
cxxflags = -g -lpthread

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
