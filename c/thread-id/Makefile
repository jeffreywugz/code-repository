target = thread-id
cc = gcc
cxx = g++
cflags = -g -lpthread -std=c99 -D_GNU_SOURCE
cxxflags = -g -lpthread

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
