target = memset
args = 1024 1024
cc = gcc
cxx = g++
cflags = -lpthread -std=c99
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	./$< $(args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
