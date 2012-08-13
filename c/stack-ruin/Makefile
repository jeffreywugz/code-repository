target = stack-ruin
stack-ruin.args = 1024
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O3 -D_GNU_SOURCE
cxxflags = -lpthread

all: $(target).exe
%.run: %.exe
	true
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
