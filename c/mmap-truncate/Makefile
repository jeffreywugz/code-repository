target = mmap-truncate
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -D_GNU_SOURCE
cxxflags = -lpthread
all: $(target).test
%.test: %.exe
	./$<
%.exe: %.c
	$(cc) $(cflags) -o $@ $^
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $^
