target = malloc-align
malloc-align.args = 1024000
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
