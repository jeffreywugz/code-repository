target = next-pow2
next-pow2.args = 10240
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O2
cxxflags = -lpthread -O2
all: $(target).test
%.test: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $^
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $^
