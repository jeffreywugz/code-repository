target = load-gen
load-gen.args = cpu 4
cc = gcc
cxx = g++
cflags = -lpthread -std=c99
cxxflags = -lpthread
all: $(target).test
%.test: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $^
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $^
