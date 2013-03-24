target = merge
merge.args = 1024
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
