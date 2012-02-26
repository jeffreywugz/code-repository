cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O3
cxxflags = -lpthread -O3
target = rbtree
all: $(target).exe
	(./$< 4 | ./plot.py |tee a.dot |dot -Tpng -o a.png ) && feh a.png
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
