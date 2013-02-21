target = static-var
cc = gcc
cxx = g++
cflags = -lpthread -std=c99
cxxflags = -lpthread
all: $(target).test
%.test: %.exe
	./$<
%.exe: %.c
	$(cc) $(cflags) -o $@ $^
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $^
compiler-out-of-order.exe: compiler-out-of-order.c compiler-out-of-order2.c
rsync:
	#scp * yuanqi.xhf@10.232.35.40:scratch/