target = test_btree
gcc-bin-dir = /home/yuanqi.xhf/gcc/bin
cxx = $(gcc-bin-dir)/g++
ld = $(gcc-bin-dir)/g++
cxxflags = -g -O2 -lpthread -lrt -Wall --no-strict-aliasing
all: $(target).test
test_btree.test: test_btree
	./$<
test_btree: test_btree.cpp allocator.h btree.h
	$(cxx) $(cxxflags) -o $@ $<
