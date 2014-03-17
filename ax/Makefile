target = echo_server
gcc-bin-dir = /home/yuanqi.xhf/gcc/bin
run-args = 5678 1
cxx = $(gcc-bin-dir)/g++
ld = $(gcc-bin-dir)/g++
cpp = $(cxx) -E 
cxxflags = -Wall -Werror -g -O2 -fno-strict-aliasing
ldflags = -g -lpthread -O2
.PHONY: all run clean dep

srcs = $(wildcard *.cpp)
exes = $(patsubst %.cpp,%,$(srcs))

all: test
test: $(target)
	./$(target) $(run-args)
$(exes): %: %.cpp
	$(cxx) -o $@ $(cxxflags) $(ldflags) $^
clean:
	rm -rf $(target) *.o *~ .dep
dep:
	(for f in $(srcs); do echo -n ""; $(cpp) -M -MM -MG $$f; done) >> .dep

### Dependencies:
-include .dep
