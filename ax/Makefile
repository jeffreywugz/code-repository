target = ax
gcc-bin-dir = /home/yuanqi.xhf/gcc/bin
run-args = format workdir 10.232.36.4:5678
cxx = $(gcc-bin-dir)/g++
ld = $(gcc-bin-dir)/g++
cpp = $(cxx) -E 
cxxflags = -Wall -Werror -g -fno-strict-aliasing
ldflags = -g -lpthread
.PHONY: all run clean dep

srcs = $(wildcard *.cpp)
exes = $(patsubst %.cpp,%,$(srcs))

all: test
test: $(target)
	./$(target) $(run-args)
$(exes): %: %.cpp
	$(cxx) -o $@ $(cxxflags) $(ldflags) $<
clean:
	rm -rf $(target) *.o *~ .dep
dep:
	rm -f .dep && (for f in $(srcs); do $(cpp) -M -MM -MG $$f |sed 's/\.o:/:/'; done) >> .dep

### Dependencies:
-include .dep
