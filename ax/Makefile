target = ax
gcc-bin-dir = /home/yuanqi.xhf/gcc/bin
run-args = echo_server 5678
cxx = $(gcc-bin-dir)/g++
ld = $(gcc-bin-dir)/g++
# cxx = g++
# ld = g++
cpp = $(cxx) -E 
cxxflags = -Wall -Werror -g -fno-inline-functions
ldflags = -g -lpthread
.PHONY: all run clean dep

srcs= $(wildcard *.cpp)
objs= $(patsubst %.cpp,%.o,$(srcs))

all: test
test: $(target)
	./$(target) $(run-args)
$(target): $(objs)
	$(ld) -o $@ $(ldflags) $^
%.o: %.cpp
	$(cxx) -c  $(cxxflags) $<
clean:
	rm -rf $(target) *.o *~ .dep
dep:
	(for f in $(srcs); do echo -n ""; $(cpp) -M -MM -MG $$f; done) >> .dep

### Dependencies:
-include .dep
