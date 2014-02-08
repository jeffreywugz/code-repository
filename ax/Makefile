target = ax
cxx = g++
ld = g++
cpp = cpp
cxxflags = -Wall -Werror -g
ldflags = -lpthread
.PHONY: all run clean  dep

srcs= $(wildcard *.cpp)
objs= $(patsubst %.cpp,%.o,$(srcs))

all: $(target)
$(target): $(objs)
	$(ld) -o $@ $(ldflags) $^
%.o: %.cpp
	$(cxx) -c  $(cxxflags) $<
run: $(target)
	./$(target)
clean:
	rm -rf $(target) *.o *~ .dep
dep:
	(for f in $(srcs);do echo -n "";$(cpp) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
