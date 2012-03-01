cc = gcc
cxx = g++
cflags = -lpthread -g -std=c99 -Wall -Werror
cxxflags = -lpthread -g -Wall -Werror
target = rbtree
#all: insert_delete_flow-16.view
all: profile
%.view: %.png
	feh $<
%.png: %.dot
	dot $< -Tpng -o $@
%.dot: n=$(lastword $(subst -, ,$*))
%.dot: cmd=$(firstword $(subst -, ,$*))
%.dot: $(target).exe
	./$< dump_$(cmd) $(n) | ./plot.py |tee $@
profile: $(target).exe
	./$< $@ 655360
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
