target = perf-test
perf-test.env = n_thread=4 time_limit=3
perf-test.args = getus
gcc-bin-dir = /home/yuanqi.xhf/gcc/bin
run-args = format workdir 10.232.36.4:5678
cxx = $(gcc-bin-dir)/g++
ld = $(gcc-bin-dir)/g++
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
cxxflags = -g -lpthread -lrt -Wall -O2 --no-strict-aliasing

all: $(target).run
%.run: %.exe
	$($*.env) ./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
