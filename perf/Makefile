target = perf-test
perf-test.env = n_thread=4 time_limit=3
perf-test.args = getus
cc = gcc
cxx = g++
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
cxxflags = -g -lpthread -lrt -Wall -O2 --no-strict-aliasing --no-omit-frame-pointer

all: $(target).run
%.run: %.exe
	$($*.env) ./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
