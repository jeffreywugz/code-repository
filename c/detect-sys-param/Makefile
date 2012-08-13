target = mac
cc = gcc
cxx = g++
cflags = -lpthread -std=c99
cxxflags = -lpthread

all: mac.run
%.run: %.exe
	./$< $(args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
%.data: %.exe
	./$< $($*.args) 2>$@
%.png: %.data plot.py
	./plot.py diff_plot_file $($*.pat) $@ $< xlabel='n_threads' ylabel='time to create threads(us/40threads)'
thread-limit.args = 8000
thread-limit.pat = 'thread\[(\d+)\].*time=(\d+)'
