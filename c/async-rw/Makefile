target = async-rw
async-rw.args = 1024000
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O3 -D_GNU_SOURCE
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	rm -rf *.tmp && touch async-rw.tmp
	#use_append_mode=true ./$< $($*.args)
	write_use_buffered_io=true read_use_buffered_io=true ./$< $($*.args)
	#read_use_buffered_io=true ./$< $($*.args)
	#./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
