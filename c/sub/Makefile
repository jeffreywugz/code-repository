target = async-rw
async-rw.args = 1024
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -O3 -D_GNU_SOURCE
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	rm -rf *.tmp && touch async-rw.tmp
	wait_fsync=false n_thread=2 use_append_mode=true write_use_direct_io=true read_use_direct_io=true ./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
