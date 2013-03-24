target = scan-dir
scan-dir.args = pool 10
cc = gcc
cxx = g++
cflags = -lpthread -lrt -std=c99 -g -D_GNU_SOURCE
cxxflags = -g  -lpthread -lrt

all: $(target).run
%.run: %.exe
	rm -rf pool && mkdir -p pool && touch pool/{1..10000} && sync
	scan_type=getdents ./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
