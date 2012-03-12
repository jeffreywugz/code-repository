target = notify
cc = gcc
cxx = g++
cflags = -lpthread -std=c99
cxxflags = -lpthread

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
notify.args = 20