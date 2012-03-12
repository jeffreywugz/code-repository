target = ticket-lock
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
notify.args = 24
seq-lock.args = 24
ticket-lock.args = 16